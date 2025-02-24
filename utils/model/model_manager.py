import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.isotonic import IsotonicRegression
import os
import shap

class ModelManager:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        """Initialize ModelManager with basic parameters."""
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize attributes
        self.df = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.model = None
        self.model_main = None
        self.calibrator = None
        self.optimal_threshold = None
        self.y_pred_calibrated = None
        
    def load_data(self, custom_path: str = None) -> pd.DataFrame:
        """Load data from default location or custom path."""
        if custom_path:
            self.df = pd.read_csv(custom_path, parse_dates=['timestamp'])
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            data_filename = f'{self.symbol}_1hr_window_labels_{self.start_date}_to_{self.end_date}.csv'
            data_path = os.path.join(project_root, 'data', data_filename)
            self.df = pd.read_csv(data_path, parse_dates=['timestamp'])
        print("🕵️♂️ Dataset loaded successfully")
        return self.df
    
    def prepare_features(self, drop_columns: list = None, fill_method: str = 'ffill') -> tuple:
        """Prepare features with configurable options."""
        drop_cols = ['label', 'timestamp'] if drop_columns is None else drop_columns
        X = self.df.drop(columns=drop_cols)
        y = self.df['label']
        
        if fill_method == 'ffill':
            X = X.ffill().fillna(0)
        elif fill_method == 'mean':
            X = X.fillna(X.mean())
            
        return X, y
    
    def split_data(self, test_window_days: int = 100) -> tuple:
        """Split data with configurable test window."""
        X, y = self.prepare_features()
        split_date = self.df['timestamp'].max() - pd.Timedelta(days=test_window_days)
        train_idx = self.df['timestamp'] < split_date
        test_idx = self.df['timestamp'] >= split_date
        
        self.X_train, self.X_test = X[train_idx], X[test_idx]
        self.y_train, self.y_test = y[train_idx], y[test_idx]
        print("📊 Data split completed")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def configure_model(self, custom_params: dict = None) -> LGBMClassifier:
        """Configure model with default or custom parameters."""
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'colsample_bytree': 0.9,
            'subsample': 0.8,
            'subsample_freq': 5,
            'verbosity': -1,
            'n_estimators': 1000,
            'early_stopping_round': 200,
            'eval_metric': 'auc'
        }
        
        if custom_params:
            default_params.update(custom_params)
            
        self.model = LGBMClassifier(**default_params)
        return self.model
    
    def fit_and_evaluate(self, weight_multiplier: float = 1, class_ratio_multiplier: float = 1):
        """Train model with sample weights and class balancing."""
        print("\n🏋️ Training model...")
        
        class_ratio = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        self.model.set_params(scale_pos_weight=class_ratio * class_ratio_multiplier)
        
        sample_weights = np.where(self.y_train == 1, weight_multiplier, 1)
        
        self.model.fit(
            self.X_train, self.y_train,
            sample_weight=sample_weights,
            eval_set=[(self.X_test, self.y_test)]
        )
        return self.model
    
    def calibrate_probabilities(self, test_size: float = 0.2, weight_multiplier: float = 2):
        """Calibrate probability predictions."""
        print("\n🔧 Calibrating probabilities...")
        
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            self.X_train, self.y_train, test_size=test_size, shuffle=False
        )
        
        sample_weights = np.where(self.y_train == 1, weight_multiplier, 1)
        n_main = len(X_train_main)
        sample_weights_main = sample_weights[:n_main]
        sample_weights_val = sample_weights[n_main:]
        
        model_params = self.model.get_params().copy()
        for param in ['early_stopping_round', 'eval_metric', 'eval_set']:
            model_params.pop(param, None)
            
        self.model_main = LGBMClassifier(**model_params).fit(
            X_train_main, y_train_main,
            sample_weight=sample_weights_main
        )
        
        val_probs = self.model_main.predict_proba(X_val)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(val_probs, y_val, sample_weight=sample_weights_val)
        
        self.y_pred_calibrated = self.calibrator.transform(
            self.model_main.predict_proba(self.X_test)[:, 1]
        )
        return self.y_pred_calibrated
    
    def tune_threshold(self, min_recall: float = 0.5, min_precision: float = 0.65) -> tuple:
        """Find optimal threshold based on precision-recall trade-off."""
        print("\n🎯 Tuning threshold...")
        
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, self.y_pred_calibrated)
        viable = np.where((recalls[:-1] >= min_recall) & (precisions[:-1] >= min_precision))[0]
        
        if len(viable) > 0:
            best_idx = viable[np.argmax(precisions[viable])]
            self.optimal_threshold = thresholds[best_idx]
            return self.optimal_threshold, precisions[best_idx], recalls[best_idx]
        
        self.optimal_threshold = 0.5
        return 0.5, precisions[-1], recalls[-1]
    
    def plot_diagnostics(self):
        """Generate diagnostic visualizations."""
        print("\n📈 Generating diagnostic plots...")
        
        self.plot_threshold_tuning()
        self.plot_shap_analysis()
        self.plot_feature_importance()
    
    def plot_threshold_tuning(self):
        """Plot threshold tuning diagnostic."""
        print("\n📊 Plotting threshold tuning diagnostics...")
        
        precisions, recalls, thresholds = precision_recall_curve(
            self.y_test, self.y_pred_calibrated)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions[:-1], label='Precision')
        plt.plot(thresholds, recalls[:-1], label='Recall')
        plt.axvline(self.optimal_threshold, color='red', 
                   linestyle='--', 
                   label=f'Chosen Threshold ({self.optimal_threshold:.3f})')
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Precision-Recall Tradeoff")
        plt.legend()
        plt.show()

    def plot_shap_analysis(self, plot_type='summary'):
        """Plot SHAP value analysis with different options."""
        print("\n🎯 Generating SHAP analysis...")
        
        explainer = shap.TreeExplainer(self.model_main)
        shap_values = explainer.shap_values(self.X_test)
        
        if plot_type == 'summary':
            shap.summary_plot(shap_values, self.X_test)
        elif plot_type == 'bar':
            shap.summary_plot(shap_values, self.X_test, plot_type='bar')
        elif plot_type == 'dependence':
            # Get most important feature
            feature_importance = np.abs(shap_values).mean(0)
            most_important = self.X_test.columns[feature_importance.argmax()]
            shap.dependence_plot(most_important, shap_values, self.X_test)

    def plot_feature_importance(self):
        """Plot traditional feature importance."""
        print("\n📈 Plotting feature importance...")
        
        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model_main.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(importance['feature'][:20], importance['importance'][:20])
        plt.xticks(rotation=45, ha='right')
        plt.title("Top 20 Feature Importance")
        plt.tight_layout()
        plt.show()
    
    def get_metrics(self) -> dict:
        """Calculate and display all relevant metrics."""
        y_pred_class = (self.y_pred_calibrated >= self.optimal_threshold).astype(int)
        
        # Get full classification report as dict
        report = classification_report(
            self.y_test, y_pred_class, 
            digits=4,
            target_names=['Class 0', 'Class 1'],
            output_dict=True  # This gives us a dictionary instead of string
        )
        
        metrics = {
            # Overall metrics
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'auc_roc': roc_auc_score(self.y_test, self.y_pred_calibrated),
            'optimal_threshold': self.optimal_threshold,
            
            # Class 0 specific metrics
            'precision_0': report['Class 0']['precision'],
            'recall_0': report['Class 0']['recall'],
            'f1_0': report['Class 0']['f1-score'],
            
            # Class 1 specific metrics
            'precision_1': report['Class 1']['precision'],
            'recall_1': report['Class 1']['recall'],
            'f1_1': report['Class 1']['f1-score']
        }
        
        # Print formatted report
        print("\n📊 Model Performance Report")
        print("=" * 50)
        print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print(f"AUC-ROC Score: {metrics['auc_roc']:.4f}")
        print("\nOverall Metrics:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print("\nClass 0 Metrics:")
        print(f"Precision: {metrics['precision_0']:.4f}")
        print(f"Recall:    {metrics['recall_0']:.4f}")
        print(f"F1-Score:  {metrics['f1_0']:.4f}")
        print("\nClass 1 Metrics:")
        print(f"Precision: {metrics['precision_1']:.4f}")
        print(f"Recall:    {metrics['recall_1']:.4f}")
        print(f"F1-Score:  {metrics['f1_1']:.4f}")
        
        return metrics
    
    def save_model(self, custom_path: str = None):
        """Retrain on full dataset and save the model."""
        print("\n🔄 Retraining model on full dataset...")
        
        # Get all features and retrain
        X_full, y_full = self.prepare_features()
        model_params = self.model.get_params()
        self.model_main = LGBMClassifier(**model_params)
        self.model_main.fit(X_full, y_full)
        
        # Calibrate on full dataset
        probs = self.model_main.predict_proba(X_full)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(probs, y_full)
        
        # Save model
        if not custom_path:
            os.makedirs('models', exist_ok=True)
            custom_path = f'models/{self.symbol}_pump_predictor.pkl'
        
        joblib.dump((self.model_main, self.calibrator), custom_path)
        print(f"\n💾 Model saved to {custom_path}")
        return custom_path

    def retrain_full(self):
        """Retrain model on full dataset before saving."""
        print("\n🔄 Retraining model on full dataset...")
        
        # Get all features
        X_full, y_full = self.prepare_features()
        
        # Configure model with same parameters
        model_params = self.model.get_params()
        self.model_main = LGBMClassifier(**model_params)
        
        # Train on full dataset
        self.model_main.fit(X_full, y_full)
        
        # Calibrate on full dataset
        probs = self.model_main.predict_proba(X_full)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(probs, y_full)

if __name__ == "__main__":
    symbol = "PEPEUSDT"
    start_date = "2023-05-20"
    end_date = "2025-02-15"
    
    manager = ModelManager(symbol, start_date, end_date)
    manager.load_data()
    manager.split_data()
    manager.configure_model()
    manager.fit_and_evaluate()
    manager.calibrate_probabilities()
    manager.tune_threshold()
    manager.plot_diagnostics()
    metrics = manager.get_metrics()
    print(metrics['classification_report'])
    print(f"AUC-ROC: {metrics['auc_roc']:.2f}")
    manager.save_model()