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
from typing import List, Dict
from .production_model import ProductionModel


class ModelManager:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        """Initialize ModelManager with basic parameters."""
        self.symbol = symbol            # Trading symbol
        self.start_date = start_date    # Start date for data
        self.end_date = end_date        # End date for data
        
        # Initialize attributes
        self.df = None                      # Dataframe
        self.X_train = self.X_test = None   # Features
        self.y_train = self.y_test = None   # Target Variable
        self.model = None                   # Model
        self.y_pred_proba = None            # Raw probabilities
        self.y_pred_calibrated = None       # Optional calibrated probabilities
        self.optimal_threshold = None       # Optimal threshold (default 0.5)    
        self.calibrator = None              # Optional calibrator

    def load_data(self, custom_path: str = None) -> pd.DataFrame:
        """Load data from custom path."""
        self.df = pd.read_csv(custom_path, parse_dates=['timestamp']) # Load data from csv into our dataframe
        print("🕵️♂️ Dataset loaded successfully")   
        return self.df
    
    def prepare_features(self, drop_columns: list = None, fill_method: str = 'ffill') -> tuple:
        """Prepare features with configurable options."""
        drop_cols = ['label', 'timestamp'] if drop_columns is None else drop_columns # Drop columns label and timestamp by default, unless otherwise specified
        X = self.df.drop(columns=drop_cols) # Drop columns
        y = self.df['label']                # Target variable
        
        if fill_method == 'ffill':  # Fill missing values with forward fill
            X = X.ffill().fillna(0)
        elif fill_method == 'mean': # Fill missing values with mean
            X = X.fillna(X.mean())
            
        return X, y
    
    def split_data(self, test_window_days: int = 100) -> tuple:
        """Split data with configurable test window."""
        X, y = self.prepare_features()                  # Prepare features through our function
        split_date = self.df['timestamp'].max() - pd.Timedelta(days=test_window_days)   # Calculate split date
        train_idx = self.df['timestamp'] < split_date   # Define train data               
        test_idx = self.df['timestamp'] >= split_date   # Define test data             
        
        self.X_train, self.X_test = X[train_idx], X[test_idx]   # Split features into train and test
        self.y_train, self.y_test = y[train_idx], y[test_idx]   # Split target variable into train and test
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
        } # Default parameters
        
        if custom_params:
            default_params.update(custom_params) # Update default parameters with custom parameters
            
        self.model = LGBMClassifier(**default_params) # Initialize model with parameters
        return self.model
    
    def fit_and_evaluate(self, weight_multiplier: float = 2):
        """Train model and store raw probabilities."""
        print("\n🏋️ Training model...")
        
        # Train with sample weights
        sample_weights = np.where(self.y_train == 1, weight_multiplier, 1) # Assign sample weights, weight_multiplier gives our Class 1 more significance.

        self.model.fit(
            self.X_train, self.y_train,             # Train the model
            sample_weight=sample_weights,           # Assign sample weights
            eval_set=[(self.X_test, self.y_test)]   # Evaluate trained model on test set
        )
        
        # Store raw probabilities
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1] # Store raw probabilities
        return self.model                                               # Only storing Class 1, as we only need one class for binary classification
    
    def calibrate_probabilities(self, val_size: float = 0.25):
        """Calibrate probability predictions using existing model."""
        print("\n🔧 Calibrating probabilities...")
        
        # Split off a validation set for calibration
        X_val, y_val = train_test_split(    # Split training data (from first split) into training and validation sets
            self.X_train, self.y_train,     # Get our training data that will be split
            test_size=val_size,             # Set validation size
            shuffle=False,                  # Do not shuffle data (time series data can't be shuffled)   
            return_train=False              # Only return validation set (dont need training set)
        )
        
        # Get probabilities from existing model
        val_probs = self.model.predict_proba(X_val)[:, 1]   # Get validation probabilities from our training data
        
        # Fit calibrator on validation set
        self.calibrator = IsotonicRegression(out_of_bounds='clip')      # Initialize calibrator
        self.calibrator.fit(val_probs, y_val)                           # Fit calibrator on validation probabilities
        
        # Transform test predictions
        self.y_pred_calibrated = self.calibrator.transform(             # Original testing predictions are calibrated
            self.model.predict_proba(self.X_test)[:, 1]                
        )
        
        return self.y_pred_calibrated
    
    def tune_threshold(self, min_recall: float = 0.5, min_precision: float = 0.65) -> tuple:
        """Tune threshold on current predictions."""
        print("\n🎯 Tuning threshold...")
        
        predictions = self.y_pred_calibrated if self.calibrator else self.y_pred_proba      # Store predictions, use calibrated probabilities if calibrator exists
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, predictions)  # Get precision, recall, and thresholds from our testing predictions
        
        viable = np.where((recalls[:-1] >= min_recall) & (precisions[:-1] >= min_precision))[0] # Get viable thresholds based on minimum recall and precision

        if len(viable) > 0: # If there are viable thresholds                
            best_idx = viable[np.argmax(precisions[viable])]    # Get the best index based on the highest precision
            self.optimal_threshold = thresholds[best_idx]       # Set the optimal threshold to the best index
        else:
            self.optimal_threshold = 0.5    # If no viable thresholds, set optimal threshold to 0.5
            
        print(f"Optimal threshold: {self.optimal_threshold:.4f}")
        return self.optimal_threshold
    
    def plot_diagnostics(self):
        """Generate diagnostic visualizations."""
        print("\n📈 Generating diagnostic plots...")
        
        # This functions just calls all our other plotting functions.
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

    def shap_analysis(self, plot_type='summary', show_bottom=False, n_features=20):
        """
        Plot SHAP value analysis with different options and return importance metrics.
        
        Args:
            plot_type (str): Type of plot ('summary', 'bar', 'dependence')
            show_bottom (bool): If True, show least important features
            n_features (int): Number of features to display
            
        Returns:
            dict: Dictionary containing feature importance metrics and values
        """
        print("\n🎯 Generating SHAP analysis...")
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        
        # Calculate feature importance based on SHAP values
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance': feature_importance
        })
        
        total_importance = feature_importance_df['importance'].sum()
        bottom_features = feature_importance_df.nsmallest(n_features, 'importance')
        top_features = feature_importance_df.nlargest(n_features, 'importance')
        bottom_importance = bottom_features['importance'].sum()
        top_importance = top_features['importance'].sum()
        
        # Calculate percentiles for all features
        feature_importance_df['percentile'] = feature_importance_df['importance'].rank(pct=True) * 100
        feature_importance_df['contribution_pct'] = (feature_importance_df['importance'] / total_importance) * 100
        
        # Prepare return dictionary
        importance_dict = {
            'feature': feature_importance_df['feature'].tolist(),
            'shap_importance': feature_importance_df['importance'].tolist(),
            'total_importance': total_importance,
            'percentiles': feature_importance_df['percentile'].tolist(),
            'contribution_pct': feature_importance_df['contribution_pct'].tolist(),
            'metrics': {
                'bottom_importance_pct': (bottom_importance/total_importance*100),
                'top_importance_pct': (top_importance/total_importance*100),
                'importance_ratio': (top_importance/bottom_importance)
            }
        }
        
        # Plot if requested
        if plot_type:
            features_to_plot = bottom_features if show_bottom else top_features
            X_test_filtered = self.X_test[features_to_plot['feature']]
            shap_values_filtered = shap_values[:, features_to_plot.index]
            
            if plot_type == 'summary':
                shap.summary_plot(shap_values_filtered, X_test_filtered)
            elif plot_type == 'bar':
                shap.summary_plot(shap_values_filtered, X_test_filtered, plot_type='bar')
            elif plot_type == 'dependence':
                most_important = features_to_plot.iloc[0]['feature']
                shap.dependence_plot(most_important, shap_values_filtered, X_test_filtered)
        
        return importance_dict

    def feature_importance(self, show_bottom=False, n_features=20):
        """
        Calculate LGBM feature importance and return metrics.
        
        Args:
            show_bottom (bool): If True, show least important features
            n_features (int): Number of features to display
            
        Returns:
            dict: Dictionary containing feature importance metrics and values
        """
        print("\n📈 Calculating feature importance...")
        
        # Create DataFrame with feature importance
        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        total_importance = importance['importance'].sum()
        bottom_features = importance.tail(n_features)
        top_features = importance.head(n_features)
        bottom_importance = bottom_features['importance'].sum()
        top_importance = top_features['importance'].sum()
        
        # Calculate percentiles and contribution percentages
        importance['percentile'] = importance['importance'].rank(pct=True) * 100
        importance['contribution_pct'] = (importance['importance'] / total_importance) * 100
        
        # Prepare return dictionary
        importance_dict = {
            'feature': importance['feature'].tolist(),
            'importance': importance['importance'].tolist(),
            'total_importance': total_importance,
            'percentiles': importance['percentile'].tolist(),
            'contribution_pct': importance['contribution_pct'].tolist(),
            'metrics': {
                'bottom_importance_pct': (bottom_importance/total_importance*100),
                'top_importance_pct': (top_importance/total_importance*100),
                'importance_ratio': (top_importance/bottom_importance)
            }
        }
        
        # Plot if requested
        if show_bottom:
            features_to_plot = bottom_features.iloc[::-1]  # Reverse order for visualization
        else:
            features_to_plot = top_features
        
        plt.figure(figsize=(12, 6))
        plt.bar(features_to_plot['feature'], features_to_plot['importance'])
        plt.xticks(rotation=45, ha='right')
        plt.title(f"{'Bottom' if show_bottom else 'Top'} {n_features} Feature Importance")
        plt.tight_layout()
        plt.show()
        
        return importance_dict
    
    def get_metrics(self) -> dict:
        """Calculate metrics using current model state."""
        predictions = self.y_pred_calibrated if self.calibrator else self.y_pred_proba  # Get predictions, use calibrated probabilities if calibrator exists
        y_pred = (predictions >= self.optimal_threshold).astype(int)            # Get predictions based on optimal threshold
        
        report = classification_report(     # Get classification report
            self.y_test, y_pred,
            digits=5,
            target_names=['Class 0', 'Class 1'],
            output_dict=True
        )
        
        metrics = {     # Store metrics in a dictionary

            # Overall metrics
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'auc_roc': roc_auc_score(self.y_test, predictions),
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
        
        print(f"\nUsing {'calibrated' if self.calibrator else 'raw'} probabilities")
        print(f"Classification threshold: {self.optimal_threshold:.4f}")
        
        return metrics
    
    def get_walk_forward_metrics(self, metrics_history: List[Dict]) -> dict:
        """Calculate aggregated metrics from walk-forward validation results."""
        metrics_df = pd.DataFrame(metrics_history)
        
        # Calculate average metrics across all iterations
        metrics = {
            'accuracy': metrics_df['accuracy'].mean(),
            'precision': metrics_df['precision_1'].mean(),  # Class 1 specific
            'recall': metrics_df['recall_1'].mean(),        # Class 1 specific
            'f1': metrics_df['f1_1'].mean(),               # Class 1 specific
            'auc_roc': metrics_df['auc_roc'].mean() if 'auc_roc' in metrics_df else None,
            'optimal_threshold': self.optimal_threshold,
            
            # Class 0 specific metrics
            'precision_0': metrics_df['precision_0'].mean(),
            'recall_0': metrics_df['recall_0'].mean(),
            'f1_0': metrics_df['f1_0'].mean(),
            
            # Class 1 specific metrics (same as above but renamed for consistency)
            'precision_1': metrics_df['precision_1'].mean(),
            'recall_1': metrics_df['recall_1'].mean(),
            'f1_1': metrics_df['f1_1'].mean()
        }
        
        print("\n📊 Walk-Forward Validation Results")
        print("=" * 50)
        print(f"Metrics averaged across {len(metrics_df)} iterations:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        
        return metrics
    
    def save_model(self, custom_path: str = None):
        """Save production-ready model with all optimizations."""
        print("\n🔄 Preparing production model...")
        
        # Retrain on full dataset with optimizations
        self.retrain_full()
        
        # Create production model with tuned threshold
        prod_model = ProductionModel(
            base_model=self.model,
            threshold=self.optimal_threshold,  # Use tuned threshold
            calibrator=self.calibrator        # Optional calibrator
        )
        
        # Save model
        if not custom_path:
            os.makedirs('models', exist_ok=True)
            custom_path = f'models/{self.symbol}_pump_predictor.pkl'
        
        joblib.dump(prod_model, custom_path)
        print(f"\n💾 Production model saved to {custom_path}")
        print(f"    • Using threshold: {self.optimal_threshold:.4f}")
        print(f"    • Calibration: {'enabled' if self.calibrator else 'disabled'}")
        
        return custom_path

    def retrain_full(self):
        """Retrain model on full dataset using optimized settings."""
        print("\n🔄 Retraining model on full dataset...")
        
        # Get all features
        X_full, y_full = self.prepare_features()
        
        # Configure model with same parameters
        model_params = self.model.get_params()
        self.model = LGBMClassifier(**model_params)
        
        # Train on full dataset
        print("Training final model...")
        sample_weights = np.where(y_full == 1, 2, 1)  # Use same weight multiplier of 2 by default
        self.model.fit(X_full, y_full, sample_weight=sample_weights)
        
        # Optional calibration on full dataset
        if self.calibrator is not None:
            print("Applying calibration to full model...")
            probs = self.model.predict_proba(X_full)[:, 1]
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(probs, y_full)
        
        return self.model

    def get_walk_forward_params(self) -> dict:
        """Get model parameters suitable for walk-forward validation."""
        # Get current model parameters
        params = self.model.get_params()
        
        # Remove validation-specific parameters
        walk_forward_params = {
            k: v for k, v in params.items() 
            if k not in ['early_stopping_round', 'eval_metric', 'callbacks']
        }
        
        return walk_forward_params
    
    def configure_walk_forward_model(self):
        """Configure model for walk-forward validation."""
        walk_forward_params = self.get_walk_forward_params()
        self.model = LGBMClassifier(**walk_forward_params)
        return self.model

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