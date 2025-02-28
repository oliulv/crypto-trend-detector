import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm

class WalkForwardAnalyzer:
    def __init__(
        self,
        initial_train_size: int = 43200 * 6,  # 6 months default
        step_size: int = 43200,  # 1 month default
        threshold: float = 0.613
    ):
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.threshold = threshold
        self.metrics_history = []
        
    def validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        timestamp_col: str = 'timestamp',
        print_results: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Perform walk-forward validation with tracking of metrics over time.
        
        Args:
            model: The model to validate
            X: Feature DataFrame
            y: Target series
            timestamp_col: Name of timestamp column
            print_results: Whether to print validation results
            
        Returns:
            Tuple containing:
            - Array of predictions
            - Array of true labels
            - List of metrics dictionaries for each iteration
        """
        if timestamp_col in X.columns:
            X = X.sort_values(by=timestamp_col)
            X_features = X.drop(columns=[timestamp_col])
        else:
            X_features = X
            
        aggregated_predictions = []
        aggregated_true_labels = []
        self.metrics_history = []
        
        for i in tqdm(range(self.initial_train_size, len(X), self.step_size)):
            # Split data
            X_train = X_features.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X_features.iloc[i:i+self.step_size]
            y_test = y.iloc[i:i+self.step_size]
            
            # Train and predict
            model.fit(X_train, y_train)
            proba_preds = model.predict_proba(X_test)[:, 1]
            binary_preds = (proba_preds >= self.threshold).astype(int)
            
            # Store predictions
            aggregated_predictions.extend(proba_preds)
            aggregated_true_labels.extend(y_test)
            
            # Calculate metrics for this iteration
            metrics = {
                'iteration': len(self.metrics_history) + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': accuracy_score(y_test, binary_preds),
                'precision': precision_score(y_test, binary_preds, zero_division=0),
                'recall': recall_score(y_test, binary_preds, zero_division=0),
                'f1': f1_score(y_test, binary_preds, zero_division=0)
            }
            self.metrics_history.append(metrics)
            
        predictions = np.array(aggregated_predictions)
        true_labels = np.array(aggregated_true_labels)
        
        if print_results:
            self.print_validation_results(predictions, true_labels)
            
        return predictions, true_labels, self.metrics_history
    
    def print_validation_results(self, predictions: np.ndarray, true_labels: np.ndarray):
        """
        Print comprehensive validation results including overall metrics and per-iteration averages.
        """
        # Convert probability predictions to binary using threshold
        binary_predictions = (predictions >= self.threshold).astype(int)
        
        print("\n📊 Overall Aggregated Validation Results:")
        # Overall metrics
        metrics = {
            'Accuracy': accuracy_score(true_labels, binary_predictions),
            'Precision': precision_score(true_labels, binary_predictions, zero_division=0),
            'Recall': recall_score(true_labels, binary_predictions, zero_division=0),
            'F1-Score': f1_score(true_labels, binary_predictions, zero_division=0),
            'AUC-ROC': roc_auc_score(true_labels, predictions)
        }
        
        for metric, value in metrics.items():
            print(f"{metric:.<15} {value:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, binary_predictions)
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0,0]:<6} FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]:<6} TP: {cm[1,1]}")
        
        # Classification Report
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, binary_predictions, digits=4))
        
        # Per-iteration averages
        print("\n📈 Average Metrics Across All Iterations:")
        metrics_df = pd.DataFrame(self.metrics_history)
        avg_metrics = metrics_df[['accuracy', 'precision', 'recall', 'f1']].mean()
        
        for metric, value in avg_metrics.items():
            print(f"{metric.capitalize():.<15} {value:.4f}")
        
        # Training size progression
        print("\n📊 Training Data Size Progression:")
        print(f"Initial: {metrics_df['train_size'].iloc[0]:,} samples")
        print(f"Final:   {metrics_df['train_size'].iloc[-1]:,} samples")
        print(f"Steps:   {len(metrics_df)} iterations")

    def plot_metrics_over_time(self):
        """Plot how metrics evolve as training data size increases."""
        metrics_df = pd.DataFrame(self.metrics_history)
        
        plt.figure(figsize=(12, 6))
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plt.plot(metrics_df['train_size'], metrics_df[metric], label=metric)
        
        plt.xlabel('Training Data Size')
        plt.ylabel('Score')
        plt.title('Model Performance Metrics Over Time')
        plt.legend()
        plt.grid(True)
        return plt