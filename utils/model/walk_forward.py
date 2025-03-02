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
        initial_train_size: int = 43200 * 6,  
        step_size: int = 43200,  
        threshold: float = 0.613
    ):
        self.initial_train_size = initial_train_size    # 6 months default initial train size
        self.step_size = step_size                      # 1 month default step size
        self.threshold = threshold                      # 0.613 default threshold 
        self.metrics_history = []
        
    def validate(
        self,
        model: Any,         # Model object
        X: pd.DataFrame,    # Features as DataFrame
        y: pd.Series,       # Labels as Series
        timestamp_col: str = 'timestamp',   # Name of timestamp column
        print_results: bool = True          # Whether to print validation results
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]: # Return type is tuple of predictions array, true labels array, and metrics history
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
        if timestamp_col in X.columns:  # Sort by timestamp if available
            X = X.sort_values(by=timestamp_col)             # Features are sorted by timestamp
            X_features = X.drop(columns=[timestamp_col])    # Drop timestamp column
        else:
            X_features = X  # If no timestamp, features are used as is (don't ever see this happening tbh)
            
        aggregated_predictions = []     # Store all predictions in a list
        aggregated_true_labels = []     # Store all true labels in a list
        self.metrics_history = []       # Store metrics history in a list
        
        for i in tqdm(range(self.initial_train_size, len(X), self.step_size)):  
            # Loop through data and visualize progress using tqdm
            # step_size increments starting from initial_train_size, until end of data len(X)

            # Split data
            X_train = X_features.iloc[:i]                   # Train data is from start to i
            y_train = y.iloc[:i]                            # Train labels are from start to i 
            X_test = X_features.iloc[i:i+self.step_size]    # Test data is from i to i+step_size
            y_test = y.iloc[i:i+self.step_size]             # Test labels are from i to i+step_size
            
            # Train and predict
            model.fit(X_train, y_train)                                     # Fit model on train data
            proba_preds = model.predict_proba(X_test)[:, 1]                 # Predict probabilities on test data
            binary_preds = (proba_preds >= self.threshold).astype(int)      # Convert probabilities to binary using threshold
            
            # Store predictions
            aggregated_predictions.extend(proba_preds)  # Store probabilities in aggregated_predictions list
            aggregated_true_labels.extend(y_test)       # Store true labels in aggregated_true_labels list
            
            # Calculate metrics for this iteration
            metrics = {
                'iteration': len(self.metrics_history) + 1, # Iteration number
                'train_size': len(X_train),                 # Size of training data
                'test_size': len(X_test),                   # Size of test data
                'accuracy': accuracy_score(y_test, binary_preds),                      
                'precision': precision_score(y_test, binary_preds, zero_division=0),
                'recall': recall_score(y_test, binary_preds, zero_division=0),
                'f1': f1_score(y_test, binary_preds, zero_division=0),
                'precision_0': precision_score(y_test, binary_preds, pos_label=0, zero_division=0),
                'precision_1': precision_score(y_test, binary_preds, pos_label=1, zero_division=0),
                'recall_0': recall_score(y_test, binary_preds, pos_label=0, zero_division=0),
                'recall_1': recall_score(y_test, binary_preds, pos_label=1, zero_division=0),
                'f1_0': f1_score(y_test, binary_preds, pos_label=0, zero_division=0),
                'f1_1': f1_score(y_test, binary_preds, pos_label=1, zero_division=0)
            }
            self.metrics_history.append(metrics)
            
        predictions = np.array(aggregated_predictions)  # Convert aggregated_predictions list to numpy array
        true_labels = np.array(aggregated_true_labels)  # Convert aggregated_true_labels list to numpy array
        
        if print_results:
            self.print_validation_results(predictions, true_labels)
            
        return predictions, true_labels, self.metrics_history   # Return predictions, true labels, and metrics history
    
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