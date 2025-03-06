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
        initial_train_ratio: float = 0.3,  # 30% of data for initial training
        step_ratio: float = 0.1,           # 10% of data for each step
        threshold: float = 0.613
    ):
        """
        Initialize WalkForwardAnalyzer with ratio-based window sizes.
        
        Args:
            initial_train_ratio: Ratio of total data to use for initial training (0-1)
            step_ratio: Ratio of total data to use for each step (0-1)
            threshold: Classification threshold for binary predictions
        """
        # Validate ratios are between 0 and 1
        if not 0 < initial_train_ratio < 1:
            raise ValueError("initial_train_ratio must be between 0 and 1")
        if not 0 < step_ratio < 1:
            raise ValueError("step_ratio must be between 0 and 1")
            
        self.initial_train_ratio = initial_train_ratio
        self.step_ratio = step_ratio
        self.threshold = threshold
        self.metrics_history = []
        
        # These will be set when validate() is called
        self.initial_train_size = None
        self.step_size = None
        self.y = None  # Add this line to store target variable
        
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
        # Store y at the start of validation
        self.y = y  # Add this line

        # Calculate actual sizes based on total data length
        total_size = len(X)
        self.initial_train_size = int(total_size * self.initial_train_ratio)
        self.step_size = int(total_size * self.step_ratio)
        
        # Ensure minimum sizes
        if self.initial_train_size < 100:
            raise ValueError(f"Initial training size too small: {self.initial_train_size} samples")
        if self.step_size < 10:
            raise ValueError(f"Step size too small: {self.step_size} samples")
            
        print(f"\n📊 Walk-Forward Configuration:")
        print(f"Total data points:     {total_size:,}")
        print(f"Initial training size: {self.initial_train_size:,} ({self.initial_train_ratio:.1%})")
        print(f"Step size:            {self.step_size:,} ({self.step_ratio:.1%})")
        print(f"Number of iterations:  {(total_size - self.initial_train_size) // self.step_size}\n")
        
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
                'threshold': self.threshold,                
                'accuracy': accuracy_score(y_test, binary_preds),                      
                'precision': precision_score(y_test, binary_preds, zero_division=0),
                'recall': recall_score(y_test, binary_preds, zero_division=0),
                'f1': f1_score(y_test, binary_preds, zero_division=0),
                'precision_0': precision_score(y_test, binary_preds, pos_label=0, zero_division=0),
                'precision_1': precision_score(y_test, binary_preds, pos_label=1, zero_division=0),
                'recall_0': recall_score(y_test, binary_preds, pos_label=0, zero_division=0),
                'recall_1': recall_score(y_test, binary_preds, pos_label=1, zero_division=0),
                'f1_0': f1_score(y_test, binary_preds, pos_label=0, zero_division=0),
                'f1_1': f1_score(y_test, binary_preds, pos_label=1, zero_division=0),
                'auc_roc': roc_auc_score(y_test, proba_preds)  # Add ROC-AUC calculation
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
        """Plot how Class 1 metrics evolve as training data size increases."""
        metrics_df = pd.DataFrame(self.metrics_history)
        
        plt.figure(figsize=(15, 8))
        
        # Plot Class 1 metrics
        metrics_to_plot = {
            'precision_1': 'Precision (Class 1)',
            'recall_1': 'Recall (Class 1)',
            'f1_1': 'F1 (Class 1)'
        }
        
        for metric, label in metrics_to_plot.items():
            plt.plot(metrics_df['train_size'], metrics_df[metric], label=label)
        
        # Add class distribution line using stored self.y
        if self.y is not None:  # Check if we have the target variable
            class_ratio = metrics_df['test_size'].map(
                lambda x: sum(self.y.iloc[-x:] == 1) / x
            )
            plt.plot(metrics_df['train_size'], class_ratio, 
                    label='Class 1 Ratio', linestyle='--', alpha=0.5)
        
        plt.xlabel('Training Data Size')
        plt.ylabel('Score')
        plt.title('Model Performance Metrics Over Time (Class 1)')
        plt.legend()
        plt.grid(True)
        return plt