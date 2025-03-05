import sys
import os
# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from src.db.classes import Experiment, Results  # Database models
from src.db.db import SessionLocal            # Database session factory
from typing import Dict, List, Optional   # Type hints
from datetime import datetime, timezone  # Add timezone to imports

class ExperimentTracker:
    """
    Tracks and logs machine learning experiments to the database.
    Handles creation and retrieval of experiments and their results.
    """
    
    def __init__(self):
        """Creates new database session when instantiated."""
        self.db = SessionLocal()

    # Context manager methods for safe database handling
    def __enter__(self):
        """Allows usage with 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures database connection is closed after usage.
        
        Args:
            exc_type: Type of exception that occurred, if any
            exc_val: Exception instance that occurred, if any
            exc_tb: Traceback if exception occurred
        """
        try:
            self.db.close()
        except Exception as e:
            print(f"Error closing database connection: {e}")
            if exc_type is not None:
                return False  # Re-raise any original exception
        return False  # Don't suppress exceptions

    def get_or_create_experiment(
        self,
        symbol: str,          
        frequency: str,         
        target_variable: str, 
        hyperparameters: Dict,
        features: List[str]   
    ):
        """
        Creates or retrieves an experiment matching database schema exactly.
        """

        # Query using native JSONB operators
        existing = (
            self.db.query(Experiment)
            .filter(
                Experiment.symbol == symbol,
                Experiment.frequency == frequency,
                Experiment.target_variable == target_variable,
                Experiment.hyperparameters == hyperparameters,
                Experiment.features == features
            )
            .first()
        )

        if existing:
            print("Found existing experiment")
            return existing

        # Create new experiment matching schema types
        experiment = Experiment(
            timestamp=datetime.now(timezone.utc),  # Fixed timezone usage
            symbol=symbol,                    # String, not null
            frequency=frequency,              # String, not null
            target_variable=target_variable,  # String, not null
            hyperparameters=hyperparameters,  # JSONB, nullable
            features=features                 # JSONB, not null
        )
        
        self.db.add(experiment)
        self.db.commit()
        print(f"Created new experiment with ID: {experiment.experiment_id}")
        return experiment

    def log_results(
        self,
        experiment: Experiment, 
        metrics: Dict[str, float],
        test_window_days: Optional[int] = None,
        walk_forward: bool = False,
        initial_train_ratio: Optional[int] = None,
        step: Optional[int] = None
    ):
        """Logs experiment results to database if no identical result exists."""
        
        # Check for existing results with same parameters
        existing_result = (
            self.db.query(Results)
            .filter(
                Results.experiment_id == experiment.experiment_id,
                Results.walk_forward == walk_forward,
                Results.initial_train_ratio == initial_train_ratio,
                Results.step == step,
                Results.test_window == test_window_days
            )
            .first()
        )

        if existing_result:
            print(f"Identical result already exists for experiment ID: {experiment.experiment_id}")
            return existing_result

        # Continue with existing logging logic if no duplicate found
        def convert_numpy(value):
            return float(value.item()) if hasattr(value, 'item') else value

        processed_metrics = {
            key: convert_numpy(value)
            for key, value in metrics.items()
        }
        
        results_metrics = {
            # Overall metrics only
            'accuracy': processed_metrics['accuracy'],
            'precision': processed_metrics['precision'],
            'recall': processed_metrics['recall'],
            'f1': processed_metrics['f1'],
            'roc_auc': processed_metrics['auc_roc'],
            'best_threshold': processed_metrics['optimal_threshold'],
            
            # Per-class metrics (no accuracy)
            'precision_0': processed_metrics['precision_0'],
            'recall_0': processed_metrics['recall_0'],
            'f1_0': processed_metrics['f1_0'],
            'precision_1': processed_metrics['precision_1'],
            'recall_1': processed_metrics['recall_1'],
            'f1_1': processed_metrics['f1_1']
        }
        
        results = Results(
            experiment_id=experiment.experiment_id,
            walk_forward=walk_forward,
            initial_train_ratio=initial_train_ratio,
            step=step,
            test_window=test_window_days,
            **results_metrics
        )
        
        self.db.add(results)
        self.db.commit()
        print(f"Logged results for experiment ID: {experiment.experiment_id}")
        return results

    def log_walk_forward_results(
        self,
        experiment: Experiment,
        metrics_history: List[Dict],
        initial_train_ratio: float,
        step_ratio: float,
        test_window_days: Optional[int] = None
    ) -> Results:
        """Log walk-forward validation results to database."""
        
        # Check for existing walk-forward results
        existing_result = (
            self.db.query(Results)
            .filter(
                Results.experiment_id == experiment.experiment_id,
                Results.walk_forward == True
            )
            .first()
        )
        
        if existing_result:
            print(f"Walk-forward results already exist for experiment ID: {experiment.experiment_id}")
            return existing_result
        
        # Calculate aggregated metrics from history
        metrics_df = pd.DataFrame(metrics_history)
        
        # Define conversion helper (same as in log_results)
        def convert_numpy(value):
            return float(value.item()) if hasattr(value, 'item') else value
        
         # Get threshold from metrics_history
        threshold = metrics_history[0].get('threshold', experiment.hyperparameters.get('optimal_threshold', 0.5))
        
        # Calculate and convert metrics
        results_metrics = {
            'accuracy': convert_numpy(metrics_df['accuracy'].mean()),
            'precision': convert_numpy(metrics_df['precision_1'].mean()),
            'recall': convert_numpy(metrics_df['recall_1'].mean()),
            'f1': convert_numpy(metrics_df['f1_1'].mean()),
            'roc_auc': convert_numpy(metrics_df['auc_roc'].mean()),   
            'best_threshold': convert_numpy(threshold),
            'precision_0': convert_numpy(metrics_df['precision_0'].mean()),
            'recall_0': convert_numpy(metrics_df['recall_0'].mean()),
            'f1_0': convert_numpy(metrics_df['f1_0'].mean()),
            'precision_1': convert_numpy(metrics_df['precision_1'].mean()),
            'recall_1': convert_numpy(metrics_df['recall_1'].mean()),
            'f1_1': convert_numpy(metrics_df['f1_1'].mean())
        }
        
        # Create new results entry
        results = Results(
            experiment_id=experiment.experiment_id,
            walk_forward=True,
            initial_train_ratio=convert_numpy(initial_train_ratio),
            step=convert_numpy(step_ratio),
            test_window=test_window_days,
            **results_metrics
        )
        
        self.db.add(results)
        self.db.commit()
        print(f"Logged walk-forward results for experiment ID: {experiment.experiment_id}")
        return results

    def get_experiment_results(self, experiment_id: int) -> List[Results]:
        """Retrieves all results for a specific experiment ID."""
        return (
            self.db.query(Results)
            .filter(Results.experiment_id == experiment_id)
            .all()
        )

    def get_best_experiments(
        self,
        metric: str = 'f1',  # Metric to sort by (default: F1 score)
        n: int = 5          # Number of top experiments to return
    ) -> List[Experiment]:
        """
        Retrieves top N experiments sorted by specified metric.
        Joins experiments with their results to sort by performance.
        """
        return (
            self.db.query(Experiment)
            .join(Results)  # Join with results table
            .order_by(getattr(Results, metric).desc())  # Sort by metric
            .limit(n)
            .all()
        )