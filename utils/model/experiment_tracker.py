import sys
import os
# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from db.classes import Experiment, Results  # Database models
from db.db import SessionLocal            # Database session factory
from typing import Dict, List, Optional   # Type hints
from datetime import datetime
import json # For serializing hyperparameters and features

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

    def __exit__(self):
        """Ensures database connection is closed after usage."""
        self.db.close()

    def get_or_create_experiment(
        self,
        symbol: str,          # Trading pair symbol (e.g., "BTCUSDT")
        frequency: str,       # Data timeframe (e.g., "1h", "4h")
        threshold: float,     # Decision threshold for classification
        target_variable: str, # Target to predict (e.g., "pump_1h")
        hyperparameters: Dict,# Model hyperparameters
        features: List[str]   # Feature names used in model
    ):
        """
        Retrieves existing experiment or creates new one if not found.
        Returns db experiment object.
        Prevents duplicate experiments in db by checking all parameters match exactly.
        """

        hyperparameters_json = json.dumps(hyperparameters, sort_keys=True)
        features_json = json.dumps(features, sort_keys=True)

        
        # Bundle all parameters into a config dictionary
        config = {
            'symbol': symbol,
            'frequency': frequency,
            'threshold': threshold,
            'target_variable': target_variable,
            'hyperparameters': hyperparameters_json,
            'features': features_json
        }

        # Query database for matching experiment
        existing = (
            self.db.query(Experiment)
            .filter(
                Experiment.symbol == symbol,
                Experiment.frequency == frequency,
                Experiment.threshold == threshold,
                Experiment.target_variable == target_variable,
                Experiment.hyperparameters == hyperparameters_json,
                Experiment.features == features_json
            ).first()
        )

        if existing:
            print("Found existing experiment")
            return existing

        # Create new experiment if none found
        experiment = Experiment(
            timestamp=datetime.now(datetime.timezone.utc),  # Record creation time
            **config  # Unpack config dictionary as kwargs
        )
        
        self.db.add(experiment)
        self.db.commit()
        print(f"Created new experiment with ID: {experiment.experiment_id}")
        return experiment

    def log_results(
        self,
        experiment: Experiment,           # Reference to existing experiment
        metrics: Dict[str, float],       # Performance metrics
        walk_forward: bool = False,      # Whether walk-forward testing was used
        initial_train_window: Optional[int] = None,  # Initial training window size
        step: Optional[int] = None,      # Step size between iterations
        test_window: Optional[int] = None # Test window size
    ):
        """
        Logs experiment results to database.
        Returns db results object.
        Supports both simple train/test splits and walk-forward testing scenarios.
        """
        
        results = Results(
            experiment_id=experiment.experiment_id,
            walk_forward=walk_forward,
            initial_train_window=initial_train_window,
            step=step,
            test_window=test_window,
            **metrics  # Unpack metrics dictionary
        )
        
        self.db.add(results)
        self.db.commit()
        print(f"Logged results for experiment {experiment.experiment_id}")
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