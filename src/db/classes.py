from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB  # Import JSONB from postgresql dialect
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class Prediction(Base):
    """
    Database model for storing trading predictions.
    Maps directly to predictions table in the database.
    """
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime, nullable=False)
    symbol = Column(String, nullable=False)
    prediction = Column(Integer, nullable=False)  
    probability = Column(Float, nullable=False)   
    confidence = Column(String, nullable=False)
    prediction_close = Column(Float, nullable=False)
    max_hour_close = Column(Float, nullable=True)
    actual_outcome = Column(Integer, nullable=True)

class Experiment(Base):
    __tablename__ = "experiments"
    experiment_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime, nullable=False)
    symbol = Column(String, nullable=False)
    frequency = Column(String, nullable=False)
    target_variable = Column(String, nullable=False)
    hyperparameters = Column(JSONB, nullable=True)
    features = Column(JSONB, nullable=False)
    
    # Define relationship to results
    results = relationship("Results", back_populates="experiment")
    feature_importance = relationship("FeatureImportance", back_populates="experiment")

class Results(Base):
    __tablename__ = "results"
    result_id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'), nullable=False)
    best_threshold = Column(Float, nullable=True)
    walk_forward = Column(Boolean, nullable=True)
    initial_train_ratio = Column(Float, nullable=True)
    step = Column(Float, nullable=True)
    test_window = Column(Integer, nullable=True)
    precision_1 = Column(Float, nullable=True)
    recall_1 = Column(Float, nullable=True)
    f1_1 = Column(Float, nullable=True)
    precision_0 = Column(Float, nullable=True)
    recall_0 = Column(Float, nullable=True)
    f1_0 = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1 = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    
    # Define relationship to experiment
    experiment = relationship("Experiment", back_populates="results")
    feature_importance = relationship("FeatureImportance", back_populates="result")

class FeatureImportance(Base):
    __tablename__ = 'feature_importance'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'))
    result_id = Column(Integer, ForeignKey('results.result_id'))
    feature_name = Column(String, nullable=False)
    
    # SHAP metrics
    shap_importance = Column(Float)
    shap_total_importance = Column(Float)
    shap_importance_percentile = Column(Float)  # Percentile rank among SHAP values
    shap_contribution_pct = Column(Float)       # Percentage contribution to predictions
    
    # LGBM metrics
    lgbm_importance = Column(Float)
    lgbm_total_importance = Column(Float)
    lgbm_importance_percentile = Column(Float)  # Percentile rank among LGBM values
    lgbm_contribution_pct = Column(Float)       # Percentage contribution to predictions
    
    # Relationships
    experiment = relationship("Experiment", back_populates="feature_importance")
    result = relationship("Results", back_populates="feature_importance")