class ProductionModel:
    """Production-ready model with built-in threshold and optional calibration."""
    
    def __init__(self, base_model, threshold, calibrator=None):
        self.base_model = base_model    # Define our base model
        self.threshold = threshold      # Define our threshold
        self.calibrator = calibrator    # Define our calibrator (optional)
    
    def predict_proba(self, X):
        """Get probabilities with optional calibration."""
        probs = self.base_model.predict_proba(X)[:, 1]  # Get positive class probabilities
        if self.calibrator is not None:
            probs = self.calibrator.transform(probs)    # Calibrate probabilities
        return probs
    
    def predict(self, X):
        """Get binary predictions using tuned threshold."""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)