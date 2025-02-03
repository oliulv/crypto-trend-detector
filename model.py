import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.isotonic import IsotonicRegression


# 1. Load Data
print("🕵️♂️ Loading dataset...")
df = pd.read_csv('data/PEPE_1hr_window_labels_2023-05-20_to_2025-02-02.csv', 
                 parse_dates=['timestamp'])

# 2. Prepare Features
X = df.drop(columns=['label', 'timestamp'])
y = df['label']

# 3. Handle Missing Data
X = X.ffill().fillna(0)

# 3.1 Tuning Dashboard:
weight_multiplier = 2
class_ratio_multiplier = 2

# 4. Time Series Split
split_date = df['timestamp'].max() - pd.Timedelta(days=62)
train_idx = df['timestamp'] < split_date
test_idx = df['timestamp'] >= split_date

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Class Weight Calculation
class_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# 5. Configure Model
model = LGBMClassifier(
    objective='binary',
    metric='auc',  # Changed from eval_metric
    num_leaves=31,
    learning_rate=0.05,
    colsample_bytree=0.9,  # Changed from feature_fraction
    subsample=0.8,  # Changed from bagging_fraction
    subsample_freq=5,  # Changed from bagging_freq
    verbosity=-1,
    scale_pos_weight=class_ratio * class_ratio_multiplier,
    n_estimators=1000,
    early_stopping_round=300,  # Singular form (FIXED)
    eval_metric='auc'  # Keep this for evaluation output
)

# 6. Train Model with Sample Weighting
print("\n🏋️ Training model...")

sample_weights = np.where(y_train == 1, weight_multiplier, 1)

# Train with early stopping
model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],  # Validation set for early stopping
)

# 7. Calibrate Probabilities (Updated)
print("\n🔧 Calibrating probabilities...")

# Split training data into main/validation sets (time-series safe)
X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=False
)

# --- NEW: Split sample weights to match the data splits ---
n_main = len(X_train_main)
sample_weights_main = sample_weights[:n_main]  # Defined in original training
sample_weights_val = sample_weights[n_main:]

# --- Get cleaned model parameters ---
model_params = model.get_params().copy()
# Remove parameters requiring validation data
for param in ['early_stopping_round', 'eval_metric', 'eval_set']:
    model_params.pop(param, None)

# Retrain on main training set with weights
model_main = LGBMClassifier(**model_params).fit(  # Now properly defined
    X_train_main, y_train_main,
    sample_weight=sample_weights_main
)

# Get validation set probabilities for calibration
val_probs = model_main.predict_proba(X_val)[:, 1]

# Train isotonic calibrator
calibrator = IsotonicRegression(out_of_bounds='clip').fit(val_probs, y_val)

# Apply calibration to test set probabilities
y_pred_calibrated = calibrator.transform(
    model_main.predict_proba(X_test)[:, 1]
)

# 8. Precision-Focused Threshold Tuning
def find_balanced_threshold(y_true, y_pred_proba, min_recall=0.5, min_precision=0.65):
    """Find threshold meeting minimum requirements while maximizing precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    viable = np.where((recalls[:-1] >= min_recall) & (precisions[:-1] >= min_precision))[0]
    if len(viable) > 0:
        best_idx = viable[np.argmax(precisions[viable])]
        return thresholds[best_idx], precisions[best_idx], recalls[best_idx]
    return 0.5, precisions[-1], recalls[-1]

print("\n🎯 Tuning threshold with precision focus...")
optimal_threshold, prec, rec = find_balanced_threshold(
    y_test, y_pred_calibrated, min_recall=0.5, min_precision=0.65
)
print(f"✅ Optimal threshold: {optimal_threshold:.3f} (Precision={prec:.1%}, Recall={rec:.1%})")

# 9. Diagnostic Visualization
def plot_threshold_tradeoff(y_true, y_pred_proba, threshold):
    """Visualize precision/recall vs threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Chosen Threshold ({threshold:.3f})')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall Tradeoff")
    plt.legend()
    plt.show()

print("\n📈 Generating diagnostic plot...")
plot_threshold_tradeoff(y_test, y_pred_calibrated, optimal_threshold)

# 10. Final Evaluation
print("\n🧪 Final evaluation with calibrated predictions:")
y_pred_class = (y_pred_calibrated >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_class, digits=4, target_names=['Class 0', 'Class 1']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_calibrated):.2f}")

# 11. Save Model with Calibrator
joblib.dump((model_main, calibrator), 'data/pepe_pump_predictor.pkl')
print("\n💾 Calibrated model saved to data/pepe_pump_predictor.pkl")