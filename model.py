# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score


# 1. Load Data
print("🕵️♂️ Loading dataset...")
df = pd.read_csv('data/PEPE_1hr_window_labels_2023-05-20_to_2025-02-02.csv', 
                 parse_dates=['timestamp'])

# 2. Prepare Features
X = df.drop(columns=['label', 'timestamp'])
y = df['label']

# 3. Handle Missing Data
X = X.ffill().fillna(0)

# 4. Time Series Split
split_date = df['timestamp'].max() - pd.Timedelta(days=62)
train_idx = df['timestamp'] < split_date
test_idx = df['timestamp'] >= split_date

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Class Weight Calculation
class_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# 5. Configure Model
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'scale_pos_weight': class_ratio * 2  # Prioritize Class 1
}

# 6. Train Model with Sample Weighting
print("\n🏋️ Training model...")

sample_weights = np.where(y_train == 1, 2, 1)
train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

def f1_score_class1(preds, train_data):
    labels = train_data.get_label()
    preds_binary = (preds >= 0.5).astype(int)
    return 'f1_class1', f1_score(labels, preds_binary, pos_label=1), True

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    feval=f1_score_class1,
    callbacks=[
        lgb.early_stopping(300),
        lgb.log_evaluation(50)
    ]
)

# 7. Evaluate with Optimal Threshold
print("\n🧪 Evaluating performance...")
y_pred = model.predict(X_test)

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred, pos_label=1)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

y_pred_class = (y_pred >= optimal_threshold).astype(int)

print(classification_report(y_test, y_pred_class, digits=4, target_names=['Class 0', 'Class 1']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.2f}")
print(f"Optimal Threshold: {optimal_threshold:.3f}")

# 8. Save Model
joblib.dump(model, 'data/pepe_pump_predictor.pkl')
print("\n💾 Model saved to data/pepe_pump_predictor.pkl")