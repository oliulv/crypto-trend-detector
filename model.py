# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization


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

split_date = df['timestamp'].max() - pd.Timedelta(days=30)
train_idx = df['timestamp'] < split_date
test_idx = df['timestamp'] >= split_date

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

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
    'verbosity': -1  # Use 'verbosity' instead of 'verbose'
}

# 6. Train Model
print("\n🏋️ Training model...")
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

def f1_score_class1(preds, train_data):
    labels = train_data.get_label()
    preds_binary = (preds >= 0.5).astype(int)
    # Compute F1-score for class 1 using your preferred method (e.g., from sklearn.metrics)
    from sklearn.metrics import f1_score
    return 'f1_class1', f1_score(labels, preds_binary, pos_label=1), True

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    feval=f1_score_class1,
    callbacks=[
        lgb.early_stopping(300),
        lgb.log_evaluation(300)
    ]
)

# 7. Evaluate
print("\n🧪 Evaluating performance...")
y_pred = model.predict(X_test)
y_pred_class = (y_pred >= 0.5).astype(int)

# Replace the existing print statement with:
print(classification_report(y_test, y_pred_class, digits=4, target_names=['Class 0', 'Class 1']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.2f}")

# 8. Save Model
joblib.dump(model, 'data/pepe_pump_predictor.pkl')
print("\n💾 Model saved to data/pepe_pump_predictor.pkl")