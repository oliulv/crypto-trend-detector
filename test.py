# test.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

def time_series_walk_forward_validation(model, X, y, initial_train_size=4320, step=1440):
    """Walk-forward validation for time series"""
    predictions = []
    true_labels = []
    
    for i in tqdm(range(initial_train_size, len(X), step), desc="Walk-forward validation"):
        # Split data
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[i:i+step]
        y_test = y.iloc[i:i+step]
        
        # Retrain model on expanding window
        model.fit(X_train, y_train)
        
        # Make predictions
        preds = model.predict_proba(X_test)[:, 1]
        predictions.extend(preds)
        true_labels.extend(y_test)

        # Print progress:
        if i % 10000 == 0:  # Every ~7 days
            print(f"Completed {i/1440:.1f} days of backtesting")
    
    return np.array(predictions), np.array(true_labels)

def analyze_model():
    np.seterr(invalid='ignore')  # Just to avoid that error message for now.

    # Load model and data
    model, calibrator = joblib.load('model/pepe_pump_predictor.pkl')
    df = pd.read_csv('data/PEPE_1hr_window_labels_2023-05-20_to_2025-02-02.csv', 
                    parse_dates=['timestamp'])
    
    df = df.sort_values(by="timestamp", ascending=True)  # Ensures chronological order
    
    # Prepare features
    X = df.drop(columns=['label']).ffill().fillna(0)
    y = df['label']
    
    # 1. Check for Data Leakage
    def check_data_leakage(df):
        print("\n🔍 Checking for data leakage...")

        # Check for duplicate timestamps
        if df['timestamp'].duplicated().any():
            print("⚠️ Warning: Duplicate timestamps detected! Possible data leakage.")

        # Check for correct time ordering
        if not df['timestamp'].is_monotonic_increasing:
            print("⚠️ Warning: Timestamps are not strictly increasing!")

        # Compute time differences between rows
        time_diff = df['timestamp'].diff().dt.total_seconds()
        if (time_diff < 60).any():
            print("⚠️ Warning: Irregular time intervals detected!")

        # Correlation check between features and label
        feature_corrs = df.drop(columns=['timestamp', 'label']).corrwith(df['label']).abs()
        high_corr_features = feature_corrs[feature_corrs > 0.9].index.tolist()

        if high_corr_features:
            print(f"⚠️ Warning: Potential label leakage! Features highly correlated with the target: {high_corr_features}")

        print("✅ No obvious data leakage found (but manual review is always recommended).")

    check_data_leakage(df)

    # 1.1 Check for future data leakage
    def check_future_data_leakage(df, timestamp_col="timestamp", label_col="label"):
        print("\n🔍 Checking for future data leakage (excluding known safe fields)...")
        ignore_cols = {'hour', 'minute', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend',
                    'asian_session', 'us_session'}
        feature_cols = [col for col in df.columns if col not in [timestamp_col, label_col] and col not in ignore_cols]
        
        leakage_found = False
        for feature in feature_cols:
            # Here, instead of checking for a repeated value,
            # we can check for any suspicious pattern.
            # One simple heuristic: check if the feature is exactly constant
            # or if its difference with a lag is zero almost everywhere.
            diff = df[feature].diff().dropna()
            if np.all(diff == 0):
                print(f"❌ Warning: Feature '{feature}' does not change over time. Check its computation.")
                leakage_found = True
        if not leakage_found:
            print("✅ No future data leakage detected (after ignoring known cyclic fields).")

    
    check_future_data_leakage(df)

    
    # 2. Walk-forward Validation
    print("\n🧪 Performing walk-forward validation...")
    X_sorted = X.sort_values(by='timestamp', ascending=True).drop(columns='timestamp')
    y_sorted = y[X_sorted.index]

    
    predictions, true_labels = time_series_walk_forward_validation(
        model, X_sorted, y_sorted
    )
    
    # 3. Feature Importance with SHAP
    print("\n📊 Calculating feature importance...")
    explainer = shap.TreeExplainer(model, approximate=True)
    # sample_size = min(1000, len(X_sorted))
    shap_values = explainer.shap_values(X_sorted.iloc[:500])
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[1], X_sorted.iloc[:500], plot_type="bar")
    plt.savefig('reports/feature_importance.png')
    plt.close()
    
    # 4. Final Evaluation
    print("\n📈 Final Performance Metrics:")
    print(classification_report(true_labels, (predictions >= 0.5).astype(int)))
    print(f"AUC-ROC: {roc_auc_score(true_labels, predictions):.2f}")

if __name__ == "__main__":
    analyze_model()