import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

# Parameters:
symbol = "PEPEUSDT"
start_date = "2023-05-20"
end_date = "2025-02-04"


def analyze_model(symbol, start_date, end_date):
    np.seterr(invalid='ignore')  # Just to avoid that error message for now.

    # Load model and data
    loaded = joblib.load(f'model/{symbol}_pump_predictor.pkl')
    # Fix: if loaded is a tuple, extract the model
    if isinstance(loaded, tuple):
        model = loaded[0]
    else:
        model = loaded

    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_filename = f'{symbol}_1hr_window_labels_{start_date}_to_{end_date}.csv'
    data_path = os.path.join(project_root, 'data', data_filename)
    
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    
    df = df.sort_values(by="timestamp", ascending=True)  # Ensures chronological order
    
    # Prepare features
    X = df.drop(columns=['label']).ffill().fillna(0)
    y = df['label']
    
    # 1. Check for Data Leakage, see code below
    check_data_leakage(df)

    # 1.1 Check for future data leakage, see code below
    check_future_data_leakage(df)
    
    # 2. Walk-forward Validation
    print("\n🧪 Performing walk-forward validation...")
    X_sorted = X.sort_values(by='timestamp', ascending=True).drop(columns='timestamp')
    y_sorted = y[X_sorted.index]
    
    # 2. Store walk forward return values
    # Set initial_train_size to 6 months (43200 minutes per month * 6)
    predictions, true_labels, iteration_reports = time_series_walk_forward_validation(
        model, X_sorted, y_sorted, threshold=0.613, initial_train_size=(43200*6), step=43200
    )
    
    # Print and save validation results
    print_validation_results(predictions, true_labels, iteration_reports, threshold=0.613)
    
   # 3. Feature Importance with SHAP
    print("\n📊 Calculating feature importance...")

    # Ignore expected warning
    warnings.filterwarnings("ignore", category=UserWarning, module="shap")

    # Use default model_output="raw" and check shap_values structure
    explainer = shap.TreeExplainer(model, approximate=True)
    shap_values = explainer.shap_values(X_sorted.iloc[:500])

    # For binary classification, shap_values will be a list of two arrays:
    # [negative_class_shap, positive_class_shap]
    if isinstance(shap_values, list):
        # Use index 1 for positive class SHAP values
        plot_values = shap_values[1]
    else:
        # If single array returned (some SHAP versions), use directly
        plot_values = shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(plot_values, X_sorted.iloc[:500], plot_type="bar")
    plt.savefig('reports/feature_importance.png')
    plt.close()


def time_series_walk_forward_validation(model, X, y, threshold=0.613, initial_train_size=(43200*6), step=43200):
    """Walk-forward validation for time series.
       Also returns per-iteration classification metrics (as dictionaries).
    """
    aggregated_predictions = []
    aggregated_true_labels = []
    iteration_reports = []  # To store per-iteration metrics
        
    for i in tqdm(range(initial_train_size, len(X), step), desc="Walk-forward validation"):
        # Split data
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[i:i+step]
        y_test = y.iloc[i:i+step]
        
        # Retrain model on expanding window
        model.fit(X_train, y_train)
        
        # Get predictions
        preds = model.predict_proba(X_test)[:, 1]
        
        # Convert to binary predictions using the given threshold
        binary_preds = (preds >= threshold).astype(int)
        
        # Append predictions and true labels for aggregated metrics
        aggregated_predictions.extend(preds)
        aggregated_true_labels.extend(y_test)
        
        # Compute iteration classification report using the imported classification_report
        report = classification_report(y_test, binary_preds, output_dict=True, target_names=['Class 0', 'Class 1'])
        iteration_reports.append(report)
        
        if i % 10000 == 0:
            print(f"Completed {i/1440:.1f} days of backtesting")
    
    return (np.array(aggregated_predictions), np.array(aggregated_true_labels), iteration_reports)


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


def print_validation_results(aggregated_predictions, aggregated_true_labels, iteration_reports, threshold=0.613):
    # Compute overall aggregated binary predictions using the threshold
    binary_predictions = (aggregated_predictions >= threshold).astype(int)
    
    # Compute overall metrics using the imported functions
    overall_accuracy = accuracy_score(aggregated_true_labels, binary_predictions)
    overall_precision = precision_score(aggregated_true_labels, binary_predictions, zero_division=0)
    overall_recall = recall_score(aggregated_true_labels, binary_predictions, zero_division=0)
    overall_f1 = f1_score(aggregated_true_labels, binary_predictions, zero_division=0)
    overall_cm = confusion_matrix(aggregated_true_labels, binary_predictions)
    overall_auc = roc_auc_score(aggregated_true_labels, aggregated_predictions)
    
    print("\n📊 Overall Aggregated Validation Results:")
    print(f"Accuracy:  {overall_accuracy:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall:    {overall_recall:.4f}")
    print(f"F1-Score:  {overall_f1:.4f}")
    print("Confusion Matrix:")
    print(overall_cm)
    print(f"AUC-ROC:   {overall_auc:.4f}\n")
    
    # Also, display the classification report (which uses the imported classification_report)
    print("Classification Report (Aggregated):")
    print(classification_report(aggregated_true_labels, binary_predictions, digits=4, target_names=['Class 0', 'Class 1']))
    
    # Average the per-iteration reports (these are produced by classification_report output_dict)
    avg_metrics = {}
    keys = iteration_reports[0].keys()  # e.g. 'Class 0', 'Class 1', 'accuracy', 'macro avg', 'weighted avg'
    for key in keys:
        if isinstance(iteration_reports[0][key], dict):
            avg_metrics[key] = {}
            for subkey in iteration_reports[0][key]:
                avg_metrics[key][subkey] = np.mean([report[key][subkey] for report in iteration_reports if key in report])
        else:
            avg_metrics[key] = np.mean([report[key] for report in iteration_reports if key in report])
    
    print("📊 Average Per-Iteration Metrics (based on predictions):")
    print("              precision    recall  f1-score   support")
    for class_name in ['Class 0', 'Class 1']:
        if class_name in overall_cm:  # Use overall report keys to check
            precision = np.mean([report[class_name]['precision'] for report in iteration_reports if class_name in report])
            recall = np.mean([report[class_name]['recall'] for report in iteration_reports if class_name in report])
            f1 = np.mean([report[class_name]['f1-score'] for report in iteration_reports if class_name in report])
            support = np.sum([report[class_name]['support'] for report in iteration_reports if class_name in report])
            print(f"{class_name:>10} {precision:10.4f} {recall:10.4f} {f1:10.4f} {int(support):10d}")
    
    overall_iter_accuracy = np.mean([report['accuracy'] for report in iteration_reports if 'accuracy' in report])
    macro_precision = np.mean([report['macro avg']['precision'] for report in iteration_reports if 'macro avg' in report])
    macro_recall = np.mean([report['macro avg']['recall'] for report in iteration_reports if 'macro avg' in report])
    macro_f1 = np.mean([report['macro avg']['f1-score'] for report in iteration_reports if 'macro avg' in report])
    
    weighted_precision = np.mean([report['weighted avg']['precision'] for report in iteration_reports if 'weighted avg' in report])
    weighted_recall = np.mean([report['weighted avg']['recall'] for report in iteration_reports if 'weighted avg' in report])
    weighted_f1 = np.mean([report['weighted avg']['f1-score'] for report in iteration_reports if 'weighted avg' in report])
    
    print(f"\nOverall Accuracy (averaged over iterations): {overall_iter_accuracy:.4f}")
    print("\nMacro Average (averaged over iterations):")
    print(f"Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1-Score: {macro_f1:.4f}")
    print("\nWeighted Average (averaged over iterations):")
    print(f"Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1-Score: {weighted_f1:.4f}")


if __name__ == "__main__":
    analyze_model(symbol, start_date, end_date)
