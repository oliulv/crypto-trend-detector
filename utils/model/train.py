import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import joblib
import os


# Parameters:
symbol = "PEPEUSDT"
start_date = "2023-05-20"
end_date = "2025-02-15"


def train_model(symbol, start_date, end_date):
    # 1. Load Data
    print("🕵️♂️ Loading dataset...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_filename = f'{symbol}_1hr_window_labels_{start_date}_to_{end_date}.csv'
    data_path = os.path.join(project_root, 'data', data_filename)
    df = pd.read_csv(data_path, parse_dates=['timestamp'])

    # 2. Prepare Features
    X = df.drop(columns=['label', 'timestamp'])
    y = df['label']

    # 3. Handle Missing Data
    X = X.ffill().fillna(0)

    # 3.1 Tuning Dashboard:
    weight_multiplier = 2
    class_ratio_multiplier = 2

    # 4. Use All Data for Training
    X_train = X
    y_train = y

    # Class Weight Calculation
    class_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # 5. Configure Model (removed early stopping and validation-related params)
    model = LGBMClassifier(
        objective='binary',
        num_leaves=31,
        learning_rate=0.05,
        colsample_bytree=0.9,
        subsample=0.8,
        subsample_freq=5,
        verbosity=-1,
        scale_pos_weight=class_ratio * class_ratio_multiplier,
        n_estimators=1000
    )

    # 6. Train Model with Sample Weighting
    print("\n🏋️ Training model on full dataset...")
    sample_weights = np.where(y_train == 1, weight_multiplier, 1)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # 7. Save Model
    joblib.dump(model, f'models/{symbol}_pump_predictor_full.pkl')
    print(f"\n💾 Full dataset model saved to models/{symbol}_pump_predictor_full.pkl")


if __name__ == "__main__":
    train_model(symbol, start_date, end_date)