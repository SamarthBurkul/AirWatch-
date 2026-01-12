#!/usr/bin/env python3
"""
Train RandomForestRegressor for AQI prediction and save as pickle.

Usage examples:
    python train_random_forest.py --csv data/city_day.csv
    python train_random_forest.py --csv data/city_day.csv --impute mean --test-size 0.2
"""

import os
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------------------------------------------------------
# Configuration: required feature list and default parameters
# -----------------------------------------------------------------------------
MODEL_FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                  'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
TARGET_COLUMN = 'AQI'  # target column name in CSV

DEFAULT_N_ESTIMATORS = 200
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger("train_rf")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def validate_and_prepare_df(df: pd.DataFrame):
    """
    Ensure required columns exist, reorder features, and return X (DataFrame) and y (Series).
    Missing feature columns that do not exist will be created filled with NaN so imputer can handle them.
    """
    # Ensure all feature columns exist in DataFrame; if not, create them with NaN
    for f in MODEL_FEATURES:
        if f not in df.columns:
            logger.warning("Feature '%s' not found in CSV. Creating column filled with NaN.", f)
            df[f] = np.nan

    # Ensure target exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV.")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    if df.shape[0] == 0:
        raise ValueError("No rows remain after dropping rows without the target column.")

    # Reorder X to required feature order and ensure dtype numeric
    X = df[MODEL_FEATURES].copy()
    # Try converting all feature columns to numeric (coerce errors to NaN)
    X = X.apply(pd.to_numeric, errors='coerce')

    # Target numeric
    y = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    # Drop rows where target could not be parsed
    mask = ~y.isna()
    X = X.loc[mask, :]
    y = y.loc[mask]

    return X, y

# -----------------------------------------------------------------------------
# Training pipeline
# -----------------------------------------------------------------------------
def train_and_save(csv_path: str,
                   output_dir: str = 'ml_models',
                   impute_strategy: str = 'median',
                   test_size: float = 0.2,
                   n_estimators: int = DEFAULT_N_ESTIMATORS,
                   random_state: int = RANDOM_STATE):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV dataset not found at: {csv_path}")

    logger.info("Loading data from: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("CSV loaded. Rows: %d, Columns: %d", df.shape[0], df.shape[1])

    # Prepare X, y with required features and target
    X, y = validate_and_prepare_df(df)
    logger.info("Prepared features X shape: %s and target y shape: %s", X.shape, y.shape)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Train split: %d rows, Test split: %d rows", X_train.shape[0], X_test.shape[0])

    # Impute missing values (on training set then apply to test)
    logger.info("Imputing missing values using strategy: %s", impute_strategy)
    imputer = SimpleImputer(strategy=impute_strategy)
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=MODEL_FEATURES,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=MODEL_FEATURES,
        index=X_test.index
    )

    # Instantiate and train RandomForestRegressor
    logger.info("Training RandomForestRegressor (n_estimators=%d, random_state=%d)...",
                n_estimators, random_state)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train_imputed, y_train)

    # Compute performance metrics on test set
    y_pred = rf.predict(X_test_imputed)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    logger.info("Evaluation on test set: R2=%.4f, MAE=%.4f, MSE=%.4f", r2, mae, mse)

    # Ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save both the model and the imputer together so downstream code can apply same imputation if necessary.
    # We'll save a simple dictionary with keys: 'model', 'imputer', 'features' for convenience.
    packaged = {
        'model': rf,
        'imputer': imputer,
        'features': MODEL_FEATURES
    }

    out_path1 = out_dir / 'random_forest_model.pkl'        # ml_models/random_forest_model.pkl
    out_path2 = Path.cwd() / 'random_forest_model.pkl'     # random_forest_model.pkl at repo root

    logger.info("Saving trained model to: %s and %s", out_path1, out_path2)
    with open(out_path1, 'wb') as f:
        pickle.dump(packaged, f)
    with open(out_path2, 'wb') as f:
        pickle.dump(packaged, f)

    logger.info("Model saved successfully.")

    # Quick check: load back and test a sample prediction (zeros)
    with open(out_path1, 'rb') as f:
        loaded = pickle.load(f)
    model_loaded = loaded['model']
    imputer_loaded = loaded['imputer']
    # Example sample: zeros or small values; must be same columns order
    sample = pd.DataFrame([np.zeros(len(MODEL_FEATURES))], columns=MODEL_FEATURES)
    sample_imputed = pd.DataFrame(imputer_loaded.transform(sample), columns=MODEL_FEATURES)
    pred_sample = model_loaded.predict(sample_imputed)[0]
    logger.info("Sanity check prediction for zero-sample: %.4f", float(pred_sample))

    # Return paths and metrics
    return {
        'model_paths': [str(out_path1), str(out_path2)],
        'metrics': {'r2': float(r2), 'mae': float(mae), 'mse': float(mse)},
        'n_train': int(X_train.shape[0]),
        'n_test': int(X_test.shape[0])
    }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train RandomForest for AQI prediction.")
    parser.add_argument('--csv', required=True, help='Path to dataset CSV (e.g., data/city_day.csv)')
    parser.add_argument('--output-dir', default='ml_models', help='Directory to save trained model')
    parser.add_argument('--impute', choices=['mean', 'median'], default='median', help='Imputation strategy')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split proportion')
    parser.add_argument('--n-estimators', type=int, default=DEFAULT_N_ESTIMATORS, help='RandomForest n_estimators')
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE, help='Random state for reproducibility')

    args = parser.parse_args()
    try:
        result = train_and_save(
            csv_path=args.csv,
            output_dir=args.output_dir,
            impute_strategy=args.impute,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            random_state=args.random_state
        )
        logger.info("Training complete. Results: %s", result)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise

if __name__ == '__main__':
    main()
