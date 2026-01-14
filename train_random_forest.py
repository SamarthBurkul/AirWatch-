#!/usr/bin/env python3
"""
Train RandomForestRegressor for AQI prediction and save as compressed joblib.

Usage:
    python train_random_forest.py --csv data/city_day.csv --n-estimators 80 --max-depth 12
"""

import argparse
import logging
import joblib  # More efficient than pickle for large numpy-based models
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                  'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
TARGET_COLUMN = 'AQI'
DEFAULT_N_ESTIMATORS = 80
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("train_rf")

def validate_and_prepare_df(df: pd.DataFrame):
    """Ensures feature alignment and prepares X, y."""
    for f in MODEL_FEATURES:
        if f not in df.columns:
            logger.warning("Feature '%s' not found. Filling with NaN.", f)
            df[f] = np.nan

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    X = df[MODEL_FEATURES].apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    
    mask = ~y.isna()
    return X.loc[mask, :], y.loc[mask]

def train_and_save(csv_path: str,
                   output_dir: str = 'ml_models',
                   impute_strategy: str = 'median',
                   test_size: float = 0.2,
                   n_estimators: int = DEFAULT_N_ESTIMATORS,
                   max_depth: int = None,
                   max_features: str = 'sqrt',
                   min_samples_leaf: int = 1,
                   random_state: int = RANDOM_STATE):
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    logger.info("Loading data from: %s", csv_path)
    df = pd.read_csv(csv_path)
    X, y = validate_and_prepare_df(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logger.info("Imputing with strategy: %s", impute_strategy)
    imputer = SimpleImputer(strategy=impute_strategy)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=MODEL_FEATURES, index=X_train.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=MODEL_FEATURES, index=X_test.index)

    logger.info("Training RF (estimators=%d, max_depth=%s, max_features=%s)...", 
                n_estimators, max_depth, max_features)
    
    # Using n_jobs=1 for memory stability; change to -1 for faster local training
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=1
    )
    rf.fit(X_train_imputed, y_train)

    y_pred = rf.predict(X_test_imputed)
    logger.info("Evaluation: R2=%.4f, MAE=%.4f", r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    packaged = {'model': rf, 'imputer': imputer, 'features': MODEL_FEATURES}

    out_path1 = out_dir / 'random_forest_model.pkl'
    out_path2 = Path.cwd() / 'random_forest_model.pkl'

    # joblib.dump with compression (level 3) significantly reduces file size
    logger.info("Saving compressed model to %s", out_path1)
    joblib.dump(packaged, out_path1, compress=3)
    joblib.dump(packaged, out_path2, compress=3)

    # Calculate and report file size in MB
    s1_mb = out_path1.stat().st_size / (1024 * 1024)
    logger.info("Final Model Size: %.2f MB", s1_mb)

    return {'model_paths': [str(out_path1), str(out_path2)], 'size_mb': s1_mb}

def main():
    parser = argparse.ArgumentParser(description="Train AQI Prediction Model")
    parser.add_argument('--csv', required=True, help="Path to input CSV")
    parser.add_argument('--output-dir', default='ml_models', help="Output folder")
    parser.add_argument('--impute', choices=['mean', 'median'], default='median', help="Missing value strategy")
    parser.add_argument('--test-size', type=float, default=0.2, help="Test set fraction")
    parser.add_argument('--n-estimators', type=int, default=DEFAULT_N_ESTIMATORS, help="Number of trees")
    parser.add_argument('--max-depth', type=int, default=None, help="Max depth of each tree (control size)")
    parser.add_argument('--max-features', type=str, default='sqrt', help="Features per split (sqrt is memory efficient)")
    parser.add_argument('--min-samples-leaf', type=int, default=1, help="Min samples per leaf node")
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)

    args = parser.parse_args()
    
    # Handle numeric vs string for max_features if needed (argparse passes as string)
    try:
        max_feat = int(args.max_features)
    except ValueError:
        max_feat = args.max_features

    train_and_save(
        csv_path=args.csv, 
        output_dir=args.output_dir, 
        impute_strategy=args.impute, 
        test_size=args.test_size, 
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=max_feat,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )

if __name__ == '__main__':
    main()