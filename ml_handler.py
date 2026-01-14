"""
ml_handler.py

Lightweight ML model loader and prediction helpers for AirWatch.

Expect the model file to be a joblib dump of a dict:
{
  'model': RandomForestRegressor,
  'imputer': SimpleImputer,
  'features': ['PM2.5', 'PM10', ...]
}

This module:
- Attempts runtime download of the model file if AQI_MODEL_URL is provided.
- Loads the model with joblib and attempts memory-map mode where possible.
- Exposes:
    load_model_if_needed(now: bool=False) -> bool
    background_model_loader(delay_seconds: int = 1) -> threading.Thread
    predict_current_aqi(input_data: dict) -> Optional[float]
    get_aqi_category(aqi_value) -> dict
    calculate_all_subindices(input_data: dict) -> dict
"""

import os
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Dict

import joblib
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("ml_handler")
logger.setLevel(logging.INFO)

# Configuration from environment
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "ml_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "random_forest_model.pkl")
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

# Public URL to a model artifact (GitHub release, S3, etc.). If provided, module
# will attempt to download the model file at runtime when missing.
AQI_MODEL_URL = os.environ.get("AQI_MODEL_URL", "").strip()

# Internal state
_loaded_package: Optional[dict] = None
_load_lock = threading.Lock()
_last_load_time = 0.0


def _download_model_from_url(url: str, target_path: Path, timeout: int = 60) -> bool:
    """Download file to target_path. Returns True on success."""
    try:
        logger.info("Downloading model from %s to %s", url, target_path)
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        tmp = target_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        tmp.replace(target_path)
        logger.info("Model downloaded successfully to %s", target_path)
        return True
    except Exception as exc:
        logger.exception("Failed to download model from %s: %s", url, exc)
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        return False


def _load_from_disk(path: Path):
    """Load the packaged model dict from disk using joblib (try mmap first)."""
    try:
        logger.info("Attempting joblib.load(..., mmap_mode='r') for %s", path)
        try:
            pkg = joblib.load(path, mmap_mode='r')
        except TypeError:
            # Older joblib versions may not accept mmap_mode on certain objects
            pkg = joblib.load(path)
        except Exception as e_mmap:
            # If mmap fails (sometimes for certain pickles), try a normal load
            logger.warning("mmap_mode load failed: %s — will retry normal joblib.load", e_mmap)
            pkg = joblib.load(path)
        # Basic validation
        if not isinstance(pkg, dict) or 'model' not in pkg:
            logger.error("Loaded package does not contain 'model' key; invalid package.")
            return None
        return pkg
    except Exception as exc:
        logger.exception("Error loading model from disk: %s", exc)
        return None


def load_model_if_needed(now: bool = False) -> bool:
    """
    Ensure model is loaded.
    - If already loaded, returns True.
    - If not, attempts to load. If now=True, blocks until loaded (useful for synchronous predict).
    - If AQI_MODEL_URL provided and local file missing or obviously corrupted, attempt download.
    """
    global _loaded_package, _last_load_time

    if _loaded_package is not None:
        return True

    # Acquire lock; if caller doesn't want to wait and another thread is loading, return False
    acquired = _load_lock.acquire(blocking=now)
    if not acquired:
        logger.debug("Model load already in progress by another thread; returning False (non-blocking).")
        return False

    try:
        # double-check inside lock
        if _loaded_package is not None:
            return True

        # If local file missing or zero-sized, try download (if URL present)
        if (not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 10) and AQI_MODEL_URL:
            ok = _download_model_from_url(AQI_MODEL_URL, MODEL_PATH)
            if not ok:
                logger.warning("Could not download model from AQI_MODEL_URL (%s). Will try to load local file if present.", AQI_MODEL_URL)

        # Try to load package from disk
        if MODEL_PATH.exists():
            pkg = _load_from_disk(MODEL_PATH)
            if pkg is None:
                logger.warning("Model package invalid or could not be loaded from disk (%s).", MODEL_PATH)
                return False

            # Basic validation
            missing_keys = [k for k in ('model', 'imputer', 'features') if k not in pkg]
            if missing_keys:
                logger.error("Model package missing required keys: %s", missing_keys)
                return False

            _loaded_package = pkg
            _last_load_time = time.time()
            logger.info("Model loaded successfully and ready (path=%s).", MODEL_PATH)
            return True
        else:
            logger.warning("Model file not present at %s and AQI_MODEL_URL not provided or download failed.", MODEL_PATH)
            return False
    finally:
        _load_lock.release()


def background_model_loader(delay_seconds: int = 1):
    """
    Start a background thread to load the model after delay_seconds. Non-blocking.
    Useful during app startup.
    """
    def _job():
        time.sleep(delay_seconds)
        try:
            load_model_if_needed(now=True)
        except Exception:
            logger.exception("Background model loader failed.")

    t = threading.Thread(target=_job, daemon=True)
    t.start()
    logger.info("Background model loader scheduled to start after %s seconds.", delay_seconds)
    return t


def predict_current_aqi(input_data: dict) -> Optional[float]:
    """
    Predict AQI from an input mapping pollutant -> numeric value (strings convertible to float).
    Returns predicted AQI (float) or None on failure.
    This will attempt to load the model synchronously if it is not loaded.
    """
    if not load_model_if_needed(now=True):
        logger.error("Model is not loaded; cannot predict.")
        return None

    pkg = _loaded_package
    if not pkg:
        logger.error("Internal: _loaded_package is empty after load.")
        return None

    model = pkg.get('model')
    imputer = pkg.get('imputer')
    features = pkg.get('features')

    if model is None or imputer is None or features is None:
        logger.error("Model package missing keys (model/imputer/features).")
        return None

    try:
        # Create a single-row DataFrame in the required order
        row = {f: input_data.get(f, None) for f in features}
        df = pd.DataFrame([row], columns=features)
        df = df.apply(pd.to_numeric, errors='coerce')
        X = pd.DataFrame(imputer.transform(df), columns=features)
        pred = model.predict(X)[0]
        return float(np.round(float(pred), 4))
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return None


def get_aqi_category(aqi_value) -> Dict[str, str]:
    """Return frontend-friendly category mapping (text color, chart color, etc.)."""
    try:
        aqi_val = float(aqi_value)
    except Exception:
        aqi_val = None

    if aqi_val is None:
        return {"category": "N/A", "description": "AQI data invalid.", "textColor": "text-slate-400", "borderColor": "border-slate-500", "bgColor": "bg-slate-500/10", "chartColor": "#64748b"}
    if aqi_val <= 50:
        return {"category": "Good", "description": "Minimal impact.", "textColor": "text-green-400", "borderColor": "border-green-500", "bgColor": "bg-green-500/20", "chartColor": "#34d399"}
    if aqi_val <= 100:
        return {"category": "Satisfactory", "description": "Minor breathing discomfort.", "textColor": "text-yellow-400", "borderColor": "border-yellow-500", "bgColor": "bg-yellow-500/20", "chartColor": "#f59e0b"}
    if aqi_val <= 200:
        return {"category": "Moderate", "description": "Breathing discomfort to sensitive groups.", "textColor": "text-orange-400", "borderColor": "border-orange-500", "bgColor": "bg-orange-500/20", "chartColor": "#f97316"}
    if aqi_val <= 300:
        return {"category": "Poor", "description": "Breathing discomfort to most people.", "textColor": "text-red-400", "borderColor": "border-red-500", "bgColor": "bg-red-500/20", "chartColor": "#ef4444"}
    if aqi_val <= 400:
        return {"category": "Very Poor", "description": "Respiratory illness on prolonged exposure.", "textColor": "text-purple-400", "borderColor": "border-purple-500", "bgColor": "bg-purple-500/20", "chartColor": "#a855f7"}
    return {"category": "Severe", "description": "Serious health effects.", "textColor": "text-rose-700", "borderColor": "border-rose-700", "bgColor": "bg-rose-800/20", "chartColor": "#be123c"}


def calculate_all_subindices(input_data: dict) -> Dict[str, float]:
    """
    Create a pollutant contribution breakdown.
    If model.feature_importances_ present, allocate predicted AQI by importance.
    Otherwise fallback to proportional split of positive input values.
    """
    # If model is not loaded, try quickly to load; if failed return empty
    if not load_model_if_needed(now=False):
        return {}

    try:
        pkg = _loaded_package
        model = pkg.get('model')
        features = pkg.get('features')
        predicted = predict_current_aqi(input_data)
        if predicted is None:
            return {}

        fi = getattr(model, "feature_importances_", None)
        if fi is None or len(fi) != len(features):
            # fallback proportional to positive inputs
            values = []
            total = 0.0
            for f in features:
                try:
                    v = float(input_data.get(f) or 0.0)
                except Exception:
                    v = 0.0
                v = max(0.0, v)
                values.append(v)
                total += v
            contributions = {}
            if total <= 0:
                per = round(predicted / len(features), 2)
                for f in features:
                    contributions[f] = per
            else:
                for f, v in zip(features, values):
                    contributions[f] = round(predicted * (v / total), 2)
            return contributions

        fi = np.array(fi, dtype=float)
        s = fi.sum()
        if s <= 0:
            fi_norm = np.ones_like(fi) / len(fi)
        else:
            fi_norm = fi / s
        contributions = {features[i]: round(float(predicted * fi_norm[i]), 2) for i in range(len(features))}
        return contributions
    except Exception as exc:
        logger.exception("Failed to calculate subindices: %s", exc)
        return {}
