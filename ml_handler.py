# ml_handler.py
import os
import logging
import pickle
import gc
from pathlib import Path
from typing import Any, Dict, Optional

# third-party
import pandas as pd

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("ml_handler")

# Default feature order (fallback)
MODEL_FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Model location configuration
MODEL_DIR = Path(__file__).resolve().parent / "ml_models"
MODEL_FILENAME = "random_forest_model.pkl"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

# Environment override: Render UI should set AQI_MODEL_URL (or MODEL_URL)
DEFAULT_MODEL_URL = os.environ.get("AQI_MODEL_URL") or os.environ.get("MODEL_URL") or \
    "https://github.com/SamarthBurkul/AirWatch-/releases/download/v1.1.0/random_forest_model.pkl"

# Globals (populated lazily)
_AQI_MODEL_OBJ: Optional[Any] = None
_AQI_IMPUTER: Optional[Any] = None
_AQI_FEATURES: Optional[list] = None
_USING_JOBLIB = False

# -----------------------
# Helper: download functions
# -----------------------
def _download_file_requests(url: str, dest: Path, timeout: int = 120) -> bool:
    try:
        import requests
    except Exception:
        return False
    try:
        logger.info("Downloading model via requests: %s -> %s", url, dest)
        tmp = dest.with_suffix(".tmp")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
        tmp.replace(dest)
        logger.info("Downloaded model to %s", dest)
        return True
    except Exception as e:
        logger.exception("requests download failed: %s", e)
        return False

def _download_file_urllib(url: str, dest: Path) -> bool:
    try:
        import urllib.request
    except Exception:
        return False
    try:
        logger.info("Downloading model via urllib: %s -> %s", url, dest)
        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        logger.info("Downloaded model to %s (urllib)", dest)
        return True
    except Exception as e:
        logger.exception("urllib download failed: %s", e)
        return False

def ensure_model_file(path: Path = MODEL_PATH, url: str = DEFAULT_MODEL_URL) -> bool:
    """
    Ensure model file exists locally; if not, try to download from `url`.
    Returns True if file exists after the call.
    """
    try:
        if path.exists() and path.is_file():
            logger.debug("Model found at %s", path)
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        # try requests then urllib
        if _download_file_requests(url, path):
            return True
        if _download_file_urllib(url, path):
            return True
        logger.error("Could not download model from %s", url)
        return False
    except Exception as e:
        logger.exception("ensure_model_file error: %s", e)
        return False

# -----------------------
# Loading logic (lazy)
# -----------------------
def _try_joblib_load(path: Path, mmap_mode: Optional[str] = "r"):
    """Attempt joblib.load with mmap_mode if available."""
    try:
        import joblib
    except Exception as e:
        logger.debug("joblib not available: %s", e)
        raise

    # Try memory-mapped load first (if requested)
    if mmap_mode:
        try:
            logger.info("Attempting joblib.load(..., mmap_mode=%s) for %s", mmap_mode, path)
            obj = joblib.load(str(path), mmap_mode=mmap_mode)
            logger.info("joblib.load with mmap_mode succeeded")
            return obj, True
        except Exception as e:
            logger.warning("joblib.load with mmap_mode failed: %s; trying without mmap", e)

    # Fallback to plain joblib.load
    logger.info("Attempting joblib.load without mmap for %s", path)
    obj = joblib.load(str(path))
    return obj, True

def _try_pickle_load(path: Path):
    """Fallback pickled-object load using pickle."""
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("Loaded model using pickle from %s", path)
        return obj
    except Exception as e:
        logger.exception("pickle.load failed: %s", e)
        raise

def load_model(path: Optional[Path] = None, mmap_mode: Optional[str] = "r") -> bool:
    """
    Load the model into module-global variables.
    - mmap_mode='r' will attempt joblib memory-mapping to reduce peak RAM.
    - Returns True on success.
    """
    global _AQI_MODEL_OBJ, _AQI_IMPUTER, _AQI_FEATURES, _USING_JOBLIB

    target = Path(path) if path else MODEL_PATH

    # ensure file present
    if not target.exists():
        logger.warning("Model file missing at %s; attempting download from %s", target, DEFAULT_MODEL_URL)
        if not ensure_model_file(target, DEFAULT_MODEL_URL):
            logger.error("Model file not available and download failed.")
            return False

    # Try joblib (with mmap), then joblib, then pickle
    try:
        try:
            obj, used_joblib = _try_joblib_load(target, mmap_mode=mmap_mode)
            _USING_JOBLIB = used_joblib
        except Exception:
            # try pickle fallback
            obj = _try_pickle_load(target)
            _USING_JOBLIB = False

        # Interpret package structure
        if isinstance(obj, dict):
            _AQI_MODEL_OBJ = obj.get("model") or obj.get("estimator") or obj.get("rf") or obj.get("pipeline")
            _AQI_IMPUTER = obj.get("imputer")
            _AQI_FEATURES = obj.get("features") or obj.get("feature_names") or MODEL_FEATURES
            logger.info("Loaded package dict: model=%s, imputer=%s, features=%s", 
                        type(_AQI_MODEL_OBJ).__name__ if _AQI_MODEL_OBJ else None,
                        type(_AQI_IMPUTER).__name__ if _AQI_IMPUTER else None,
                        _AQI_FEATURES)
        else:
            # plain model object
            _AQI_MODEL_OBJ = obj
            _AQI_IMPUTER = None
            _AQI_FEATURES = MODEL_FEATURES
            logger.info("Loaded plain model object: %s", type(_AQI_MODEL_OBJ).__name__)

        # Defensive defaults
        if _AQI_FEATURES is None:
            _AQI_FEATURES = MODEL_FEATURES

        # Suggest garbage collect to reduce peak allocations
        gc.collect()
        return True

    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        _AQI_MODEL_OBJ = None
        _AQI_IMPUTER = None
        _AQI_FEATURES = MODEL_FEATURES
        return False

def get_model(mmap_mode: Optional[str] = "r") -> Optional[Any]:
    """
    Public: ensure model is loaded and return the model object (or None).
    Use mmap_mode='r' for memory-mapped load (recommended on small instances).
    """
    global _AQI_MODEL_OBJ
    if _AQI_MODEL_OBJ is not None:
        return _AQI_MODEL_OBJ

    ok = load_model(mmap_mode=mmap_mode)
    if not ok:
        logger.error("get_model: model could not be loaded")
        return None
    return _AQI_MODEL_OBJ

# -----------------------
# Prediction function (uses lazy loader)
# -----------------------
def predict_current_aqi(data: Dict[str, Any]) -> Optional[float]:
    """
    Predicts AQI. `data` should contain keys for all model features.
    This will lazily load the model (memory-mapped if possible).
    """
    model = get_model(mmap_mode="r")
    if model is None:
        logger.error("Prediction aborted: model not loaded")
        return None

    # Which features does the model expect?
    features = _AQI_FEATURES or MODEL_FEATURES

    # Validate / build input row
    try:
        row = []
        for feat in features:
            if feat not in data or data[feat] is None:
                logger.error("Missing feature for prediction: %s", feat)
                return None
            try:
                row.append(float(data[feat]))
            except Exception:
                logger.error("Invalid value for feature %s: %s", feat, data.get(feat))
                return None

        X = pd.DataFrame([row], columns=features)

        # Apply imputer if present
        if _AQI_IMPUTER is not None:
            try:
                X_trans = _AQI_IMPUTER.transform(X)
                X = pd.DataFrame(X_trans, columns=features)
            except Exception as e:
                logger.exception("Imputer transform failed: %s", e)
                return None
        else:
            if X.isna().any().any():
                logger.error("Input contains NaN and no imputer is present")
                return None

        # Predict
        if hasattr(model, "predict"):
            pred = model.predict(X)
        elif isinstance(model, dict) and "model" in model and hasattr(model["model"], "predict"):
            pred = model["model"].predict(X)
        else:
            logger.error("Loaded object isn't a predictor: %s", type(model))
            return None

        predicted = float(pred[0])
        logger.info("Predicted AQI: %s", predicted)
        return round(predicted, 2)

    except Exception as e:
        logger.exception("Unexpected error during prediction: %s", e)
        return None

# Sub-index and helper functions (unchanged, keep original logic)
def get_pm25_subindex(x):
    try: x = float(x)
    except (ValueError, TypeError): return 0
    if x <= 30: return x * 50 / 30
    elif x <= 60: return 50 + (x - 30) * 50 / 30
    elif x <= 90: return 100 + (x - 60) * 100 / 30
    elif x <= 120: return 200 + (x - 90) * 100 / 30
    elif x <= 250: return 300 + (x - 120) * 100 / 130
    elif x > 250: return 400 + (x - 250) * 100 / 130
    else: return 0

def get_pm10_subindex(x):
    try: x = float(x)
    except (ValueError, TypeError): return 0
    if x <= 50: return x
    elif x <= 100: return x
    elif x <= 250: return 100 + (x - 100) * 100 / 150
    elif x <= 350: return 200 + (x - 250)
    elif x <= 430: return 300 + (x - 350) * 100 / 80
    elif x > 430: return 400 + (x - 430) * 100 / 80
    else: return 0

def get_so2_subindex(x):
    try: x = float(x)
    except (ValueError, TypeError): return 0
    if x <= 40: return x * 50 / 40
    elif x <= 80: return 50 + (x - 40) * 50 / 40
    elif x <= 380: return 100 + (x - 80) * 100 / 300
    elif x <= 800: return 200 + (x - 380) * 100 / 420
    elif x <= 1600: return 300 + (x - 800) * 100 / 800
    elif x > 1600: return 400 + (x - 1600) * 100 / 800
    else: return 0

def get_nox_subindex(x):
    try: x = float(x)
    except (ValueError, TypeError): return 0
    if x <= 40: return x * 50 / 40
    elif x <= 80: return 50 + (x - 40) * 50 / 40
    elif x <= 180: return 100 + (x - 80) * 100 / 100
    elif x <= 280: return 200 + (x - 180) * 100 / 100
    elif x <= 400: return 300 + (x - 280) * 100 / 120
    elif x > 400: return 400 + (x - 400) * 100 / 120
    else: return 0

def get_nh3_subindex(x):
    try: x = float(x)
    except (ValueError, TypeError): return 0
    if x <= 200: return x * 50 / 200
    elif x <= 400: return 50 + (x - 200) * 50 / 200
    elif x <= 800: return 100 + (x - 400) * 100 / 400
    elif x <= 1200: return 200 + (x - 800) * 100 / 400
    elif x <= 1800: return 300 + (x - 1200) * 100 / 600
    elif x > 1800: return 400 + (x - 1800) * 100 / 600
    else: return 0

def get_co_subindex(x):
    try:
        x_mg = float(x) / 1000.0
    except (ValueError, TypeError): return 0
    if x_mg <= 1.0: return x_mg * 50 / 1.0
    elif x_mg <= 2.0: return 50 + (x_mg - 1.0) * 50 / 1.0
    elif x_mg <= 10.0: return 100 + (x_mg - 2.0) * 100 / 8.0
    elif x_mg <= 17.0: return 200 + (x_mg - 10.0) * 100 / 7.0
    elif x_mg <= 34.0: return 300 + (x_mg - 17.0) * 100 / 17.0
    elif x_mg > 34.0: return 400 + (x_mg - 34.0) * 100 / 17.0
    else: return 0

def get_o3_subindex(x):
    try: x = float(x)
    except (ValueError, TypeError): return 0
    if x <= 50: return x * 50 / 50
    elif x <= 100: return 50 + (x - 50) * 50 / 50
    elif x <= 168: return 100 + (x - 100) * 100 / 68
    elif x <= 208: return 200 + (x - 168) * 100 / 40
    elif x <= 748: return 300 + (x - 208) * 100 / 540
    elif x > 748: return 400 + (x - 748) * 100 / 540
    else: return 0

def calculate_all_subindices(data: Dict[str, Any]) -> Dict[str, float]:
    subindices = {}
    subindices['PM2.5'] = get_pm25_subindex(data.get('PM2.5'))
    subindices['PM10'] = get_pm10_subindex(data.get('PM10'))
    subindices['SO2'] = get_so2_subindex(data.get('SO2'))
    subindices['NOx'] = get_nox_subindex(data.get('NOx'))
    subindices['NH3'] = get_nh3_subindex(data.get('NH3'))
    subindices['CO'] = get_co_subindex(data.get('CO'))
    subindices['O3'] = get_o3_subindex(data.get('O3'))
    return {k: round(v, 1) for k, v in subindices.items() if v is not None and v > 0}
