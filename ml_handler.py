# ml_handler.py
import os
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# NOTE: keep imports minimal at module import time to reduce startup memory.
# Heavy libs (joblib, pandas) are imported lazily inside functions.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

MODEL_FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Global placeholders; model will be loaded lazily by load_model_if_needed()
AQI_PREDICTOR_MODEL: Optional[Any] = None
AQI_IMPUTER: Optional[Any] = None
AQI_MODEL_FEATURES: Optional[list] = None

DEFAULT_MODEL_URL = os.environ.get(
    "AQI_MODEL_URL",
    "https://github.com/SamarthBurkul/AirWatch-/releases/download/v1.1.0/random_forest_model.pkl"
)
MODEL_DIR = Path(__file__).resolve().parent / "ml_models"
MODEL_FILENAME = "random_forest_model.pkl"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME


def _download_file_requests(url: str, dest: Path, timeout: int = 60) -> bool:
    try:
        import requests
    except Exception:
        return False
    try:
        logger.info("Downloading ML model from %s to %s", url, dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        tmp.replace(dest)
        logger.info("Model downloaded to %s", dest)
        return True
    except Exception as e:
        logger.exception("requests download failed: %s", e)
        return False


def ensure_model_file(path: Path = MODEL_PATH, url: str = DEFAULT_MODEL_URL) -> bool:
    """Ensure the model file exists; if not attempt to download it. Return True if file exists."""
    try:
        if path.exists() and path.is_file():
            logger.debug("Model file present at %s", path)
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        if _download_file_requests(url, path):
            return True
        logger.error("Failed to obtain model file from %s", url)
        return False
    except Exception as e:
        logger.exception("Unexpected error ensuring model file: %s", e)
        return False


def _load_with_joblib(path: Path):
    """Try joblib.load with memory-mapping for large numpy arrays."""
    try:
        import joblib
        logger.info("Attempting joblib.load(..., mmap_mode='r') for %s", path)
        loaded = joblib.load(path, mmap_mode='r')  # memory-map large arrays
        return loaded
    except Exception as e:
        logger.debug("joblib load failed: %s", e)
        return None


def _load_with_pickle(path: Path):
    """Fallback pickle loader (may use more memory)."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.exception("pickle load failed: %s", e)
        return None


def load_model_if_needed(path: Optional[Path] = None) -> bool:
    """
    Ensure model globals are populated. This is called lazily when prediction is requested.
    Returns True if model is ready.
    """
    global AQI_PREDICTOR_MODEL, AQI_IMPUTER, AQI_MODEL_FEATURES

    if AQI_PREDICTOR_MODEL is not None:
        return True

    path = path or MODEL_PATH
    if not path.exists():
        logger.info("Model not found locally; attempting download from %s", DEFAULT_MODEL_URL)
        if not ensure_model_file(path, DEFAULT_MODEL_URL):
            logger.error("Model file unavailable and download failed.")
            return False

    # Try joblib mmap load first (very memory efficient for large numpy arrays).
    loaded = _load_with_joblib(path)
    if loaded is None:
        # fallback to pickle
        loaded = _load_with_pickle(path)
        if loaded is None:
            logger.error("All model loading attempts failed.")
            return False

    # loaded may be a dict (model + imputer + features) or a plain model object
    if isinstance(loaded, dict):
        AQI_PREDICTOR_MODEL = loaded.get("model") or loaded.get("estimator") or None
        AQI_IMPUTER = loaded.get("imputer")
        AQI_MODEL_FEATURES = loaded.get("features") or MODEL_FEATURES
        logger.info("Loaded model package from %s (dict)", path)
    else:
        AQI_PREDICTOR_MODEL = loaded
        AQI_IMPUTER = None
        AQI_MODEL_FEATURES = MODEL_FEATURES
        logger.info("Loaded plain model object from %s", path)

    if AQI_MODEL_FEATURES is None:
        AQI_MODEL_FEATURES = MODEL_FEATURES

    return AQI_PREDICTOR_MODEL is not None


def get_aqi_category(aqi: float) -> Dict[str, str]:
    try:
        aqi_val = float(aqi)
    except (ValueError, TypeError):
        aqi_val = None
    if aqi_val is None:
        return {"category": "N/A", "description": "AQI data invalid.", "color_class": "bg-slate-500/20 text-slate-300 border-slate-500", "chartColor": "#64748b"}
    if aqi_val <= 50:
        return {"category": "Good", "description": "Minimal impact.", "color_class": "bg-green-500/20 text-green-300 border-green-500", "chartColor": "#34d399"}
    if aqi_val <= 100:
        return {"category": "Satisfactory", "description": "Minor breathing discomfort.", "color_class": "bg-yellow-500/20 text-yellow-300 border-yellow-500", "chartColor": "#f59e0b"}
    if aqi_val <= 200:
        return {"category": "Moderate", "description": "Breathing discomfort to sensitive groups.", "color_class": "bg-orange-500/20 text-orange-300 border-orange-500", "chartColor": "#f97316"}
    if aqi_val <= 300:
        return {"category": "Poor", "description": "Breathing discomfort to most people.", "color_class": "bg-red-500/20 text-red-300 border-red-500", "chartColor": "#ef4444"}
    if aqi_val <= 400:
        return {"category": "Very Poor", "description": "Respiratory illness on prolonged exposure.", "color_class": "bg-purple-500/20 text-purple-300 border-purple-500", "chartColor": "#a855f7"}
    return {"category": "Severe", "description": "Serious health effects.", "color_class": "bg-rose-800/20 text-rose-400 border-rose-700", "chartColor": "#be123c"}


def predict_current_aqi(data: Dict[str, Any]) -> Optional[float]:
    """Lazy-load model on first prediction and predict. Uses pandas locally only inside this function."""
    global AQI_PREDICTOR_MODEL, AQI_IMPUTER, AQI_MODEL_FEATURES

    # Ensure model is loaded (lazy)
    ok = load_model_if_needed()
    if not ok:
        logger.error("Model not available for prediction.")
        return None

    try:
        import pandas as pd  # local import reduces startup memory
    except Exception as e:
        logger.exception("pandas not available: %s", e)
        return None

    if AQI_MODEL_FEATURES is None:
        AQI_MODEL_FEATURES = MODEL_FEATURES

    # Build input DataFrame in the required order
    input_values = {}
    missing, invalid = [], []
    for feat in AQI_MODEL_FEATURES:
        if feat not in data:
            missing.append(feat)
            continue
        try:
            input_values[feat] = [float(data[feat])]
        except Exception:
            invalid.append(feat)

    if missing:
        logger.error("Missing features for prediction: %s", missing)
        return None
    if invalid:
        logger.error("Invalid numeric values for features: %s", invalid)
        return None

    input_df = pd.DataFrame(input_values, columns=AQI_MODEL_FEATURES)

    try:
        if AQI_IMPUTER is not None:
            input_df = pd.DataFrame(AQI_IMPUTER.transform(input_df), columns=AQI_MODEL_FEATURES)
        elif input_df.isna().any().any():
            logger.error("Missing values and no imputer available.")
            return None

        pred = AQI_PREDICTOR_MODEL.predict(input_df)
        predicted = round(float(pred[0]), 2)
        logger.info("Predicted AQI: %s", predicted)
        return predicted
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return None


# Sub-index functions (unchanged)
def get_pm25_subindex(x): ...
def get_pm10_subindex(x): ...
def get_so2_subindex(x): ...
def get_nox_subindex(x): ...
def get_nh3_subindex(x): ...
def get_co_subindex(x): ...
def get_o3_subindex(x): ...

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
