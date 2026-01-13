# ml_handler.py
import os
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Set up logger basic config if not already configured elsewhere
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Default model features (fallback)
MODEL_FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Globals populated by load_model()
AQI_PREDICTOR_MODEL: Optional[Any] = None
AQI_IMPUTER: Optional[Any] = None
AQI_MODEL_FEATURES: Optional[list] = None

# Environment-configurable model URL (set MODEL_URL in Render service if you want to override)
DEFAULT_MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/SamarthBurkul/AirWatch-/releases/download/v1.0.0/random_forest_model.pkl"
)
MODEL_DIR = Path(__file__).resolve().parent / "ml_models"
MODEL_FILENAME = "random_forest_model.pkl"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME


def _download_file_requests(url: str, dest: Path, timeout: int = 60) -> bool:
    """
    Try to download using requests (streamed). Returns True on success.
    """
    try:
        import requests
    except ImportError:
        logger.debug("requests not installed; falling back to urllib.")
        return False

    try:
        logger.info("Downloading ML model from %s to %s", url, dest)
        # download to a temp file first
        tmp_path = dest.with_suffix(".tmp")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        tmp_path.replace(dest)
        logger.info("Model downloaded to %s", dest)
        return True
    except Exception as e:
        logger.exception("Failed to download model via requests: %s", e)
        return False


def _download_file_urllib(url: str, dest: Path, timeout: int = 60) -> bool:
    """
    Fallback download using urllib.request. Returns True on success.
    """
    try:
        import urllib.request
    except Exception as e:
        logger.debug("urllib not available: %s", e)
        return False

    try:
        logger.info("Downloading ML model (urllib) from %s to %s", url, dest)
        tmp_path = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(dest)
        logger.info("Model downloaded to %s (urllib)", dest)
        return True
    except Exception as e:
        logger.exception("Failed to download model via urllib: %s", e)
        return False


def ensure_model_file(path: Path = MODEL_PATH, url: str = DEFAULT_MODEL_URL) -> bool:
    """
    Ensure the model file exists at `path`. If missing, try to download it.
    Returns True if file exists after the call.
    """
    try:
        if path.exists() and path.is_file():
            logger.debug("Model file already exists: %s", path)
            return True

        # create directory
        path.parent.mkdir(parents=True, exist_ok=True)

        # Try requests first, then urllib fallback
        if _download_file_requests(url, path):
            return True
        if _download_file_urllib(url, path):
            return True

        logger.error("Failed to download model from %s. File does not exist at %s", url, path)
        return False
    except Exception as e:
        logger.exception("Unexpected error in ensure_model_file: %s", e)
        return False


def load_model(path: Optional[str] = None) -> None:
    """
    Load the trained model package saved by train_random_forest.py.

    The training script saves a dict with keys: 'model', 'imputer', 'features'.
    This loader is backward-compatible with older pickles that contain a plain model object.
    """
    global AQI_PREDICTOR_MODEL, AQI_IMPUTER, AQI_MODEL_FEATURES

    # Resolve path (priority: function arg -> MODEL_PATH)
    if path is None:
        path_obj = MODEL_PATH
    else:
        path_obj = Path(path)

    # If file missing, attempt download
    if not path_obj.exists():
        logger.warning("Model path %s does not exist; attempting to download from %s", path_obj, DEFAULT_MODEL_URL)
        ok = ensure_model_file(path_obj, DEFAULT_MODEL_URL)
        if not ok:
            logger.error("❌ Error: Model file is not available at %s and download failed.", path_obj)
            AQI_PREDICTOR_MODEL = None
            AQI_IMPUTER = None
            AQI_MODEL_FEATURES = MODEL_FEATURES
            return

    # Try loading
    try:
        with open(path_obj, "rb") as f:
            pkg = pickle.load(f)

        if isinstance(pkg, dict):
            AQI_PREDICTOR_MODEL = pkg.get("model")
            AQI_IMPUTER = pkg.get("imputer")
            AQI_MODEL_FEATURES = pkg.get("features", MODEL_FEATURES)
            logger.info("✅ AQI Predictor package loaded from %s (model + imputer).", path_obj)
        else:
            AQI_PREDICTOR_MODEL = pkg
            AQI_IMPUTER = None
            AQI_MODEL_FEATURES = MODEL_FEATURES
            logger.info("✅ AQI Predictor model (plain object) loaded from %s.", path_obj)

        if AQI_MODEL_FEATURES is None:
            AQI_MODEL_FEATURES = MODEL_FEATURES
        logger.debug("Model feature list: %s", AQI_MODEL_FEATURES)

    except FileNotFoundError:
        logger.error("❌ Error: %s not found. AQI predictor will not work.", path_obj)
        AQI_PREDICTOR_MODEL = None
        AQI_IMPUTER = None
        AQI_MODEL_FEATURES = MODEL_FEATURES
    except Exception as e:
        logger.exception("❌ Error loading model at %s: %s", path_obj, e)
        AQI_PREDICTOR_MODEL = None
        AQI_IMPUTER = None
        AQI_MODEL_FEATURES = MODEL_FEATURES


# Auto-load at import-time (keeps previous behaviour but ensures model gets downloaded if missing)
try:
    load_model()
except Exception:
    # load_model logs errors already
    pass


def get_aqi_category(aqi: float) -> Dict[str, str]:
    """Classifies the AQI value and returns a category, description, and color code."""
    try:
        aqi_val = float(aqi)
    except (ValueError, TypeError):
        return {
            "category": "N/A",
            "description": "AQI data invalid.",
            "color_class": "bg-slate-500/20 text-slate-300 border-slate-500",
            "chartColor": "#64748b"
        }
    if aqi_val <= 50:
        return {"category": "Good", "description": "Minimal impact.", "color_class": "bg-green-500/20 text-green-300 border-green-500", "chartColor": "#34d399"}
    elif aqi_val <= 100:
        return {"category": "Satisfactory", "description": "Minor breathing discomfort.", "color_class": "bg-yellow-500/20 text-yellow-300 border-yellow-500", "chartColor": "#f59e0b"}
    elif aqi_val <= 200:
        return {"category": "Moderate", "description": "Breathing discomfort to sensitive groups.", "color_class": "bg-orange-500/20 text-orange-300 border-orange-500", "chartColor": "#f97316"}
    elif aqi_val <= 300:
        return {"category": "Poor", "description": "Breathing discomfort to most people.", "color_class": "bg-red-500/20 text-red-300 border-red-500", "chartColor": "#ef4444"}
    elif aqi_val <= 400:
        return {"category": "Very Poor", "description": "Respiratory illness on prolonged exposure.", "color_class": "bg-purple-500/20 text-purple-300 border-purple-500", "chartColor": "#a855f7"}
    else:
        return {"category": "Severe", "description": "Serious health effects.", "color_class": "bg-rose-800/20 text-rose-400 border-rose-700", "chartColor": "#be123c"}


def predict_current_aqi(data: Dict[str, Any]) -> Optional[float]:
    """
    Predicts AQI using the loaded Random Forest model.
    - Expects a dict with keys matching AQI_MODEL_FEATURES (model feature order is enforced).
    - Uses saved AQI_IMPUTER if available to handle missing values.
    Returns the predicted AQI (rounded to 2 decimals) or None on failure.
    """
    global AQI_PREDICTOR_MODEL, AQI_IMPUTER, AQI_MODEL_FEATURES

    if not AQI_PREDICTOR_MODEL:
        logger.error("AQI prediction failed: Model not loaded.")
        return None

    if AQI_MODEL_FEATURES is None:
        AQI_MODEL_FEATURES = MODEL_FEATURES

    try:
        input_values = {}
        missing_features, invalid_features = [], []
        for feature in AQI_MODEL_FEATURES:
            if feature not in data:
                missing_features.append(feature)
                continue
            try:
                input_values[feature] = [float(data[feature])]
            except (ValueError, TypeError):
                invalid_features.append(feature)

        if missing_features:
            logger.error("Prediction failed: Missing features: %s", missing_features)
            return None
        if invalid_features:
            logger.error("Prediction failed: Invalid features: %s", invalid_features)
            return None

        input_df = pd.DataFrame(input_values, columns=AQI_MODEL_FEATURES)
        logger.debug("Input DataFrame for prediction:\n%s", input_df)

        if AQI_IMPUTER is not None:
            try:
                transformed = AQI_IMPUTER.transform(input_df)
                input_df = pd.DataFrame(transformed, columns=AQI_MODEL_FEATURES)
            except Exception as e:
                logger.exception("Error applying imputer to input: %s", e)
                return None
        else:
            if input_df.isna().any().any():
                logger.error("Prediction failed: missing input values and no imputer available.")
                return None

        prediction = AQI_PREDICTOR_MODEL.predict(input_df)
        logger.info("Raw prediction: %s", prediction)
        predicted_aqi = round(float(prediction[0]), 2)
        logger.info("Predicted AQI: %s", predicted_aqi)
        return predicted_aqi
    except Exception as e:
        logger.exception("Unexpected Error during AQI prediction: %s", e)
        return None


# --- START: Functions to calculate individual sub-indices ---
# (Kept intact from original logic)

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
    """ Calculates all relevant sub-indices from a dictionary of pollutant values. """
    subindices = {}
    subindices['PM2.5'] = get_pm25_subindex(data.get('PM2.5'))
    subindices['PM10'] = get_pm10_subindex(data.get('PM10'))
    subindices['SO2'] = get_so2_subindex(data.get('SO2'))
    subindices['NOx'] = get_nox_subindex(data.get('NOx'))
    subindices['NH3'] = get_nh3_subindex(data.get('NH3'))
    subindices['CO'] = get_co_subindex(data.get('CO'))
    subindices['O3'] = get_o3_subindex(data.get('O3'))

    relevant_subindices = {k: round(v, 1) for k, v in subindices.items() if v is not None and v > 0}
    return relevant_subindices

# End of file
