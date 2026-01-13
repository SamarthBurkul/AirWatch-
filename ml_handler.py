# ml_handler.py
import os
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("ml_handler")

# Features expected by the model (fallback)
MODEL_FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Globals populated by load_model()
AQI_PREDICTOR_MODEL: Optional[Any] = None
AQI_IMPUTER: Optional[Any] = None
AQI_MODEL_FEATURES: Optional[list] = None

# Default model URL / local path (can be overridden with AQI_MODEL_URL env)
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
        logger.warning("requests download failed: %s", e)
        return False


def _download_file_urllib(url: str, dest: Path) -> bool:
    try:
        import urllib.request
        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        logger.info("Downloaded model (urllib) to %s", dest)
        return True
    except Exception as e:
        logger.warning("urllib download failed: %s", e)
        return False


def ensure_model_file(path: Path = MODEL_PATH, url: str = DEFAULT_MODEL_URL) -> bool:
    try:
        if path.exists() and path.is_file():
            logger.debug("Model exists: %s", path)
            return True
        path.parent.mkdir(parents=True, exist_ok=True)

        # Try requests then urllib
        if _download_file_requests(url, path):
            return True
        if _download_file_urllib(url, path):
            return True

        logger.error("Model not available at %s and download failed", url)
        return False
    except Exception as e:
        logger.exception("ensure_model_file error: %s", e)
        return False


def load_model(path: Optional[str] = None) -> None:
    global AQI_PREDICTOR_MODEL, AQI_IMPUTER, AQI_MODEL_FEATURES
    path_obj = Path(path) if path else MODEL_PATH

    if not path_obj.exists():
        logger.info("Model missing locally; will attempt to download from %s", DEFAULT_MODEL_URL)
        ok = ensure_model_file(path_obj, os.environ.get("AQI_MODEL_URL", DEFAULT_MODEL_URL))
        if not ok:
            logger.warning("Model not available; continuing without predictor.")
            AQI_PREDICTOR_MODEL = None
            AQI_IMPUTER = None
            AQI_MODEL_FEATURES = MODEL_FEATURES
            return

    try:
        # Try joblib first (if file saved with joblib); else pickle
        try:
            import joblib
            pkg = joblib.load(path_obj)
        except Exception:
            with open(path_obj, "rb") as fh:
                pkg = pickle.load(fh)

        if isinstance(pkg, dict):
            AQI_PREDICTOR_MODEL = pkg.get("model")
            AQI_IMPUTER = pkg.get("imputer")
            AQI_MODEL_FEATURES = pkg.get("features", MODEL_FEATURES)
            logger.info("Loaded model package from %s", path_obj)
        else:
            AQI_PREDICTOR_MODEL = pkg
            AQI_IMPUTER = None
            AQI_MODEL_FEATURES = MODEL_FEATURES
            logger.info("Loaded plain model from %s", path_obj)

        if AQI_MODEL_FEATURES is None:
            AQI_MODEL_FEATURES = MODEL_FEATURES
        logger.debug("Model features: %s", AQI_MODEL_FEATURES)

    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        AQI_PREDICTOR_MODEL = None
        AQI_IMPUTER = None
        AQI_MODEL_FEATURES = MODEL_FEATURES


# Do NOT do heavy downloads at import time; just try to load if present.
try:
    if MODEL_PATH.exists():
        load_model()
    else:
        logger.info("Model file not present on import. Will download on-demand.")
except Exception:
    logger.exception("Silent failure during import-time load_model()")


def get_aqi_category(aqi: float) -> Dict[str, str]:
    try:
        aqi_val = float(aqi)
    except (ValueError, TypeError):
        return {"category": "N/A", "description": "AQI data invalid.", "color_class": "bg-slate-500/20 text-slate-300 border-slate-500", "chartColor": "#64748b"}
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
    global AQI_PREDICTOR_MODEL, AQI_IMPUTER, AQI_MODEL_FEATURES
    if AQI_PREDICTOR_MODEL is None:
        logger.error("Model not loaded; cannot predict.")
        return None

    if AQI_MODEL_FEATURES is None:
        AQI_MODEL_FEATURES = MODEL_FEATURES

    try:
        input_values = {}
        for feature in AQI_MODEL_FEATURES:
            input_values[feature] = float(data.get(feature, 0.0))

        df = pd.DataFrame([input_values], columns=AQI_MODEL_FEATURES)
        if AQI_IMPUTER is not None:
            try:
                transformed = AQI_IMPUTER.transform(df)
                df = pd.DataFrame(transformed, columns=AQI_MODEL_FEATURES)
            except Exception as e:
                logger.warning("Imputer transform failed: %s", e)
                return None

        pred = AQI_PREDICTOR_MODEL.predict(df)
        result = round(float(pred[0]), 2)
        logger.info("Predicted AQI: %s", result)
        return result
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return None


# Sub-index functions + aggregator (kept concise)
def get_pm25_subindex(x):
    try: x = float(x)
    except: return 0
    if x <= 30: return x * 50 / 30
    if x <= 60: return 50 + (x - 30) * 50 / 30
    if x <= 90: return 100 + (x - 60) * 100 / 30
    if x <= 120: return 200 + (x - 90) * 100 / 30
    if x <= 250: return 300 + (x - 120) * 100 / 130
    return 400 + (x - 250) * 100 / 130


def get_pm10_subindex(x):
    try: x = float(x)
    except: return 0
    if x <= 100: return x
    if x <= 250: return 100 + (x - 100) * 100 / 150
    if x <= 350: return 200 + (x - 250)
    if x <= 430: return 300 + (x - 350) * 100 / 80
    return 400 + (x - 430) * 100 / 80


def get_so2_subindex(x):
    try: x = float(x)
    except: return 0
    if x <= 40: return x * 50 / 40
    if x <= 80: return 50 + (x - 40) * 50 / 40
    if x <= 380: return 100 + (x - 80) * 100 / 300
    if x <= 800: return 200 + (x - 380) * 100 / 420
    if x <= 1600: return 300 + (x - 800) * 100 / 800
    return 400 + (x - 1600) * 100 / 800


def get_nox_subindex(x):
    try: x = float(x)
    except: return 0
    if x <= 40: return x * 50 / 40
    if x <= 80: return 50 + (x - 40) * 50 / 40
    if x <= 180: return 100 + (x - 80) * 100 / 100
    if x <= 280: return 200 + (x - 180) * 100 / 100
    if x <= 400: return 300 + (x - 280) * 100 / 120
    return 400 + (x - 400) * 100 / 120


def get_nh3_subindex(x):
    try: x = float(x)
    except: return 0
    if x <= 200: return x * 50 / 200
    if x <= 400: return 50 + (x - 200) * 50 / 200
    if x <= 800: return 100 + (x - 400) * 100 / 400
    if x <= 1200: return 200 + (x - 800) * 100 / 400
    if x <= 1800: return 300 + (x - 1200) * 100 / 600
    return 400 + (x - 1800) * 100 / 600


def get_co_subindex(x):
    try: x_mg = float(x) / 1000.0
    except: return 0
    if x_mg <= 1.0: return x_mg * 50 / 1.0
    if x_mg <= 2.0: return 50 + (x_mg - 1.0) * 50 / 1.0
    if x_mg <= 10.0: return 100 + (x_mg - 2.0) * 100 / 8.0
    if x_mg <= 17.0: return 200 + (x_mg - 10.0) * 100 / 7.0
    if x_mg <= 34.0: return 300 + (x_mg - 17.0) * 100 / 17.0
    return 400 + (x_mg - 34.0) * 100 / 17.0


def get_o3_subindex(x):
    try: x = float(x)
    except: return 0
    if x <= 50: return x * 50 / 50
    if x <= 100: return 50 + (x - 50) * 50 / 50
    if x <= 168: return 100 + (x - 100) * 100 / 68
    if x <= 208: return 200 + (x - 168) * 100 / 40
    if x <= 748: return 300 + (x - 208) * 100 / 540
    return 400 + (x - 748) * 100 / 540


def calculate_all_subindices(data: Dict[str, Any]) -> Dict[str, float]:
    subindices = {
        'PM2.5': get_pm25_subindex(data.get('PM2.5', 0)),
        'PM10': get_pm10_subindex(data.get('PM10', 0)),
        'SO2': get_so2_subindex(data.get('SO2', 0)),
        'NOx': get_nox_subindex(data.get('NOx', 0)),
        'NH3': get_nh3_subindex(data.get('NH3', 0)),
        'CO': get_co_subindex(data.get('CO', 0)),
        'O3': get_o3_subindex(data.get('O3', 0))
    }
    return {k: round(v, 1) for k, v in subindices.items() if v is not None and v > 0}
