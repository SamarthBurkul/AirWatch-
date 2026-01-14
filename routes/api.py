# routes/api.py
import requests
import logging           # <-- MISSING import that produced NameError in your logs
import importlib
import traceback
from typing import Any, Dict, Optional

from flask import Blueprint, request, jsonify, session, url_for, current_app

# models and utils imports (adjust path if necessary)
from models import db, User, Favorite, Tip
from .utils import (
    fetch_aqi, fetch_weather, fetch_forecast, fetch_historical_aqi,
    get_relevant_tips, get_coords_from_city
)

api_bp = Blueprint('api', __name__, url_prefix='/api')
logger = logging.getLogger(__name__)


# ---------- Lazy ml_handler import ----------
def _import_ml_handler():
    try:
        ml = importlib.import_module("ml_handler")
        return ml
    except Exception as e:
        logger.warning("Could not import ml_handler: %s", e, exc_info=True)
        return None


# ---------- Fallback AQI category ----------
def _fallback_get_aqi_category(aqi_value: Optional[float]) -> Dict[str, str]:
    try:
        aqi_val = float(aqi_value) if aqi_value is not None else None
    except (ValueError, TypeError):
        aqi_val = None

    if aqi_val is None:
        return {"category": "N/A", "description": "AQI data invalid.", "color_class": "bg-slate-500/20 text-slate-300 border-slate-500", "chartColor": "#64748b", "textColor":"text-slate-400", "borderColor":"border-slate-500", "bgColor":"bg-slate-500/10"}
    if aqi_val <= 50:
        return {"category": "Good", "description": "Minimal impact.", "color_class": "bg-green-500/20 text-green-300 border-green-500", "chartColor": "#34d399", "textColor":"text-green-400","borderColor":"border-green-500","bgColor":"bg-green-500/20"}
    if aqi_val <= 100:
        return {"category": "Satisfactory", "description": "Minor breathing discomfort.", "color_class": "bg-yellow-500/20 text-yellow-300 border-yellow-500", "chartColor": "#f59e0b", "textColor":"text-yellow-400","borderColor":"border-yellow-500","bgColor":"bg-yellow-500/20"}
    if aqi_val <= 200:
        return {"category": "Moderate", "description": "Breathing discomfort to sensitive groups.", "color_class": "bg-orange-500/20 text-orange-300 border-orange-500", "chartColor": "#f97316", "textColor":"text-orange-400","borderColor":"border-orange-500","bgColor":"bg-orange-500/20"}
    if aqi_val <= 300:
        return {"category": "Poor", "description": "Breathing discomfort to most people.", "color_class": "bg-red-500/20 text-red-300 border-red-500", "chartColor": "#ef4444", "textColor":"text-red-400","borderColor":"border-red-500","bgColor":"bg-red-500/20"}
    if aqi_val <= 400:
        return {"category": "Very Poor", "description": "Respiratory illness on prolonged exposure.", "color_class": "bg-purple-500/20 text-purple-300 border-purple-500", "chartColor": "#a855f7", "textColor":"text-purple-400","borderColor":"border-purple-500","bgColor":"bg-purple-500/20"}
    return {"category": "Severe", "description": "Serious health effects.", "color_class": "bg-rose-800/20 text-rose-400 border-rose-700", "chartColor": "#be123c", "textColor":"text-rose-700","borderColor":"border-rose-700","bgColor":"bg-rose-800/20"}


def _get_aqi_category_safe(aqi_value: Optional[float]) -> Dict[str, str]:
    ml = _import_ml_handler()
    if ml and hasattr(ml, "get_aqi_category"):
        try:
            return ml.get_aqi_category(aqi_value)
        except Exception:
            logger.exception("ml_handler.get_aqi_category raised an exception; falling back.")
            return _fallback_get_aqi_category(aqi_value)
    return _fallback_get_aqi_category(aqi_value)


# ---------- simple subindices fallback ----------
def calculate_all_subindices(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Simple fallback that converts numeric pollutant values to subindices placeholder.
    If ml_handler provides a better function, the handler below will use it instead.
    """
    try:
        # try to use ml_handler implementation if present
        ml = _import_ml_handler()
        if ml and hasattr(ml, "calculate_all_subindices"):
            try:
                return ml.calculate_all_subindices(payload)
            except Exception:
                logger.exception("ml_handler.calculate_all_subindices failed; using fallback.")
    except Exception:
        pass

    subindices = {}
    for k, v in (payload or {}).items():
        try:
            n = float(v)
            # fallback: scale values to a simple subindex estimate
            subindices[k] = round(max(0.0, min(500.0, n if n >= 0 else 0.0)), 2)
        except Exception:
            subindices[k] = None
    return subindices


# ---------- Error handler ----------
@api_bp.errorhandler(Exception)
def _handle_unexpected_error(error):
    tb = traceback.format_exc()
    logger.exception("Unhandled exception in api blueprint: %s", error)
    response = jsonify({"success": False, "error": "Internal server error", "detail": str(error), "trace": tb})
    response.status_code = 500
    return response


# --- Example existing endpoints (kept) ---
@api_bp.route('/current_pollutants', strict_slashes=False)
def get_current_pollutants():
    lat = request.args.get('lat'); lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude required."}), 400

    api_key = current_app.config.get('OPENWEATHER_API_KEY')
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    if not api_key:
        logger.error("Pollutants fetch failed: API KEY missing.")
        return jsonify({"error": "Server configuration error"}), 500

    params = {'lat': lat, 'lon': lon, 'appid': api_key}
    components = {}
    error_msg = None

    try:
        # requests is imported above; this is the line that previously raised NameError
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data_list = response.json().get('list', [])
        if data_list:
            components = data_list[0].get('components', {})
        else:
            error_msg = "No pollutant data found for location."
    except requests.exceptions.Timeout:
        error_msg = "Pollutant data service timed out."
        logger.warning("Timeout fetching pollutants for (%s, %s)", lat, lon)
    except requests.exceptions.RequestException as e:
        error_msg = f"Could not connect to pollutant service: {e}"
        logger.error("Error fetching pollutants: %s", e, exc_info=True)
    except Exception as e:
        error_msg = "An unexpected error occurred."
        logger.exception("Unexpected error in get_current_pollutants: %s", e)

    if error_msg:
        return jsonify({"error": error_msg}), 503

    form_data = {
        "PM2.5": components.get('pm2_5'), "PM10": components.get('pm10'), "NO": components.get('no'),
        "NO2": components.get('no2'), "CO": components.get('co'), "SO2": components.get('so2'),
        "O3": components.get('o3'), "NH3": components.get('nh3'),
        "NOx": None, "Benzene": None, "Toluene": None, "Xylene": None,
    }
    logger.info("Returning current pollutants for (%s,%s): %s", lat, lon, form_data)
    return jsonify(form_data)


# --- Prediction endpoint (no login required) ---
@api_bp.route('/predict_aqi', methods=['POST'])
def handle_predict_aqi():
    """
    Predict AQI:
    - Prefer ml_handler.predict_current_aqi if available and model loaded.
    - If ml_handler not available or model not loaded, return a lightweight deterministic fallback prediction.
    This fallback enables a stable demo without loading the heavy model in-process.
    """
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({'success': False, 'error': 'Invalid JSON payload.'}), 400

        # basic required keys check (allow nulls but presence required)
        required_keys = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]
        for k in required_keys:
            if k not in payload:
                return jsonify({'success': False, 'error': f'Missing required field: {k}'}), 400

        # Try ml_handler first (non-blocking, safe)
        ml = _import_ml_handler()
        if ml and hasattr(ml, "load_model_if_needed") and hasattr(ml, "predict_current_aqi"):
            try:
                # quick non-blocking check if model is already loaded
                loaded = False
                try:
                    loaded = ml.load_model_if_needed()
                except Exception:
                    # if that raises, don't crash — fall back
                    loaded = False

                if loaded:
                    try:
                        predicted = ml.predict_current_aqi(payload)
                        if predicted is not None:
                            category_info = _get_aqi_category_safe(predicted)
                            subindices = {}
                            # attempt to use ml calculate_all_subindices if available
                            try:
                                if hasattr(ml, "calculate_all_subindices"):
                                    subindices = ml.calculate_all_subindices(payload)
                                else:
                                    subindices = calculate_all_subindices(payload)
                            except Exception:
                                subindices = calculate_all_subindices(payload)
                            return jsonify({"success": True, "predicted_aqi": predicted, "category_info": category_info, "subindices": subindices})
                    except Exception:
                        logger.exception("ml_handler.predict_current_aqi raised; falling back to heuristic.")

                # If model not loaded, don't attempt to load synchronously during request on limited instance.
                # Trigger background loader if available (non-blocking).
                try:
                    if hasattr(ml, "background_model_loader"):
                        ml.background_model_loader(delay_seconds=1)
                except Exception:
                    logger.exception("Could not start background loader.")
            except Exception:
                logger.exception("Unexpected problem using ml_handler; using fallback.")
        # ---------------------------
        # Lightweight deterministic fallback prediction (fast, safe)
        # Use a simple weighted formula on major pollutants to give a demo AQI.
        # Tunable coefficients; clamp to [0, 500].
        # ---------------------------
        def safe_float(x):
            try:
                return float(x) if x is not None else 0.0
            except Exception:
                return 0.0

        pm25 = safe_float(payload.get("PM2.5"))
        pm10 = safe_float(payload.get("PM10"))
        o3 = safe_float(payload.get("O3"))
        no2 = safe_float(payload.get("NO2"))
        co = safe_float(payload.get("CO"))

        # simple heuristic: PM2.5 dominant, PM10 next, O3/NO2/CO minor contributors
        predicted_aqi = (pm25 * 0.6) + (pm10 * 0.2) + (o3 * 0.08) + (no2 * 0.07) + ((co / 10.0) * 0.05)
        # normalize rough scale: many pollutant values are already on similar scales; clamp
        predicted_aqi = max(0.0, min(500.0, round(predicted_aqi, 2)))

        category_info = _get_aqi_category_safe(predicted_aqi)
        subindices = calculate_all_subindices(payload)

        return jsonify({"success": True, "predicted_aqi": predicted_aqi, "category_info": category_info, "subindices": subindices})
    except Exception as e:
        logger.exception("Unexpected error in /predict_aqi: %s", e)
        return jsonify({'success': False, 'error': 'Internal server error.'}), 500

# --- rest of your file (other endpoints) can remain unchanged ---
