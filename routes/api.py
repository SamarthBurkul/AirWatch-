# routes/api.py
import logging
import importlib
import traceback
from typing import Optional, Dict

from flask import Blueprint, request, jsonify, session, url_for, current_app

logger = logging.getLogger(__name__)

# Attempt to import ml_handler functions. If unavailable provide safe fallbacks.
try:
    from ml_handler import load_model_if_needed, background_model_loader, predict_current_aqi, get_aqi_category
except Exception as e:
    logger.warning("Could not import ml_handler functions: %s", e, exc_info=True)
    # Safe fallbacks
    def load_model_if_needed(*args, **kwargs):
        return False

    def background_model_loader(*args, **kwargs):
        return None

    predict_current_aqi = None

    # Provide a simple get_aqi_category fallback (used to return same keys frontend expects)
    def get_aqi_category(aqi_val: Optional[float]) -> Dict[str, str]:
        try:
            aqi_num = float(aqi_val) if aqi_val is not None else None
        except Exception:
            aqi_num = None

        if aqi_num is None:
            return {
                "category": "N/A",
                "description": "AQI data invalid.",
                "textColor": "text-slate-400",
                "borderColor": "border-slate-500",
                "bgColor": "bg-slate-500/10",
                "chartColor": "#64748b"
            }
        if aqi_num <= 50:
            return {
                "category": "Good",
                "description": "Minimal impact.",
                "textColor": "text-green-400",
                "borderColor": "border-green-500",
                "bgColor": "bg-green-500/10",
                "chartColor": "#34d399"
            }
        if aqi_num <= 100:
            return {
                "category": "Satisfactory",
                "description": "Minor breathing discomfort.",
                "textColor": "text-yellow-400",
                "borderColor": "border-yellow-500",
                "bgColor": "bg-yellow-500/10",
                "chartColor": "#f59e0b"
            }
        if aqi_num <= 200:
            return {
                "category": "Moderate",
                "description": "Breathing discomfort to sensitive groups.",
                "textColor": "text-orange-400",
                "borderColor": "border-orange-500",
                "bgColor": "bg-orange-500/10",
                "chartColor": "#f97316"
            }
        if aqi_num <= 300:
            return {
                "category": "Poor",
                "description": "Breathing discomfort to most people.",
                "textColor": "text-red-400",
                "borderColor": "border-red-500",
                "bgColor": "bg-red-500/10",
                "chartColor": "#ef4444"
            }
        if aqi_num <= 400:
            return {
                "category": "Very Poor",
                "description": "Respiratory illness on prolonged exposure.",
                "textColor": "text-purple-400",
                "borderColor": "border-purple-500",
                "bgColor": "bg-purple-500/10",
                "chartColor": "#a855f7"
            }
        return {
            "category": "Severe",
            "description": "Serious health effects.",
            "textColor": "text-rose-700",
            "borderColor": "border-rose-700",
            "bgColor": "bg-rose-700/10",
            "chartColor": "#be123c"
        }

# Import models and utilities after fallback setup
from models import db, User, Favorite, Tip
from .utils import (
    fetch_aqi, fetch_weather, fetch_forecast, fetch_historical_aqi,
    get_relevant_tips, get_coords_from_city
)

api_bp = Blueprint('api', __name__, url_prefix='/api')


# ---------- Helpers ----------

def _import_ml_handler():
    """Lazily import ml_handler module. Return module or None on failure."""
    try:
        ml = importlib.import_module("ml_handler")
        return ml
    except Exception as e:
        logger.warning("Could not import ml_handler lazily: %s", e, exc_info=True)
        return None


def _get_aqi_category_safe(aqi_value: Optional[float]):
    """
    Use real ml_handler.get_aqi_category if available, otherwise fallback to our get_aqi_category.
    This returns the same key names frontend expects: category, description, textColor, borderColor, bgColor, chartColor.
    """
    ml = _import_ml_handler()
    if ml and hasattr(ml, "get_aqi_category"):
        try:
            return ml.get_aqi_category(aqi_value)
        except Exception:
            logger.exception("ml_handler.get_aqi_category raised an exception; falling back.")
    # fallback to our local helper (either the real import provided fallback earlier or the ml provided one)
    try:
        return get_aqi_category(aqi_value)
    except Exception:
        logger.exception("Local get_aqi_category failed; returning safe default.")
        return {
            "category": "N/A",
            "description": "AQI data invalid.",
            "textColor": "text-slate-400",
            "borderColor": "border-slate-500",
            "bgColor": "bg-slate-500/10",
            "chartColor": "#64748b"
        }


def _calculate_subindices_simple(data: dict):
    """
    Return a simple mapping of pollutant -> numeric value (rounded) suitable for the front-end contribution chart.
    This should be replaced by calculate_all_subindices from your ML code if available.
    """
    keys = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
    out = {}
    for k in keys:
        try:
            val = data.get(k, None)
            if val is None or val == "":
                out[k] = 0.0
            else:
                out[k] = round(float(val), 2)
        except Exception:
            out[k] = 0.0
    return out


def _heuristic_predict_from_pm25(data: dict):
    """
    Cheap demo heuristic (not production): approximate AQI from PM2.5 so reviewers can see UI behavior when model not present.
    """
    try:
        pm25 = float(data.get("PM2.5") or 0)
    except Exception:
        pm25 = 0.0
    # scale and clamp
    estimated = int(min(500, max(0, pm25 * 2)))
    return estimated


# ---------- Error handler ----------
@api_bp.errorhandler(Exception)
def _handle_unexpected_error(error):
    tb = traceback.format_exc()
    logger.exception("Unhandled exception in api blueprint: %s", error)
    response = jsonify({"success": False, "error": "Internal server error", "detail": str(error), "trace": tb})
    response.status_code = 500
    return response


# ---------- Authentication routes (unchanged) ----------
@api_bp.route('/signup', methods=['POST'])
def api_signup():
    data = request.form
    full_name = data.get('full_name')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    if not all([full_name, email, password, confirm_password]):
        return jsonify({'success': False, 'error': 'All fields are required.'}), 400
    if password != confirm_password:
        return jsonify({'success': False, 'error': 'Passwords do not match.'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'Email already registered.'}), 409

    try:
        new_user = User(full_name=full_name, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        logger.info("New user registered: %s", email)
        return jsonify({'success': True, 'message': 'Registration successful! Please log in.', 'redirect': url_for('auth.login')})
    except Exception as e:
        db.session.rollback()
        logger.error("Signup Error: %s", e, exc_info=True)
        return jsonify({'success': False, 'error': 'An internal error occurred during registration.'}), 500


@api_bp.route('/login', methods=['POST'])
def api_login():
    data = request.form
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'success': False, 'error': 'Email and password are required.'}), 400

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        session['user_id'] = user.id
        session['full_name'] = user.full_name
        session['city'] = user.preferred_city
        logger.info("User '%s' logged in successfully.", user.email)
        return jsonify({'success': True, 'message': 'Login successful!', 'redirect': url_for('main.dashboard')})
    logger.warning("Failed login attempt for email: %s", email)
    return jsonify({'success': False, 'error': 'Invalid email or password.'}), 401


# ---------- dashboard / weather / forecast endpoints (unchanged structure) ----------
@api_bp.route('/aqi/<city>')
def get_aqi(city):
    coords = get_coords_from_city(city)
    if 'error' in coords:
        return jsonify({'error': coords['error']}), 404
    data = fetch_aqi(coords['lat'], coords['lon'], coords['name'])
    try:
        if 'aqi' in data:
            data['category'] = _get_aqi_category_safe(data.get('aqi'))
    except Exception:
        logger.exception("Failed adding category to aqi response for city %s", city)
    return jsonify(data)


@api_bp.route('/weather/<city>')
def get_weather(city):
    coords = get_coords_from_city(city)
    if 'error' in coords:
        return jsonify({'error': coords['error']}), 404
    data = fetch_weather(coords['lat'], coords['lon'], coords['name'])
    return jsonify(data)


@api_bp.route('/forecast/<city>')
def get_forecast(city):
    coords = get_coords_from_city(city)
    if 'error' in coords:
        logger.error("[get_forecast] Geocoding failed for %s: %s", city, coords.get('error'))
        return jsonify({'error': coords['error'], 'daily': [], 'hourly': []}), 404
    daily_summary, hourly_slice = fetch_forecast(coords['lat'], coords['lon'])
    return jsonify({'daily': daily_summary, 'hourly': hourly_slice})


@api_bp.route('/historical/<city>')
def get_historical(city):
    coords = get_coords_from_city(city)
    if 'error' in coords:
        return jsonify({'error': coords['error']}), 404
    data = fetch_historical_aqi(coords['lat'], coords['lon'])
    return jsonify(data)


@api_bp.route('/tips', methods=['POST'])
def get_dynamic_tips():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON payload.'}), 400
    city = data.get('city'); context = data.get('context')
    if not city or not context:
        return jsonify({'error': 'City and context are required'}), 400
    coords = get_coords_from_city(city)
    if 'error' in coords:
        return jsonify({'error': coords['error']}), 404
    aqi_data = fetch_aqi(coords['lat'], coords['lon'], coords['name'])
    relevant_tips = get_relevant_tips(aqi_data, context)
    tips_json = [{'title': tip.title, 'description': tip.description, 'pollutants_targeted': tip.pollutants_targeted} for tip in relevant_tips]
    return jsonify({'tips': tips_json})


# ---------- current_pollutants endpoint ----------
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


# ---------- predict_aqi endpoint (robust) ----------
@api_bp.route('/predict_aqi', methods=['POST'])
def handle_predict_aqi():
    # If your application requires login keep this check. Remove to allow anonymous predictions.
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Login required'}), 401

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON payload.'}), 400

        required_keys = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
        for key in required_keys:
            if key not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {key}'}), 400
            try:
                # allow blank strings to be parsed as errors
                if data[key] is None or data[key] == "":
                    return jsonify({'success': False, 'error': f'Missing required field: {key}'}), 400
                float(data[key])
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': f'Invalid value for {key}. Must be a number.'}), 400

        # Quick check if model is loaded (non-blocking)
        model_ready = load_model_if_needed()
        if not model_ready:
            # start background loader (non-blocking)
            try:
                background_model_loader(delay_seconds=1)
            except Exception:
                logger.exception("Could not start background model loader.")
            # If demo mode allowed (set env var ALLOW_DEMO_PREDICTIONS=true) — return cheap heuristic for UI testing
            if current_app.config.get('ALLOW_DEMO_PREDICTIONS', False):
                predicted = _heuristic_predict_from_pm25(data)
                category_info = _get_aqi_category_safe(predicted)
                subindices = _calculate_subindices_simple(data)
                return jsonify({"success": True, "predicted_aqi": predicted, "category_info": category_info, "subindices": subindices, "demo": True})

            return jsonify({'success': False, 'error': 'Model not ready. Loading on server - try again shortly.'}), 503

        # Ensure synchronous load (guarded) for immediate prediction
        if not load_model_if_needed(now=True):
            return jsonify({'success': False, 'error': 'Prediction model failed to load on server.'}), 503

        # Ensure predict function exists
        if predict_current_aqi is None:
            if current_app.config.get('ALLOW_DEMO_PREDICTIONS', False):
                predicted = _heuristic_predict_from_pm25(data)
                category_info = _get_aqi_category_safe(predicted)
                subindices = _calculate_subindices_simple(data)
                return jsonify({"success": True, "predicted_aqi": predicted, "category_info": category_info, "subindices": subindices, "demo": True})
            return jsonify({'success': False, 'error': 'Prediction functionality not available on server.'}), 500

        # Run prediction
        predicted_aqi = predict_current_aqi(data)
        if predicted_aqi is None:
            # fallback demo if allowed
            if current_app.config.get('ALLOW_DEMO_PREDICTIONS', False):
                predicted = _heuristic_predict_from_pm25(data)
                category_info = _get_aqi_category_safe(predicted)
                subindices = _calculate_subindices_simple(data)
                return jsonify({"success": True, "predicted_aqi": predicted, "category_info": category_info, "subindices": subindices, "demo": True})
            return jsonify({'success': False, 'error': 'Prediction failed. See server logs for details.'}), 500

        category_info = _get_aqi_category_safe(predicted_aqi)
        subindices = _calculate_subindices_simple(data)  # ideally use ml handler's subindex calculation
        return jsonify({"success": True, "predicted_aqi": predicted_aqi, "category_info": category_info, "subindices": subindices})
    except Exception as e:
        logger.exception("Unexpected error in /predict_aqi: %s", e)
        return jsonify({'success': False, 'error': 'Internal server error.'}), 500


# ---------- user preference & other endpoints follow (unchanged) ----------
# Add your remaining endpoints (update_city, add_favorite, remove_favorite, top_cities_aqi, map_cities_data,
# city_data, autocomplete_city, get_city_from_coords) — keep the implementation you already have.
# (For brevity this file includes predict and the crucial endpoints; include the rest exactly as you used above.)
