# routes/api.py
from flask import Blueprint, request, jsonify, session, url_for, current_app
import logging
import requests
import importlib
import traceback
from typing import Any, Dict, List, Optional

from models import db, User, Favorite, Tip
from .utils import (
    fetch_aqi, fetch_weather, fetch_forecast, fetch_historical_aqi,
    get_relevant_tips, get_coords_from_city
)

api_bp = Blueprint('api', __name__, url_prefix='/api')
logger = logging.getLogger(__name__)


# ---------- Helper: lazy import ml_handler ----------
def _import_ml_handler():
    """
    Lazily import ml_handler module. Return module or None on failure.
    """
    try:
        ml = importlib.import_module("ml_handler")
        return ml
    except Exception as e:
        logger.warning("Could not import ml_handler: %s", e, exc_info=True)
        return None


# ---------- Fallback get_aqi_category ----------
def _fallback_get_aqi_category(aqi_value: Optional[float]) -> Dict[str, str]:
    """
    Local fallback for get_aqi_category. This is used if ml_handler is unavailable.
    """
    try:
        aqi_val = float(aqi_value) if aqi_value is not None else None
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


def _get_aqi_category_safe(aqi_value: Optional[float]) -> Dict[str, str]:
    """
    Try calling ml_handler.get_aqi_category if available; otherwise use fallback.
    """
    ml = _import_ml_handler()
    if ml and hasattr(ml, "get_aqi_category"):
        try:
            return ml.get_aqi_category(aqi_value)
        except Exception:
            logger.exception("ml_handler.get_aqi_category raised an exception; falling back.")
            return _fallback_get_aqi_category(aqi_value)
    return _fallback_get_aqi_category(aqi_value)


# ---------- Blueprint-level JSON error handler ----------
@api_bp.errorhandler(Exception)
def _handle_unexpected_error(error):
    """
    Ensure that unhandled exceptions inside this blueprint return valid JSON
    with a server-side traceback in the logs (and included in the response to help debugging).
    """
    tb = traceback.format_exc()
    logger.exception("Unhandled exception in api blueprint: %s", error)
    # In production you might want to hide 'trace' content from clients; for debugging include it.
    response = jsonify({"success": False, "error": "Internal server error", "detail": str(error), "trace": tb})
    response.status_code = 500
    return response


# --- AUTHENTICATION ROUTES ---
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


# --- DASHBOARD DATA ROUTES ---
@api_bp.route('/aqi/<city>')
def get_aqi(city):
    coords = get_coords_from_city(city)
    if 'error' in coords:
        return jsonify({'error': coords['error']}), 404
    data = fetch_aqi(coords['lat'], coords['lon'], coords['name'])
    # Ensure category field is available via safe call
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
    tips_json = [{
        'title': tip.title, 'description': tip.description, 'pollutants_targeted': tip.pollutants_targeted
    } for tip in relevant_tips]
    return jsonify({'tips': tips_json})


# --- AQI PREDICTOR ENDPOINTS ---
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


@api_bp.route('/predict_aqi', methods=['POST'])
def handle_predict_aqi():
    """
    Robust predict endpoint:
    - Validates JSON and numeric inputs
    - Lazily imports ml_handler and calls predict_current_aqi
    - Always returns JSON (success or error) with useful info
    """
    # Example auth check: remove if not required
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Login required'}), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'Invalid JSON payload.'}), 400

    required_keys = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
    missing_keys: List[str] = []
    invalid_keys: List[str] = []

    for key in required_keys:
        if key not in data or data[key] is None:
            missing_keys.append(key)
            continue
        try:
            # Accept numeric strings and numbers
            float(data[key])
        except (ValueError, TypeError):
            invalid_keys.append(key)

    if missing_keys:
        logger.error("Predict request missing fields: %s", missing_keys)
        return jsonify({'success': False, 'error': 'Missing required fields', 'missing': missing_keys}), 400
    if invalid_keys:
        logger.error("Predict request invalid numeric values for: %s", invalid_keys)
        return jsonify({'success': False, 'error': 'Invalid numeric values', 'invalid': invalid_keys}), 400

    # Lazy import ml_handler and call predictor
    ml = _import_ml_handler()
    if not ml:
        logger.error("ml_handler unavailable when handling /predict_aqi")
        return jsonify({'success': False, 'error': 'Prediction service currently unavailable (ml handler missing).'}), 500

    try:
        predicted_aqi = ml.predict_current_aqi(data)
    except Exception as e:
        logger.exception("Exception while calling ml_handler.predict_current_aqi: %s", e)
        return jsonify({'success': False, 'error': 'Prediction failed', 'detail': str(e)}), 500

    if predicted_aqi is None:
        # Try to provide more info from ml_handler
        model_loaded = getattr(ml, "AQI_PREDICTOR_MODEL", None)
        if not model_loaded:
            logger.error("Predict failed because model is not loaded (AQI_PREDICTOR_MODEL is None).")
            return jsonify({'success': False, 'error': 'Prediction model not loaded.'}), 500
        logger.error("Predict returned None for data: %s", data)
        return jsonify({'success': False, 'error': 'Prediction failed. Check server logs.'}), 500

    # compute category and subindices (both lazily via ml_handler)
    try:
        category_info = ml.get_aqi_category(predicted_aqi) if hasattr(ml, "get_aqi_category") else _fallback_get_aqi_category(predicted_aqi)
    except Exception:
        logger.exception("ml_handler.get_aqi_category failed; using fallback.")
        category_info = _fallback_get_aqi_category(predicted_aqi)

    try:
        subindices = ml.calculate_all_subindices(data) if hasattr(ml, "calculate_all_subindices") else {}
    except Exception:
        logger.exception("ml_handler.calculate_all_subindices failed; returning empty subindices.")
        subindices = {}

    logger.info("Prediction OK. AQI: %s. User: %s", predicted_aqi, session.get('user_id'))
    return jsonify({"success": True, "predicted_aqi": predicted_aqi, "category_info": category_info, "subindices": subindices}), 200


# --- USER PREFERENCE ROUTES ---
@api_bp.route('/update_city', methods=['POST'])
def update_city():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    payload = request.get_json(silent=True) or {}
    city = payload.get('city')
    if not city or len(city.strip()) == 0:
        return jsonify({'success': False, 'error': 'City name cannot be empty.'}), 400
    if len(city) > 50:
        return jsonify({'success': False, 'error': 'City name too long (max 50 chars).'}), 400
    user = db.session.get(User, session['user_id'])
    if user:
        user.preferred_city = city.strip()
        try:
            db.session.commit()
            session['city'] = user.preferred_city
            logger.info("User %s updated preferred city to %s", user.id, user.preferred_city)
            return jsonify({'success': True, 'city': user.preferred_city})
        except Exception as e:
            db.session.rollback()
            logger.error("DB error updating city for user %s: %s", user.id, e, exc_info=True)
            return jsonify({'success': False, 'error': 'Database error.'}), 500
    return jsonify({'success': False, 'error': 'User not found'}), 404


@api_bp.route('/add_favorite', methods=['POST'])
def add_favorite():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Login required'}), 401
    payload = request.get_json(silent=True) or {}
    city = payload.get('city')
    user_id = session['user_id']
    if not city or len(city.strip()) == 0:
        return jsonify({'success': False, 'error': 'City name cannot be empty.'}), 400
    if len(city) > 50:
        return jsonify({'success': False, 'error': 'City name too long (max 50 chars).'}), 400
    city_cleaned = city.strip()
    if Favorite.query.filter_by(user_id=user_id, city=city_cleaned).first():
        return jsonify({'success': True, 'message': f'{city_cleaned} is already in favorites.'})
    fav = Favorite(user_id=user_id, city=city_cleaned)
    try:
        db.session.add(fav)
        db.session.commit()
        logger.info("User %s added favorite: %s", user_id, city_cleaned)
        return jsonify({'success': True, 'message': f'{city_cleaned} added to favorites.'})
    except Exception as e:
        db.session.rollback()
        logger.error("DB error adding favorite for user %s: %s", user_id, e, exc_info=True)
        return jsonify({'success': False, 'error': 'Database error.'}), 500


@api_bp.route('/remove_favorite', methods=['POST'])
def remove_favorite():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Login required'}), 401
    payload = request.get_json(silent=True) or {}
    city = payload.get('city')
    user_id = session['user_id']
    if not city:
        return jsonify({'success': False, 'error': 'City name required.'}), 400
    fav = Favorite.query.filter_by(user_id=user_id, city=city).first()
    if fav:
        try:
            db.session.delete(fav)
            db.session.commit()
            logger.info("User %s removed favorite: %s", user_id, city)
            return jsonify({'success': True, 'message': f'{city} removed from favorites.'})
        except Exception as e:
            db.session.rollback()
            logger.error("DB error removing favorite for user %s: %s", user_id, e, exc_info=True)
            return jsonify({'success': False, 'error': 'Database error.'}), 500
    else:
        return jsonify({'success': True, 'message': f'{city} was not in favorites.'})


# --- TOP CITIES AQI ENDPOINT ---
@api_bp.route('/top_cities_aqi')
def top_cities_aqi():
    indian_cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Patna", "Indore", "Thane"]
    world_cities = ["Beijing", "New York", "London", "Tokyo", "Paris", "Los Angeles", "Mexico City", "Sao Paulo", "Cairo", "Moscow", "Jakarta", "Seoul", "Sydney", "Berlin", "Rome"]
    top_cities_data = {'india': [], 'world': []}

    for city_name in indian_cities:
        coords = get_coords_from_city(city_name)
        if 'error' in coords:
            logger.warning("Skipping Indian city %s (geocoding error)", city_name)
            continue
        aqi_data = fetch_aqi(coords['lat'], coords['lon'], coords['name'])
        if 'error' not in aqi_data:
            cat = _get_aqi_category_safe(aqi_data.get('aqi', -1))
            top_cities_data['india'].append({'city': aqi_data.get('city'), 'aqi': aqi_data.get('aqi'), 'category': cat})
        else:
            logger.warning("Skipping Indian city %s (AQI fetch error)", city_name)

    for city_name in world_cities:
        coords = get_coords_from_city(city_name)
        if 'error' in coords:
            logger.warning("Skipping World city %s (geocoding error)", city_name)
            continue
        aqi_data = fetch_aqi(coords['lat'], coords['lon'], coords['name'])
        if 'error' not in aqi_data:
            cat = _get_aqi_category_safe(aqi_data.get('aqi', -1))
            top_cities_data['world'].append({'city': aqi_data.get('city'), 'aqi': aqi_data.get('aqi'), 'category': cat})
        else:
            logger.warning("Skipping World city %s (AQI fetch error)", city_name)

    # Sort lists by AQI (descending - worst first). Use int conversion with fallback.
    def _aqi_sort_key(x):
        try:
            return int(x.get('aqi', -1))
        except Exception:
            return -1

    top_cities_data['india'].sort(key=_aqi_sort_key, reverse=True)
    top_cities_data['world'].sort(key=_aqi_sort_key, reverse=True)

    logger.info("Successfully compiled top cities AQI data.")
    return jsonify(top_cities_data)


# --- MAP & CITY ROUTES ---
@api_bp.route('/map_cities_data')
def map_cities_data():
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'New York', 'London', 'Tokyo', 'Beijing', 'Sydney']
    data = []
    for city in cities:
        coords = get_coords_from_city(city)
        if 'error' in coords:
            logger.warning("Skipping map city %s (geocoding error): %s", city, coords.get('error'))
            continue
        aqi_data = fetch_aqi(coords['lat'], coords['lon'], coords['name'])
        weather_data = fetch_weather(coords['lat'], coords['lon'], coords['name'])
        if 'error' not in aqi_data:
            aqi_data['weather'] = weather_data
            # add safe category
            aqi_data['category'] = _get_aqi_category_safe(aqi_data.get('aqi'))
            data.append(aqi_data)
        else:
            logger.warning("Skipping map city %s (AQI error): %s", city, aqi_data.get('error'))
    return jsonify(data)


@api_bp.route('/city_data/<city_from_url>')
def get_city_data(city_from_url):
    logger.info("--- [get_city_data] Received request for city: '%s' ---", city_from_url)
    coords = get_coords_from_city(city_from_url)
    if 'error' in coords:
        logger.error("[get_city_data] Geocoding failed for '%s': %s", city_from_url, coords['error'])
        return jsonify({'error': coords['error']}), 404
    official_city_name = coords.get('name', city_from_url)
    aqi_data = fetch_aqi(coords['lat'], coords['lon'], official_city_name)
    weather_data = fetch_weather(coords['lat'], coords['lon'], official_city_name)
    logger.info("AQI data fetched: %s", aqi_data)
    logger.info("Weather data fetched: %s", weather_data)
    if 'error' in aqi_data:
        logger.error("[get_city_data] AQI fetch failed for '%s': %s", official_city_name, aqi_data.get('error'))
        return jsonify({'error': f'Could not find AQI data for {official_city_name}'}), 404
    aqi_data['weather'] = weather_data.copy() if isinstance(weather_data, dict) else weather_data
    aqi_data['category'] = _get_aqi_category_safe(aqi_data.get('aqi'))
    logger.info("Final combined data for '%s': %s", official_city_name, aqi_data)
    return jsonify(aqi_data)


# --- AUTOCOMPLETE ENDPOINT ---
@api_bp.route('/autocomplete_city')
def autocomplete_city():
    query = request.args.get('query', '').strip()
    limit = request.args.get('limit', 5, type=int)

    if not query or len(query) < 2:
        return jsonify([])

    api_key = current_app.config.get('OPENWEATHER_API_KEY')
    url = current_app.config.get('GEOCODING_API_URL', "http://api.openweathermap.org/geo/1.0/direct")

    if not api_key:
        logger.error("Autocomplete failed: OPENWEATHER_API_KEY missing.")
        return jsonify({"error": "Server configuration error"}), 500

    params = {'q': query, 'limit': limit, 'appid': api_key}
    suggestions = []
    unique_names = set()

    try:
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        data = response.json()

        for item in data:
            parts = [item.get('name')]
            if item.get('state'):
                parts.append(item.get('state'))
            if item.get('country'):
                parts.append(item.get('country'))
            full_name = ", ".join(filter(None, parts))
            city_name_lower = item.get('name', '').lower()
            if city_name_lower and city_name_lower not in unique_names:
                suggestions.append(full_name)
                unique_names.add(city_name_lower)
    except requests.exceptions.Timeout:
        logging.warning("Autocomplete request timed out for query: %s", query)
        return jsonify([])
    except requests.exceptions.RequestException as e:
        logging.error("Autocomplete API request error for query '%s': %s", query, e)
        return jsonify([])
    except Exception as e:
        logging.exception("Unexpected error in autocomplete_city for query '%s': %s", query, e)
        return jsonify({"error": "Autocomplete service error"}), 500

    return jsonify(suggestions)


# --- REVERSE GEOCODING ---
@api_bp.route('/get_city_from_coords')
def get_city_from_coords():
    lat = request.args.get('lat'); lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude are required."}), 400

    api_key = current_app.config.get('OPENWEATHER_API_KEY')
    url = "http://api.openweathermap.org/geo/1.0/reverse"

    if not api_key:
        logger.error("Reverse geocoding failed: OPENWEATHER_API_KEY missing.")
        return jsonify({"error": "Server configuration error"}), 500

    params = {'lat': lat, 'lon': lon, 'limit': 1, 'appid': api_key}

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            city_info = data[0]
            city_name = city_info.get('name')
            if not city_name:
                parts = [city_info.get('state'), city_info.get('country')]
                city_name = ", ".join(filter(None, parts))
            if not city_name:
                logger.warning("Reverse geocoding no name: %s", data)
                return jsonify({"error": "Could not determine location name."}), 404
            logger.info("Reverse geocoded (%s,%s) to: %s", lat, lon, city_name)
            return jsonify({"city": city_name})
        else:
            logger.warning("Reverse geocoding no data for (%s,%s).", lat, lon)
            return jsonify({"error": "Location name not found."}), 404
    except requests.exceptions.Timeout:
        logger.warning("Reverse geocoding timeout for (%s,%s)", lat, lon)
        return jsonify({"error": "Reverse geocoding service timed out."}), 504
    except requests.exceptions.RequestException as e:
        logger.error("Reverse geocoding error for (%s,%s): %s", lat, lon, e, exc_info=True)
        return jsonify({"error": "Could not connect to location service."}), 503
    except Exception as e:
        logger.exception("Unexpected error in get_city_from_coords (%s,%s): %s", lat, lon, e)
        return jsonify({"error": "Unexpected error during reverse geocoding."}), 500
