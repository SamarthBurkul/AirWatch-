import requests
import logging
from flask import Blueprint, request, jsonify, current_app, session
from extensions import db
from models import Tip, User, Favorite
from ml_handler import predict_current_aqi, get_aqi_category, calculate_all_subindices
# Import utility functions from your existing utils.py
from .utils import (
    get_coords_from_city,
    fetch_aqi,
    fetch_weather,
    fetch_forecast,
    fetch_historical_aqi,
    get_relevant_tips
)

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)

# -----------------------------
#  AUTHENTICATION ENDPOINTS
# -----------------------------
@api_bp.route("/signup", methods=["POST"])
def api_signup():
    """Handle user registration."""
    try:
        data = request.get_json()
        
        # Validation
        required_fields = ['full_name', 'email', 'password', 'confirm_password']
        if not all(field in data for field in required_fields):
            return jsonify({"success": False, "error": "All fields are required"}), 400
        
        if data['password'] != data['confirm_password']:
            return jsonify({"success": False, "error": "Passwords do not match"}), 400
        
        # Check if user exists
        existing_user = User.query.filter_by(email=data['email']).first()
        if existing_user:
            return jsonify({"success": False, "error": "Email already registered"}), 400
        
        # Create new user
        new_user = User(
            full_name=data['full_name'],
            email=data['email'],
            preferred_city=data.get('preferred_city', 'Delhi')
        )
        new_user.set_password(data['password'])
        
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Registration successful",
            "redirect": "/auth/login"
        }), 201
        
    except Exception as e:
        logger.exception("Signup failed")
        db.session.rollback()
        return jsonify({"success": False, "error": "Registration failed"}), 500


@api_bp.route("/login", methods=["POST"])
def api_login():
    """Handle user login."""
    try:
        data = request.get_json()
        
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"success": False, "error": "Email and password required"}), 400
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            return jsonify({"success": False, "error": "Invalid credentials"}), 401
        
        # Set session
        session['user_id'] = user.id
        session['full_name'] = user.full_name
        session['city'] = user.preferred_city or 'Delhi'
        
        return jsonify({
            "success": True,
            "message": "Login successful",
            "redirect": "/dashboard"
        }), 200
        
    except Exception as e:
        logger.exception("Login failed")
        return jsonify({"success": False, "error": "Login failed"}), 500


# -----------------------------
#  GEOLOCATION → CITY NAME
# -----------------------------
@api_bp.route("/get_city_from_coords", strict_slashes=False)
def get_city_from_coords():
    """Reverse geocode coordinates to city name."""
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "lat and lon required"}), 400

    api_key = current_app.config.get("OPENWEATHER_API_KEY")
    if not api_key:
        logger.error("OPENWEATHER_API_KEY not configured.")
        return jsonify({"error": "Server configuration error"}), 500

    try:
        url = "https://api.openweathermap.org/geo/1.0/reverse"
        params = {"lat": lat, "lon": lon, "limit": 1, "appid": api_key}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, list) and len(data) > 0:
            place = data[0]
            city = (
                place.get("name")
                or (place.get("local_names") or {}).get("en")
                or place.get("state")
                or place.get("country")
            )
            if city:
                return jsonify({"city": city}), 200

        return jsonify({"error": "API returned no city name."}), 200

    except Exception as e:
        logger.exception("Geocoding failed")
        return jsonify({"error": "Geocoding service error"}), 502


# -----------------------------
#  CURRENT POLLUTANTS (FOR PREDICTOR)
# -----------------------------
@api_bp.route("/current_pollutants", strict_slashes=False)
def current_pollutants():
    """Fetch current pollutant data for a location (used by predictor page)."""
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "lat and lon required"}), 400

    api_key = current_app.config.get("OPENWEATHER_API_KEY")
    if not api_key:
        return jsonify({"error": "API KEY missing"}), 500

    try:
        url = "https://api.openweathermap.org/data/2.5/air_pollution"
        params = {"lat": lat, "lon": lon, "appid": api_key}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()

        components = data["list"][0]["components"]

        return jsonify({
            "PM2.5": components.get("pm2_5"),
            "PM10": components.get("pm10"),
            "NO": components.get("no"),
            "NO2": components.get("no2"),
            "NOx": components.get("no2"),
            "NH3": components.get("nh3"),
            "CO": components.get("co"),
            "SO2": components.get("so2"),
            "O3": components.get("o3"),
            "Benzene": 0,
            "Toluene": 0,
            "Xylene": 0
        }), 200

    except Exception:
        logger.exception("Pollutants fetch failed")
        return jsonify({"error": "Failed to fetch pollutants"}), 500


# -----------------------------
#  DASHBOARD DATA ENDPOINTS
# -----------------------------
@api_bp.route("/aqi/<city>", strict_slashes=False)
def get_city_aqi(city):
    """
    Fetch AQI for a city using your existing utils.py functions.
    Returns data in the format expected by dashboard.js
    """
    try:
        # Step 1: Get coordinates from city name
        coords = get_coords_from_city(city)
        if 'error' in coords:
            return jsonify(coords), 404
        
        # Step 2: Fetch AQI data using your existing function
        aqi_data = fetch_aqi(coords['lat'], coords['lon'], coords['name'])
        
        if 'error' in aqi_data:
            return jsonify(aqi_data), 500
        
        return jsonify(aqi_data), 200
        
    except Exception as e:
        logger.exception(f"Error fetching AQI for {city}")
        return jsonify({"error": "Failed to fetch AQI data"}), 500


@api_bp.route("/weather/<city>", strict_slashes=False)
def get_city_weather(city):
    """
    Fetch weather for a city using your existing utils.py functions.
    Returns data in the format expected by dashboard.js
    """
    try:
        # Step 1: Get coordinates from city name
        coords = get_coords_from_city(city)
        if 'error' in coords:
            return jsonify(coords), 404
        
        # Step 2: Fetch weather data using your existing function
        weather_data = fetch_weather(coords['lat'], coords['lon'], coords['name'])
        
        if 'error' in weather_data:
            return jsonify(weather_data), 500
        
        return jsonify(weather_data), 200
        
    except Exception as e:
        logger.exception(f"Error fetching weather for {city}")
        return jsonify({"error": "Failed to fetch weather data"}), 500


@api_bp.route("/forecast/<city>", strict_slashes=False)
def get_city_forecast(city):
    """
    Fetch 5-day forecast for a city.
    Returns both daily summary and hourly slice as expected by dashboard.js
    """
    try:
        # Step 1: Get coordinates from city name
        coords = get_coords_from_city(city)
        if 'error' in coords:
            return jsonify(coords), 404
        
        # Step 2: Fetch forecast data (returns both daily and hourly)
        daily_forecast, hourly_forecast = fetch_forecast(coords['lat'], coords['lon'])
        
        return jsonify({
            "daily": daily_forecast,
            "hourly": hourly_forecast
        }), 200
        
    except Exception as e:
        logger.exception(f"Error fetching forecast for {city}")
        return jsonify({"error": "Failed to fetch forecast data"}), 500


@api_bp.route("/historical/<city>", strict_slashes=False)
def get_historical_aqi(city):
    """
    Fetch historical AQI data (last 24 hours).
    Uses your existing fetch_historical_aqi function which includes simulation fallback.
    """
    try:
        # Step 1: Get coordinates from city name
        coords = get_coords_from_city(city)
        if 'error' in coords:
            return jsonify(coords), 404
        
        # Step 2: Fetch historical data
        historical_data = fetch_historical_aqi(coords['lat'], coords['lon'])
        
        # Return as array (format expected by dashboard.js)
        return jsonify(historical_data), 200
        
    except Exception as e:
        logger.exception(f"Error fetching historical data for {city}")
        return jsonify({"error": "Failed to fetch historical data"}), 500


# -----------------------------
#  TOP CITIES AQI
# -----------------------------
@api_bp.route("/top_cities_aqi", strict_slashes=False)
def get_top_cities_aqi():
    """
    Fetch AQI for major Indian and world cities.
    Returns two separate lists as expected by dashboard.js
    """
    api_key = current_app.config.get("OPENWEATHER_API_KEY")
    if not api_key:
        return jsonify({"error": "API key not configured"}), 500

    # Define city lists with coordinates
    indian_cities = [
        {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
        {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
        {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
        {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    ]
    
    world_cities = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
        {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
        {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
    ]

    def fetch_cities_data(cities_list):
        """Helper function to fetch AQI for a list of cities."""
        results = []
        for city_info in cities_list:
            try:
                # Use your existing fetch_aqi function
                aqi_data = fetch_aqi(city_info["lat"], city_info["lon"], city_info["name"])
                
                if 'error' not in aqi_data:
                    # Get category info for UI styling
                    category = get_aqi_category(aqi_data.get('aqi', 0))
                    
                    results.append({
                        "city": city_info["name"],
                        "aqi": aqi_data.get('aqi', 'N/A'),
                        "category": category
                    })
                else:
                    # Add placeholder for failed cities
                    results.append({
                        "city": city_info["name"],
                        "aqi": 'N/A',
                        "category": {
                            "category": "Unavailable",
                            "textColor": "text-slate-400",
                            "borderColor": "border-slate-500"
                        }
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch AQI for {city_info['name']}: {e}")
                results.append({
                    "city": city_info["name"],
                    "aqi": 'N/A',
                    "category": {
                        "category": "Unavailable",
                        "textColor": "text-slate-400",
                        "borderColor": "border-slate-500"
                    }
                })
        
        # Sort by AQI (highest first), put N/A at end
        results.sort(key=lambda x: (x['aqi'] == 'N/A', -x['aqi'] if x['aqi'] != 'N/A' else 0))
        return results

    try:
        india_results = fetch_cities_data(indian_cities)
        world_results = fetch_cities_data(world_cities)
        
        return jsonify({
            "india": india_results,
            "world": world_results
        }), 200
        
    except Exception as e:
        logger.exception("Top cities AQI fetch failed")
        return jsonify({"error": "Failed to fetch top cities data"}), 500


# -----------------------------
#  TIPS ENDPOINT
# -----------------------------
@api_bp.route("/tips", methods=["POST"])
def get_tips_for_context():
    """
    Fetch context-aware tips for the current city and AQI.
    Uses your existing get_relevant_tips function.
    """
    try:
        data = request.get_json()
        city = data.get('city', 'Delhi')
        context = data.get('context', 'home')
        
        # Fetch current AQI for the city
        coords = get_coords_from_city(city)
        if 'error' in coords:
            # Return generic tips if city fetch fails
            generic_tips = Tip.query.filter_by(category='home', difficulty='easy').limit(3).all()
            return jsonify({
                "tips": [{
                    "title": t.title,
                    "description": t.description,
                    "pollutants_targeted": t.pollutants_targeted,
                    "category": t.category
                } for t in generic_tips]
            }), 200
        
        aqi_data = fetch_aqi(coords['lat'], coords['lon'], coords['name'])
        
        # Get relevant tips using your existing function
        tips = get_relevant_tips(aqi_data, context)
        
        return jsonify({
            "tips": [{
                "title": t.title,
                "description": t.description,
                "pollutants_targeted": t.pollutants_targeted,
                "category": t.category,
                "impact": t.impact,
                "difficulty": t.difficulty
            } for t in tips]
        }), 200
        
    except Exception as e:
        logger.exception("Tips fetch failed")
        return jsonify({"error": "Could not load tips"}), 500


@api_bp.route("/tips", methods=["GET"])
def get_all_tips():
    """Get all tips (for tips page)."""
    try:
        tips = Tip.query.order_by(Tip.id.desc()).all()
        return jsonify([{
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "category": t.category,
            "difficulty": t.difficulty,
            "impact": t.impact,
            "pollutants_targeted": t.pollutants_targeted,
            "related_diseases": t.related_diseases
        } for t in tips]), 200
    except Exception:
        logger.exception("Tips fetch failed")
        return jsonify({"error": "Could not load tips"}), 500


# -----------------------------
#  USER MANAGEMENT
# -----------------------------
@api_bp.route("/update_city", methods=["POST"])
def update_city():
    """Update user's preferred city."""
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.get_json()
        city = data.get('city')
        
        if not city:
            return jsonify({"success": False, "error": "City name required"}), 400
        
        user = db.session.get(User, session['user_id'])
        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404
        
        user.preferred_city = city
        db.session.commit()
        session['city'] = city
        
        return jsonify({"success": True, "city": city}), 200
        
    except Exception as e:
        logger.exception("Update city failed")
        db.session.rollback()
        return jsonify({"success": False, "error": "Failed to update city"}), 500


@api_bp.route("/add_favorite", methods=["POST"])
def add_favorite():
    """Add a city to user's favorites."""
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.get_json()
        city = data.get('city')
        
        if not city:
            return jsonify({"success": False, "error": "City name required"}), 400
        
        # Check if already favorited
        existing = Favorite.query.filter_by(
            user_id=session['user_id'],
            city=city
        ).first()
        
        if existing:
            return jsonify({"success": False, "error": "City already in favorites"}), 400
        
        favorite = Favorite(user_id=session['user_id'], city=city)
        db.session.add(favorite)
        db.session.commit()
        
        return jsonify({"success": True, "message": "City added to favorites"}), 201
        
    except Exception as e:
        logger.exception("Add favorite failed")
        db.session.rollback()
        return jsonify({"success": False, "error": "Failed to add favorite"}), 500


@api_bp.route("/remove_favorite", methods=["POST"])
def remove_favorite():
    """Remove a city from user's favorites."""
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        data = request.get_json()
        city = data.get('city')
        
        if not city:
            return jsonify({"success": False, "error": "City name required"}), 400
        
        favorite = Favorite.query.filter_by(
            user_id=session['user_id'],
            city=city
        ).first()
        
        if not favorite:
            return jsonify({"success": False, "error": "City not in favorites"}), 404
        
        db.session.delete(favorite)
        db.session.commit()
        
        return jsonify({"success": True, "message": "City removed from favorites"}), 200
        
    except Exception as e:
        logger.exception("Remove favorite failed")
        db.session.rollback()
        return jsonify({"success": False, "error": "Failed to remove favorite"}), 500


# -----------------------------
#  ML PREDICTION ENDPOINT
# -----------------------------
@api_bp.route("/predict_aqi", methods=["POST"])
def predict_aqi():
    """ML-based AQI prediction (for predictor page)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No input data"}), 400

        aqi = predict_current_aqi(data)
        if aqi is None:
            return jsonify({"success": False, "error": "Prediction failed"}), 500

        category = get_aqi_category(aqi)
        subindices = calculate_all_subindices(data)

        return jsonify({
            "success": True,
            "predicted_aqi": aqi,
            "category_info": category,
            "subindices": subindices
        }), 200

    except Exception:
        logger.exception("Prediction error")
        return jsonify({"success": False, "error": "Prediction failed"}), 500


# -----------------------------
#  AUTOCOMPLETE ENDPOINT
# -----------------------------
# -----------------------------
#  AUTOCOMPLETE ENDPOINT
# -----------------------------
@api_bp.route("/autocomplete_city", strict_slashes=False)
def autocomplete_city():
    """Provide city name suggestions for autocomplete."""
    query = request.args.get('query', '').strip()
    
    if not query or len(query) < 2:
        return jsonify([]), 200
    
    api_key = current_app.config.get("OPENWEATHER_API_KEY")
    if not api_key:
        return jsonify({"error": "API key not configured"}), 500
    
    try:
        url = "https://api.openweathermap.org/geo/1.0/direct"
        params = {"q": query, "limit": 5, "appid": api_key}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        suggestions = []
        
        for place in data:
            name = place.get('name')
            country = place.get('country')
            state = place.get('state')
            
            if name:
                display_name = f"{name}, {country}" if country else name
                if state and state != name:
                    display_name = f"{name}, {state}, {country}"
                suggestions.append(display_name)
        
        return jsonify(suggestions), 200
        
    except Exception as e:
        logger.exception("Autocomplete failed")
        return jsonify([]), 200


# -----------------------------
#  MAP ENDPOINTS (MUST BE OUTSIDE autocomplete_city!)
# -----------------------------
@api_bp.route('/map_cities_data', methods=['GET'])
def map_cities_data():
    """Return initial set of cities to display on map load."""
    initial_cities = [
        'Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai',
        'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'
    ]
    
    cities_data = []
    
    for city_name in initial_cities:
        try:
            coords = get_coords_from_city(city_name)
            if 'error' in coords:
                logger.warning(f"Could not fetch coords for {city_name}")
                continue
            
            aqi_data = fetch_aqi(coords['lat'], coords['lon'], coords['name'])
            weather_data = fetch_weather(coords['lat'], coords['lon'], coords['name'])
            
            city_info = {
                'city': coords['name'],
                'country': coords.get('country', 'IN'),
                'geo': [coords['lat'], coords['lon']],
                'aqi': aqi_data.get('aqi', 'N/A') if 'error' not in aqi_data else 'N/A',
                'weather': weather_data if 'error' not in weather_data else None
            }
            
            cities_data.append(city_info)
            
        except Exception as e:
            logger.exception(f"Error fetching data for {city_name}")
            continue
    
    return jsonify(cities_data)


@api_bp.route('/city_data/<city_name>', methods=['GET'])
def city_data(city_name):
    """Get comprehensive data (AQI + Weather) for a specific city."""
    if not city_name:
        return jsonify({'error': 'City name is required'}), 400
    
    try:
        coords = get_coords_from_city(city_name)
        if 'error' in coords:
            return jsonify({'error': coords['error']}), 404
        
        aqi_data = fetch_aqi(coords['lat'], coords['lon'], coords['name'])
        weather_data = fetch_weather(coords['lat'], coords['lon'], coords['name'])
        
        response_data = {
            'city': coords['name'],
            'country': coords.get('country', ''),
            'geo': [coords['lat'], coords['lon']],
            'aqi': aqi_data.get('aqi', 'N/A') if 'error' not in aqi_data else 'N/A',
            'weather': weather_data if 'error' not in weather_data else {'error': weather_data.get('error')}
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.exception(f"Error fetching city data for {city_name}")
        return jsonify({'error': f'Failed to fetch data for {city_name}'}), 500