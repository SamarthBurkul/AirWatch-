# routes/utils.py

# ... (imports and other functions remain the same) ...
import requests
from flask import current_app
from datetime import datetime, timedelta, timezone
import math
from models import db, Tip
from extensions import cache # Make sure cache is imported
from sqlalchemy import or_
import time
from collections import defaultdict
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

# --- CPCB AQI Calculation (No Changes Needed) ---
# ... (calculate_indian_aqi function remains the same) ...
def calculate_indian_aqi(components):
    # (Existing Correct Code)
    pm25 = components.get("pm2_5", 0)
    pm10 = components.get("pm10", 0)
    no2 = components.get("no2", 0)
    so2 = components.get("so2", 0)
    o3 = components.get("o3", 0)
    co_val = components.get("co", 0)
    co = float(co_val) / 1000.0 if co_val is not None else 0.0
    def get_sub_index(c, breakpoints):
        try: c_float = float(c)
        except (ValueError, TypeError): return None
        if not breakpoints or not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in breakpoints):
             logging.error(f"Invalid breakpoints format: {breakpoints}"); return None
        if c_float < breakpoints[0][0]: return breakpoints[0][1]
        if c_float > breakpoints[-1][0]: return breakpoints[-1][1]
        for i in range(len(breakpoints)-1):
            bp_low = breakpoints[i]; bp_high = breakpoints[i+1]
            try:
                c_low, i_low = float(bp_low[0]), float(bp_low[1])
                c_high, i_high = float(bp_high[0]), float(bp_high[1])
            except (ValueError, TypeError):
                 logging.error(f"Invalid numeric value in breakpoints: {bp_low} or {bp_high}"); continue
            if c_low <= c_float <= c_high:
                if (c_high - c_low) == 0: return i_low
                return ((i_high - i_low)/(c_high - c_low)) * (c_float - c_low) + i_low
        logging.warning(f"Could not find range for {c_float} in {breakpoints}"); return breakpoints[-1][1]
    PM25_bp = [(0,0),(30,50),(60,100),(90,200),(120,300),(250,400),(500,500)]
    PM10_bp = [(0,0),(50,50),(100,100),(250,200),(350,300),(430,400),(600,500)]
    NO2_bp  = [(0,0),(40,50),(80,100),(180,200),(280,300),(400,400),(600,500)]
    SO2_bp  = [(0,0),(40,50),(80,100),(380,200),(800,300),(1600,400),(2000,500)]
    O3_bp   = [(0,0),(50,50),(100,100),(168,200),(208,300),(748,400),(1000,500)]
    CO_bp   = [(0,0),(1,50),(2,100),(10,200),(17,300),(34,400),(50,500)]
    indices = {"PM2.5": get_sub_index(pm25, PM25_bp), "PM10": get_sub_index(pm10, PM10_bp), "NO2": get_sub_index(no2, NO2_bp), "SO2": get_sub_index(so2, SO2_bp), "O3": get_sub_index(o3, O3_bp), "CO": get_sub_index(co, CO_bp)}
    valid_indices = {k: v for k, v in indices.items() if v is not None}
    if not valid_indices: logging.warning(f"No valid sub-indices for: {components}"); return 'N/A', 'N/A'
    max_pollutant = max(valid_indices, key=valid_indices.get); aqi = valid_indices[max_pollutant]
    return round(aqi) if aqi is not None else 'N/A', max_pollutant

# --- API Fetching Functions ---

# Keep prefix for coords as city name should be unique enough and used directly
# --- @cache.cached(timeout=3600, key_prefix='coords_%s')
def get_coords_from_city(city_name):
    """Get coordinates from city name using OpenWeatherMap Geocoding API."""
    logging.debug(f"Fetching coordinates for city: {city_name}")
    api_key = current_app.config.get('OPENWEATHER_API_KEY')
    url = current_app.config.get('GEOCODING_API_URL', "https://api.openweathermap.org/geo/1.0/direct")
    
    if not api_key:
        logging.error("OPENWEATHER_API_KEY not configured.")
        return {'error': 'Server configuration error: API key missing.'}
    
    params = {'q': city_name, 'limit': 1, 'appid': api_key}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            logging.warning(f"Geocoding API returned no results for city: {city_name}")
            return {'error': f'City "{city_name}" not found.'}
        
        result = {
            'lat': data[0].get('lat'),
            'lon': data[0].get('lon'),
            'name': data[0].get('name'),
            'country': data[0].get('country', '')
        }
        
        if result['lat'] is None or result['lon'] is None:
            logging.error(f"Geocoding API response missing lat/lon for {city_name}: {data[0]}")
            return {'error': f'Incomplete location data for "{city_name}".'}
        
        logging.debug(f"Coordinates found for {city_name}: {result}")
        return result
        
    except requests.exceptions.Timeout:
        logging.error(f"Geocoding API request timed out for city: {city_name}")
        return {'error': 'Geocoding service timed out.'}
    except requests.exceptions.RequestException as e:
        logging.error(f"Geocoding API request error for city {city_name}: {e}")
        return {'error': f'Could not connect to geocoding service: {e}'}
    except Exception as e:
        logging.exception(f"Unexpected error in get_coords_from_city for {city_name}: {e}")
        return {'error': 'An unexpected error occurred during geocoding.'}
# --- REMOVED key_prefix ---
# --- @cache.cached(timeout=300) # Let Flask-Caching use args for key
def fetch_aqi(lat, lon, city_name_display):
    # Cache key will now be based on function name + lat + lon + city_name_display
    logging.debug(f"Fetching AQI for {city_name_display} ({lat}, {lon})")
    # ... (rest of function is correct) ...
    api_key = current_app.config.get('OPENWEATHER_API_KEY')
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    if not api_key: logging.error("OPENWEATHER_API_KEY not configured for fetch_aqi."); return {'error': 'Server configuration error: API key missing.'}
    params = {'lat': lat, 'lon': lon, 'appid': api_key}
    try:
        response = requests.get(url, params=params, timeout=10); response.raise_for_status()
        api_response_data = response.json(); data_list = api_response_data.get('list', [])
        if not data_list: logging.warning(f"Air Pollution API returned empty list for ({lat}, {lon})"); return {'error': 'Air Pollution data currently unavailable for this location.'}
        data = data_list[0]; comp = data.get('components', {}); dt_timestamp = data.get('dt')
        aqi_value, main_pollutant = calculate_indian_aqi(comp)
        result = {'aqi': aqi_value, 'main_pollutant': main_pollutant, 'city': city_name_display, 'geo': [lat, lon], 'pm25': comp.get('pm2_5', 'N/A'), 'pm10': comp.get('pm10', 'N/A'), 'no': comp.get('no', 'N/A'), 'no2': comp.get('no2', 'N/A'), 'so2': comp.get('so2', 'N/A'), 'co': comp.get('co', 'N/A'), 'o3': comp.get('o3', 'N/A'), 'nh3': comp.get('nh3', 'N/A'), 'updated': datetime.fromtimestamp(dt_timestamp).strftime('%d %b %Y, %I:%M %p') if dt_timestamp else 'N/A'}
        logging.debug(f"AQI Result for {city_name_display}: {result}"); return result
    except requests.exceptions.Timeout: logging.error(f"Air Pollution API request timed out for ({lat}, {lon})"); return {'error': 'Air pollution service timed out.'}
    except requests.exceptions.RequestException as e: logging.error(f"Air Pollution API request error for ({lat}, {lon}): {e}"); return {'error': f'Could not connect to air pollution service: {e}'}
    except IndexError: logging.error(f"Air Pollution API returned malformed list for ({lat}, {lon})"); return {'error': 'Received incomplete air pollution data.'}
    except Exception as e: logging.exception(f"Unexpected error in fetch_aqi for {city_name_display} ({lat}, {lon}): {e}"); return {'error': 'An unexpected error occurred fetching AQI.'}


# --- REMOVED key_prefix ---
# --- @cache.cached(timeout=900) # Let Flask-Caching use args for key
def fetch_weather(lat, lon, city_name_display):
    # Cache key will now be based on function name + lat + lon + city_name_display
    logging.debug(f"Fetching Weather for {city_name_display} ({lat}, {lon})")
    # ... (rest of function is correct) ...
    api_key = current_app.config.get('OPENWEATHER_API_KEY')
    url = "https://api.openweathermap.org/data/2.5/weather"
    if not api_key: logging.error("OPENWEATHER_API_KEY not configured for fetch_weather."); return {'error': 'Server configuration error: API key missing.'}
    params = {'lat': lat, 'lon': lon, 'appid': api_key, 'units': 'metric'}
    try:
        response = requests.get(url, params=params, timeout=10); response.raise_for_status()
        data = response.json()
        sys_data = data.get('sys', {}); tz_shift = data.get('timezone', 0)
        try: tz = timezone(timedelta(seconds=int(tz_shift)))
        except ValueError: logging.warning(f"Invalid timezone offset {tz_shift} for ({lat}, {lon}), using UTC."); tz = timezone.utc
        sunrise_ts = sys_data.get('sunrise'); sunset_ts = sys_data.get('sunset')
        sunrise = datetime.fromtimestamp(sunrise_ts, tz=tz).strftime('%I:%M %p') if sunrise_ts else 'N/A'
        sunset = datetime.fromtimestamp(sunset_ts, tz=tz).strftime('%I:%M %p') if sunset_ts else 'N/A'
        main_data = data.get('main', {}); wind_data = data.get('wind', {}); weather_list = data.get('weather', [{}]); weather_info = weather_list[0] if weather_list else {}
        result = {'temp': round(main_data.get('temp', 0)), 'feels_like': round(main_data.get('feels_like', 0)), 'pressure': main_data.get('pressure', 'N/A'), 'humidity': main_data.get('humidity', 'N/A'), 'wind_speed': round(wind_data.get('speed', 0) * 3.6, 1), 'visibility': round(data.get('visibility', 10000) / 1000, 1), 'description': weather_info.get('description', 'N/A').title(), 'icon': weather_info.get('icon', '01d'), 'sunrise': sunrise, 'sunset': sunset}
        logging.debug(f"Weather Result for {city_name_display}: {result}"); return result
    except requests.exceptions.Timeout: logging.error(f"Weather API request timed out for ({lat}, {lon})"); return {'error': 'Weather service timed out.'}
    except requests.exceptions.RequestException as e: logging.error(f"Weather API request error for ({lat}, {lon}): {e}"); return {'error': f'Weather data unavailable: {e}'}
    except Exception as e: logging.exception(f"Unexpected error in fetch_weather for {city_name_display} ({lat}, {lon}): {e}"); return {'error': 'An unexpected error occurred fetching weather.'}


# --- Weather-only forecast helper (No Changes Needed) ---
# ... ( _process_daily_forecast function remains the same) ...
def _process_daily_forecast(forecast_list):
    """Processes 3-hourly forecast timestamps into a 5-day daily summary."""
    daily_data = defaultdict(lambda: {
        'temps': [], 'icons': [], 'descs': [], 'pops': [], 'dt': []
    })
    hourly_forecast_slice = [] # To store the next ~24h raw data

    now_ts = time.time()
    limit_ts = now_ts + 30 * 3600 # Approx 30 hours limit for hourly slice

    # Group data by UTC day AND collect hourly slice
    for entry in forecast_list:
        try:
            dt_ts = entry['dt']
            main_data = entry['main']
            weather_list = entry.get('weather', [{}])
            weather_info = weather_list[0] if weather_list else {}

            utc_dt = datetime.fromtimestamp(dt_ts, tz=timezone.utc)
            day_str = utc_dt.strftime('%Y-%m-%d')

            # --- Collect hourly data for the chart ---
            if dt_ts >= now_ts and dt_ts <= limit_ts:
                 hourly_forecast_slice.append({
                     'time': utc_dt.strftime('%I%p').lstrip('0'), # Format as 9AM, 12PM etc.
                     'temp': round(main_data.get('temp', 0)),
                     'pop': round(entry.get('pop', 0) * 100), # Probability 0-100
                     'icon': weather_info.get('icon', '01d')
                 })
            # --- End hourly collection ---

            daily_data[day_str]['temps'].append(main_data['temp'])
            daily_data[day_str]['pops'].append(entry.get('pop', 0))
            daily_data[day_str]['dt'].append(dt_ts)

            hour = utc_dt.hour
            if hour >= 11 and hour <= 13:
                daily_data[day_str]['icons'].insert(0, weather_info.get('icon', '01d'))
                daily_data[day_str]['descs'].insert(0, weather_info.get('description', 'Clear Sky').title())
            else:
                daily_data[day_str]['icons'].append(weather_info.get('icon', '01d'))
                daily_data[day_str]['descs'].append(weather_info.get('description', 'Clear Sky').title())
        except KeyError as e: logging.warning(f"Skipping forecast entry due to missing key {e}: {entry}"); continue
        except Exception as e: logging.exception(f"Error processing forecast entry {entry}: {e}"); continue

    daily_summary = []
    sorted_days = sorted(daily_data.items(), key=lambda item: min(item[1]['dt']))
    for day_str, data in sorted_days[:5]:
        if not data['temps']: continue
        try:
            day_dt = datetime.strptime(day_str, '%Y-%m-%d')
            daily_summary.append({'day': day_dt.strftime('%a, %b %d'), 'temp_min': round(min(data['temps'])), 'temp_max': round(max(data['temps'])), 'max_pop': round(max(data['pops']) * 100) if data['pops'] else 0, 'icon': data['icons'][0], 'desc': data['descs'][0]})
        except Exception as e: logging.exception(f"Error finalizing daily forecast for {day_str}: {e}")

    # Return both the daily summary and the hourly slice
    return daily_summary, hourly_forecast_slice


# NO @cache.cached(...) decorator
def fetch_forecast(lat, lon):
    logging.debug(f"Fetching Weather Forecast ({lat}, {lon})")
    api_key = current_app.config.get('OPENWEATHER_API_KEY')
    url = "https://api.openweathermap.org/data/2.5/forecast"
    if not api_key: logging.error("OPENWEATHER_API_KEY not configured for fetch_forecast."); return [], [] # Return two empty lists
    params = {'lat': lat, 'lon': lon, 'appid': api_key, 'units': 'metric'}
    try:
        response = requests.get(url, params=params, timeout=10); response.raise_for_status()
        api_response_data = response.json(); full_forecast_list = api_response_data.get('list', [])
        if not full_forecast_list: logging.warning(f"Weather forecast API returned empty list for ({lat}, {lon})"); return [], []
        # Process returns two lists now
        daily_summary, hourly_slice = _process_daily_forecast(full_forecast_list)
        logging.debug(f"Weather Forecast Result (Daily Count): {len(daily_summary)}, (Hourly Count): {len(hourly_slice)}")
        return daily_summary, hourly_slice # Return both
    except requests.exceptions.Timeout: logging.error(f"Weather Forecast API request timed out for ({lat}, {lon})"); return [], []
    except requests.exceptions.RequestException as e: logging.error(f"Weather Forecast API request error for ({lat}, {lon}): {e}"); return [], []
    except Exception as e: logging.exception(f"Unexpected error in fetch_forecast for ({lat}, {lon}): {e}"); return [], []

# --- Historical AQI Fetching & Simulation (No Changes Needed) ---
# ... (fetch_historical_aqi and _simulate_historical_if_needed functions remain the same) ...
def fetch_historical_aqi(lat, lon):
    logging.debug(f"Fetching Historical AQI ({lat}, {lon})")
    api_key = current_app.config.get('OPENWEATHER_API_KEY')
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    if not api_key: logging.error("OPENWEATHER_API_KEY not configured..."); logging.warning(f"Simulating historical AQI..."); return _simulate_historical_if_needed([])
    end_time_dt = datetime.now(timezone.utc); start_time_dt = end_time_dt - timedelta(hours=24)
    end_time = int(end_time_dt.timestamp()); start_time = int(start_time_dt.timestamp())
    params = {'lat': lat, 'lon': lon, 'start': start_time, 'end': end_time, 'appid': api_key}
    try:
        response = requests.get(url, params=params, timeout=15); response.raise_for_status()
        data = response.json().get('list', [])
        historical = []
        for entry in data:
            components = entry.get('components'); dt_ts = entry.get('dt')
            if components is not None and dt_ts is not None:
                aqi_value, _ = calculate_indian_aqi(components)
                historical.append({'dt': dt_ts, 'hour': datetime.fromtimestamp(dt_ts, tz=timezone.utc).strftime('%H:00'), 'aqi': aqi_value})
            else: logging.warning(f"Skipping historical entry: {entry}")
        historical.sort(key=lambda x: x['dt'])
        if len(historical) < 20: logging.warning(f"Historical AQI API returned {len(historical)} points. Simulating."); return _simulate_historical_if_needed(historical)
        logging.debug(f"Historical AQI Result (count): {len(historical)}")
        return [{'hour': item['hour'], 'aqi': item['aqi']} for item in historical]
    except requests.exceptions.HTTPError as e:
         if e.response.status_code == 401: logging.error(f"Historical AQI API key invalid. Simulating.")
         elif e.response.status_code == 429: logging.error(f"Historical AQI API rate limit exceeded. Simulating.")
         else: logging.error(f"Historical AQI API HTTP error {e.response.status_code}: {e}. Simulating.")
         return _simulate_historical_if_needed([])
    except requests.exceptions.Timeout: logging.error(f"Historical AQI API timed out. Simulating."); return _simulate_historical_if_needed([])
    except requests.exceptions.RequestException as e: logging.error(f"Historical AQI API request error: {e}. Simulating."); return _simulate_historical_if_needed([])
    except Exception as e: logging.exception(f"Unexpected error fetching historical AQI: {e}. Simulating."); return _simulate_historical_if_needed([])

def _simulate_historical_if_needed(partial_data):
    logging.debug(f"Simulating historical AQI data. Based on {len(partial_data)} real points.")
    base_aqi = 50
    if partial_data:
        valid_aqi_values = [item['aqi'] for item in reversed(partial_data) if isinstance(item.get('aqi'), (int, float))]
        if valid_aqi_values: base_aqi = int(valid_aqi_values[0]); logging.debug(f"Simulation base AQI set to {base_aqi}.")
        else: logging.debug("No valid numeric AQI found, using default base 50.")
    simulated_historical = []
    now_utc = datetime.now(timezone.utc)
    for i in range(24):
        hour_dt = now_utc - timedelta(hours=i); hour_ts = int(hour_dt.timestamp())
        real_point = next((p for p in partial_data if abs(p['dt'] - hour_ts) < 1800), None)
        if real_point and real_point['aqi'] != 'N/A':
            simulated_historical.append({'hour': hour_dt.strftime('%H:00'), 'aqi': real_point['aqi']})
        else:
            variation = 15 * math.sin(i * math.pi / 12) + (5 * math.sin(i * math.pi / 4))
            simulated_aqi = max(10, int(base_aqi + variation))
            simulated_historical.append({'hour': hour_dt.strftime('%H:00'), 'aqi': simulated_aqi})
    logging.debug(f"Generated {len(simulated_historical)} historical points (simulated+real).")
    return simulated_historical[::-1]

# --- Tip Selection Logic (No Changes Needed) ---
# ... (get_relevant_tips function remains the same) ...
def get_relevant_tips(aqi_data, context='home'):
    logging.debug(f"Getting relevant tips for context '{context}' and AQI data: {aqi_data}")
    tips = []
    aqi_value = None
    if aqi_data and 'error' not in aqi_data and aqi_data.get('aqi') != 'N/A':
        try: aqi_value = int(aqi_data['aqi'])
        except (ValueError, TypeError): logging.warning(f"Invalid AQI value '{aqi_data.get('aqi')}' for tip selection.")
    category_map = {'home': ['home'], 'outdoors': ['personal', 'community'], 'commuting': ['transport']}
    target_categories = category_map.get(context, ['home'])
    try:
        base_query = Tip.query.filter(Tip.category.in_(target_categories))
        if aqi_value is not None:
            if aqi_value > 200:
                tips = base_query.filter(Tip.impact == 'high').order_by(db.func.random()).limit(3).all()
                if len(tips) < 3: tips.extend(base_query.filter(Tip.impact == 'medium', Tip.id.notin_([t.id for t in tips])).order_by(db.func.random()).limit(3 - len(tips)).all())
            elif aqi_value > 100:
                tips = base_query.filter(Tip.difficulty == 'easy').order_by(db.func.random()).limit(3).all()
                if len(tips) < 3: tips.extend(base_query.filter(Tip.impact == 'medium', Tip.difficulty != 'easy', Tip.id.notin_([t.id for t in tips])).order_by(db.func.random()).limit(3 - len(tips)).all())
            else:
                tips = base_query.filter(Tip.impact == 'low').order_by(db.func.random()).limit(3).all()
                if len(tips) < 3: tips.extend(base_query.filter(Tip.difficulty == 'easy', Tip.id.notin_([t.id for t in tips])).order_by(db.func.random()).limit(3- len(tips)).all())
        else:
            logging.info(f"AQI invalid, providing generic tips for context '{context}'")
            tips = base_query.filter(Tip.difficulty == 'easy').order_by(db.func.random()).limit(3).all()
            if len(tips) < 3: tips.extend(base_query.filter(Tip.impact == 'low', Tip.id.notin_([t.id for t in tips])).order_by(db.func.random()).limit(3 - len(tips)).all())
        final_tips = list({tip.id: tip for tip in tips}.values())
        logging.debug(f"Selected {len(final_tips)} relevant tips for context '{context}'.")
        return final_tips[:3]
    except Exception as e: logging.exception(f"Database error fetching tips for context '{context}': {e}"); return []