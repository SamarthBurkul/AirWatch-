import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'samarth-airwatch-sdg13-2026')  # fallback for local dev
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///airwatch.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Read OpenWeather API key from environment (no secrets in repo)
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

    # Geocoding API endpoint (still safe to keep here)
    GEOCODING_API_URL = os.getenv('GEOCODING_API_URL', "http://api.openweathermap.org/geo/1.0/direct")
