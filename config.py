import os

class Config:
    """Configuration for both local development and Render production."""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'samarth-airwatch-sdg13-2026')
    
    # Database Configuration
    # Render provides DATABASE_URL for PostgreSQL, fallback to SQLite for local dev
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    if DATABASE_URL:
        # CRITICAL FIX: Render uses 'postgres://' but SQLAlchemy needs 'postgresql://'
        if DATABASE_URL.startswith('postgres://'):
            DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        SQLALCHEMY_DATABASE_URI = DATABASE_URL
    else:
        # Local development: use SQLite
        SQLALCHEMY_DATABASE_URI = 'sqlite:///airwatch.db'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,  # Verify connections before using
        'pool_recycle': 300,    # Recycle connections after 5 minutes
    }
    
    # OpenWeather API
    OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
    # ✅ CHANGED: http → https
    GEOCODING_API_URL = "https://api.openweathermap.org/geo/1.0/direct"
    
    # Flask-Caching
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Session Configuration
    SESSION_COOKIE_SECURE = os.environ.get('FLASK_ENV') == 'production'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'