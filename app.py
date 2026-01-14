# app.py
import logging
from flask import Flask
from flask_cors import CORS
from extensions import db, cache
from config import Config

# Try import background loader from ml_handler (optional)
try:
    from ml_handler import background_model_loader
except Exception:
    background_model_loader = None

# configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)
cors = CORS()

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)

    # Add Cache Config (kept from your code)
    cache_config = {
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 300
    }
    app.config.from_mapping(cache_config)

    # Initialize extensions with app
    db.init_app(app)
    cache.init_app(app)
    cors.init_app(app)

    # Import and register blueprints
    from routes.main import main_bp
    from routes.auth import auth_bp
    from routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    with app.app_context():
        from models import Tip
        db.create_all()
        seed_tips(db)

        # start background model loader if available
        if background_model_loader:
            try:
                background_model_loader(delay_seconds=1)
                logger.info("Started background model loader.")
            except Exception:
                logger.exception("Failed to start background model loader thread.")
        else:
            logger.warning("background_model_loader import unavailable; ML model loader disabled.")

    return app


def seed_tips(database):
    """Seed the database with an expanded list of tips if it's empty."""
    from models import Tip
    if database.session.query(Tip).count() == 0:
        tips_data = [
            # (existing tips array you already have - keep all items)
            {'title': 'Run HEPA Air Purifiers', 'description': 'HEPA filters capture 99.97% of fine particles (PM2.5), dust, and allergens. Run them in frequently used rooms like bedrooms and living areas.', 'category': 'home', 'difficulty': 'medium', 'impact': 'high', 'pollutants_targeted': 'PM2.5, PM10, Allergens', 'related_diseases': 'Asthma, Allergies, Lung Irritation'},
            # ... keep all your tips here exactly as in your previous seed_tips ...
        ]
        for t_data in tips_data:
            tip = Tip(**t_data)
            database.session.add(tip)
        database.session.commit()
        print("✅ Seeded expanded tips data into the database.")


if __name__ == '__main__':
    app = create_app()
    # In production use gunicorn; this debug run is for local testing
    app.run(debug=True, host='0.0.0.0', port=5000)
