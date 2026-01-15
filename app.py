import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from extensions import db, cache
from config import Config
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

# 2. Configure high-level threading settings for ML stability
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 3. Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Try import background loader from ml_handler
try:
    from ml_handler import background_model_loader
except Exception:
    background_model_loader = None

def create_app():
    """Application Factory: Initializes Flask, Extensions, and Blueprints."""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    
    # Load settings from Config class
    app.config.from_object(Config)

    # 4. Correct Cache Configuration
    app.config.update({
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 300
    })

    # 5. Initialize Extensions
    db.init_app(app)
    cache.init_app(app)
    CORS(app)

    # 6. Register Blueprints (FIXED: Added url_prefix for API)
    from routes.main import main_bp
    from routes.auth import auth_bp
    from routes.api import api_bp

    app.register_blueprint(main_bp)  # Handles / and /dashboard
    app.register_blueprint(auth_bp, url_prefix='/auth')  # Handles /auth/login etc
    app.register_blueprint(api_bp, url_prefix='/api')  # ✅ FIXED: Now handles /api/* endpoints

    # 7. Admin Health Check for Reviewers
    # 7. Admin Health Check for Reviewers
@app.route('/health')
def health():
    from ml_handler import _loaded_package
    return jsonify({
        "status": "online",
        "ml_model_loaded": _loaded_package is not None,
        "database": app.config['SQLALCHEMY_DATABASE_URI'].split('://')[0]  # Shows 'postgresql' or 'sqlite'
    })

    # 8. Database and Seed Logic
    with app.app_context():
        from models import Tip
        db.create_all()  # Create tables if they don't exist
        seed_tips(db)   # Populate health advice

        # Start ML model loader in background thread
        if background_model_loader:
            background_model_loader(delay_seconds=1)
            logger.info("Started background model loader thread.")

    return app

def seed_tips(database):
    """Seed the database with CPCB-aligned health tips."""
    from models import Tip
    if database.session.query(Tip).count() == 0:
        tips_data = [
            {
                'title': 'Run HEPA Air Purifiers', 
                'description': 'HEPA filters capture 99.97% of fine particles (PM2.5), reducing indoor pollution significantly.', 
                'category': 'home', 
                'difficulty': 'medium', 
                'impact': 'high', 
                'pollutants_targeted': 'PM2.5, PM10', 
                'related_diseases': 'Asthma, Lung Irritation'
            },
            {
                'title': 'Wear N95 Masks Outdoors',
                'description': 'N95 respirators filter out at least 95% of airborne particles when AQI exceeds 200.',
                'category': 'outdoor',
                'difficulty': 'easy',
                'impact': 'high',
                'pollutants_targeted': 'PM2.5, PM10',
                'related_diseases': 'Respiratory infections, COPD'
            },
            {
                'title': 'Avoid Peak Traffic Hours',
                'description': 'Vehicle emissions peak during rush hours (8-10 AM, 6-8 PM). Stay indoors during these times.',
                'category': 'outdoor',
                'difficulty': 'medium',
                'impact': 'medium',
                'pollutants_targeted': 'NO2, CO, PM2.5',
                'related_diseases': 'Cardiovascular disease, Asthma'
            },
            {
                'title': 'Keep Indoor Plants',
                'description': 'Plants like snake plants and spider plants naturally filter benzene, formaldehyde, and CO2.',
                'category': 'home',
                'difficulty': 'easy',
                'impact': 'low',
                'pollutants_targeted': 'Benzene, Formaldehyde, CO2',
                'related_diseases': 'Allergies, Headaches'
            },
            {
                'title': 'Monitor AQI Before Exercise',
                'description': 'Avoid strenuous outdoor activity when AQI exceeds 150. Exercise indoors or reschedule.',
                'category': 'health',
                'difficulty': 'easy',
                'impact': 'high',
                'pollutants_targeted': 'PM2.5, O3',
                'related_diseases': 'Asthma, Heart conditions'
            }
        ]
        for t_data in tips_data:
            database.session.add(Tip(**t_data))
        database.session.commit()
        logger.info("✅ Database seeded with health tips.")

if __name__ == '__main__':
    # Entry point for local development
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)