# 🌍 AirWatch — Localized AQI Monitoring & ML Prediction Dashboard

<div align="center">

![AirWatch](https://img.shields.io/badge/AirWatch-AQI%20Monitoring-green?style=for-the-badge&logo=python)

**AI-powered air quality monitoring and short-term AQI prediction platform**

[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=flat-square&logo=postgresql)](https://www.postgresql.org/)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square&logo=render)](https://airwatch-0j5i.onrender.com)

🔗 **Live Demo**: [airwatch-0j5i.onrender.com](https://airwatch-0j5i.onrender.com)

</div>

---

## 🎯 Project Overview

AirWatch delivers a localized, explainable air quality monitoring and short-term prediction platform to help citizens, institutions, and municipal planners make informed health decisions. The system provides real-time pollutant breakdowns, hourly weather forecasts, and an ML-backed short-term AQI prediction (1–24 hours).

Built as a capstone project aligned with **SDG 13 (Climate Action)**.

---

## ✨ Key Features

- 🤖 **ML-Powered AQI Prediction** — RandomForest model predicting AQI from 12 pollutant features
- 📊 **Interactive Dashboard** — Real-time AQI cards, pollutant composition charts, hourly & 5-day weather forecast
- 🗺️ **Map View** — Leaflet-based interactive map with color-coded city AQI markers
- 🔍 **Explainability** — Feature importance visualization showing top pollutant contributors
- 💡 **Health Tips** — Personalized mitigation tips based on CPCB AQI guidelines
- 📈 **Historical Trends** — 24-hour AQI trend charts with simulation fallback
- 🔐 **User Auth** — Signup/login with secure password hashing, profile & favorite cities
- 🌐 **Live Weather** — OpenWeather API integration for real-time weather and forecasts

---

## 🏗️ Architecture

```
┌───────────────────────┐     ┌─────────────────────┐
│   Frontend (Static)   │────▶│   Flask Backend      │
│  Chart.js + Leaflet   │     │   18+ REST Endpoints │
│  Jinja2 Templates     │     │                      │
└───────────────────────┘     └─────────┬───────────┘
                                        │
                    ┌──────────┬────────┼─────────┐
                    │          │        │         │
             ┌──────────┐ ┌────────┐ ┌───────┐ ┌──────────┐
             │ ML Model │ │ DB     │ │ Cache │ │ External │
             │ (joblib) │ │ PG/SQL │ │       │ │ APIs     │
             │ RF Model │ │ ite    │ │       │ │ OpenWx   │
             └──────────┘ └────────┘ └───────┘ └──────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python, Flask 3.1, Gunicorn, Flask-SQLAlchemy, Flask-Caching |
| **ML / Data** | scikit-learn, RandomForestRegressor, joblib, pandas, numpy |
| **Database** | PostgreSQL (production), SQLite (local development) |
| **Frontend** | Vanilla JS, Chart.js, Leaflet.js, Jinja2 HTML Templates |
| **External APIs** | OpenWeather (geocoding, AQI, weather, forecast) |
| **Deployment** | Docker, Render.com, GitHub Releases (model versioning) |

---

## 📁 Project Structure

```
AirWatch/
├── app.py                      # Flask app factory, health endpoint, DB seeding
├── config.py                   # Config (PostgreSQL/SQLite, API keys, caching)
├── extensions.py               # SQLAlchemy + Flask-Caching instances
├── models.py                   # ORM models: User, Tip, Favorite
├── ml_handler.py               # ML model loader, predictor, AQI category helpers
├── train_random_forest.py      # CLI training script (RandomForest + joblib)
├── wsgi.py                     # Gunicorn entry point
│
├── routes/
│   ├── api.py                  # REST API (18+ endpoints: auth, AQI, weather, ML)
│   ├── auth.py                 # Auth page routes (login/signup/logout)
│   ├── main.py                 # Page-serving routes (dashboard, map, predictor)
│   └── utils.py                # Data helpers (CPCB AQI calc, weather, geocoding)
│
├── templates/                  # 11 Jinja2 HTML templates
│   ├── base.html               # Base layout (navbar, footer)
│   ├── dashboard.html          # Main AQI dashboard
│   ├── map.html                # Leaflet map view
│   ├── predictor.html          # ML predictor page
│   ├── profile.html            # User profile & favorites
│   ├── tips.html               # Health mitigation tips
│   └── ...                     # index, login, signup, about, guide
│
├── static/
│   ├── css/styles.css          # Global stylesheet
│   └── js/
│       ├── dashboard.js        # Dashboard charts & data fetching
│       ├── predictor_v3.js     # Predictor form, gauge, contribution chart
│       ├── map.js              # Leaflet map initialization
│       └── main.js             # Auth handlers & utilities
│
├── data/city_day.csv           # Training dataset (~2.5 MB)
├── Dockerfile                  # Python 3.11-slim + Gunicorn
├── requirements.txt            # 13 pinned dependencies
├── runtime.txt                 # Render runtime (python-3.11.4)
└── .gitignore
```

---

## 🤖 ML Model

| Detail | Value |
|--------|-------|
| **Algorithm** | RandomForestRegressor (scikit-learn) |
| **Features** | PM2.5, PM10, NO, NO₂, NOx, NH₃, CO, SO₂, O₃, Benzene, Toluene, Xylene |
| **Imputation** | Median (SimpleImputer) |
| **Hyperparameters** | n_estimators=80, max_features=sqrt, n_jobs=1 |
| **Serialization** | joblib with compress=3 (~1 MB compressed) |
| **Explainability** | Feature importance-based pollutant contribution breakdown |

### Train the Model

```bash
python train_random_forest.py --csv data/city_day.csv --n-estimators 80 --max-depth 12
```

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.11+
- pip
- OpenWeather API key (free tier)

### 1. Clone & Setup

```bash
git clone https://github.com/SamarthBurkul/AirWatch-.git
cd AirWatch-
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Environment Variables

Create `.env` in project root:

```env
OPENWEATHER_API_KEY=your_openweather_api_key
```

### 3. Run Locally

```bash
python app.py
```

Open **http://localhost:5000** 🚀

---

## 🐳 Docker

```bash
docker build -t airwatch .
docker run -p 5000:5000 --env-file .env airwatch
```

---

## 🚀 Deployment (Render)

- **Web Service**: Docker-based, Gunicorn on port 5000
- **Database**: PostgreSQL 16 (Render managed)
- **Model Loading**: Background thread auto-downloads model from GitHub Releases at startup

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/signup` | User registration |
| POST | `/api/login` | User login |
| GET | `/api/aqi/<city>` | Current AQI for a city |
| GET | `/api/weather/<city>` | Current weather |
| GET | `/api/forecast/<city>` | 5-day weather forecast |
| GET | `/api/historical/<city>` | 24-hour AQI history |
| POST | `/api/predict` | ML-based AQI prediction |
| GET | `/api/top-cities` | Top Indian & world cities AQI |
| GET | `/api/tips` | All health tips |
| GET | `/api/tips/context` | Context-aware tips |
| GET | `/api/map/cities` | Map marker data |
| GET | `/api/autocomplete` | City name suggestions |
| POST | `/api/update-city` | Update preferred city |
| POST | `/api/favorites/add` | Add favorite city |
| DELETE | `/api/favorites/remove` | Remove favorite city |

---

## 🔐 Security

- 🔑 Werkzeug password hashing (scrypt/bcrypt compatible, 255-char hash column)
- 🔒 Session-based authentication with secure cookies
- 🛡️ CORS configured for allowed origins
- ⚙️ `postgres://` → `postgresql://` URI fix for Render compatibility

---

## 📊 Impact & Use Cases

- 🏃 **Citizens**: Avoid outdoor activity when AQI > 200
- 🏥 **Health**: Pollutant-specific mitigation tips aligned with CPCB guidelines
- 🏙️ **Municipal**: Spatial hotspot detection via interactive map
- 🔬 **Research**: Reproducible model artifacts via GitHub Releases

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit and push your changes
4. Open a Pull Request

---

## 📄 License

This project is intended for **academic, demo, and educational use**. For commercial usage, contact the maintainer.

---

<div align="center">

**🌍 AirWatch — Making air quality data intelligent, visual, and actionable.**

*SDG 13: Climate Action*

</div>
