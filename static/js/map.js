// static/js/map.js

document.addEventListener('DOMContentLoaded', () => {
    const mapElement = document.getElementById('map');
    if (!mapElement) return; // Exit if not on the map page

    // --- Configuration ---
    const API_KEY = window.OPENWEATHER_API_KEY; // Get API key passed from Flask template
    const TILE_URL_BASE = `https://maps.openweathermap.org/maps/2.0/weather/`; // CHANGED: http → https
    const TILE_ATTRIBUTION = 'Weather Maps © OpenWeatherMap';

    // Global variable to track the currently active weather overlay layer
    let currentWeatherLayer = null;
    const markers = {}; // Object to store Leaflet markers

    // Initialize Leaflet map centered on India
    const map = L.map('map').setView([20.5937, 78.9629], 5);
    
    // Add base tiles (OpenStreetMap)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    // --- AQI CATEGORY HELPER (Copied from dashboard.js for standalone functionality) ---
    function getAqiCategoryInfo(aqi) {
         if (aqi === 'N/A' || aqi === null || aqi === undefined || aqi < 0) return { category: 'N/A', chartColor: '#64748b' };
         const aqiNum = parseInt(aqi);
         if (isNaN(aqiNum)) return { category: 'N/A', chartColor: '#64748b' };
         if (aqiNum <= 50) return { category: 'Good', chartColor: '#34d399' };
         if (aqiNum <= 100) return { category: 'Satisfactory', chartColor: '#f59e0b' };
         if (aqiNum <= 200) return { category: 'Moderate', chartColor: '#f97316' };
         if (aqiNum <= 300) return { category: 'Poor', chartColor: '#ef4444' };
         if (aqiNum <= 400) return { category: 'Very Poor', chartColor: '#a855f7' };
         return { category: 'Severe', chartColor: '#be123c' };
     }

    // Helper function to determine marker color and styling based on AQI
    function getAqiStyle(aqi) {
        const categoryInfo = getAqiCategoryInfo(aqi); 
        const colorMap = {
            '#34d399': '#22c55e', '#f59e0b': '#eab308', '#f97316': '#f97316', '#ef4444': '#ef4444',
            '#a855f7': '#a855f7', '#be123c': '#be123c', '#64748b': '#64748b'
        };
        const color = colorMap[categoryInfo.chartColor] || '#64748b';
        return { color: color, fillColor: color }; // Set marker color and fill
    }

    // --- Weather Layer Definitions and Legend Scales ---
    const LAYER_LEGENDS = {
        'TA2': { 
            title: 'Air Temperature (°C)',
            colors: ['#208CEC', '#23DDDD', '#C2FF28', '#FFF028', '#FFC228', '#FC8014'], // Cold to Hot scale
            values: ['< 0', '10', '20', '25', '30', '> 30'], // Values for the labels
            palette: 'TA2' 
        },
        'WND': { 
            title: 'Wind Speed (m/s)',
            colors: ['#FFF', '#EECECC', '#B364BC', '#4600AF', '#0D1126'], 
            values: ['< 1', '5', '15', '50', '> 100'],
            palette: 'WND'
        },
        'CL': { 
            title: 'Cloudiness (%)',
            colors: ['#FDFDFF', '#FAFAFF', '#F7F7FF', '#E9E9DF', '#D2D2D2'], 
            values: ['< 10', '30', '60', '90', '> 90'],
            palette: 'CL'
        },
        'APM': {
            title: 'Atmospheric Pressure (hPa)',
            colors: ['#0073FF', '#8DE7C7', '#B0F720', '#F0B800', '#C60000'],
            values: ['< 960', '1000', '1010', '1020', '> 1060'],
            palette: 'APM'
        },
        'PR0': {
             title: 'Precipitation Intensity (mm/s)',
             colors: ['#FEF9CA', '#93F57D', '#50B033', '#EB4726', '#B02318'],
             values: ['Very Light', 'Light', 'Moderate', 'Heavy', 'Very Heavy'],
             palette: 'PR0'
        }
    };

    // --- Legend Rendering Function (labels on the color segments) ---
    function renderLegend(layerCode) {
        const legendContainer = document.getElementById('layer-legend');
        const legendTitle = document.getElementById('legend-title');
        const legendScale = document.getElementById('legend-scale');
        
        if (layerCode === 'none' || !LAYER_LEGENDS[layerCode]) {
            legendContainer.classList.add('hidden');
            return;
        }

        const legend = LAYER_LEGENDS[layerCode];
        legendTitle.textContent = legend.title;
        legendScale.innerHTML = ''; // Clear previous content

        // === COLOR BAR WITH LABELS ON TOP ===
        const colorBar = document.createElement('div');
        colorBar.className = 'flex w-full h-6 rounded overflow-hidden border border-slate-700 text-[10px] text-white font-medium';
        colorBar.style.lineHeight = '1.2';
        colorBar.style.textShadow = '0 0 3px rgba(0, 0, 0, 0.8)'; // makes text readable over bright colors

        legend.colors.forEach((color, index) => {
            const segment = document.createElement('div');
            segment.style.backgroundColor = color;
            segment.style.flex = '1';
            segment.style.height = '100%';
            segment.style.display = 'flex';
            segment.style.alignItems = 'center';
            segment.style.justifyContent = 'center';
            segment.style.padding = '0 2px';
            segment.style.boxSizing = 'border-box';

            const label = document.createElement('span');
            label.textContent = legend.values[index] || '';
            label.style.whiteSpace = 'nowrap';
            label.style.textAlign = 'center';
            segment.appendChild(label);

            colorBar.appendChild(segment);
        });

        legendScale.appendChild(colorBar);
        legendContainer.classList.remove('hidden');
    }

    // --- Weather Layer Toggle Logic ---
    const layerSelect = document.getElementById('weatherLayerSelect');
    if (layerSelect) {
        layerSelect.addEventListener('change', (e) => {
            const layerCode = e.target.value;
            toggleWeatherLayer(layerCode);
            renderLegend(layerCode); // RENDER LEGEND when layer changes
        });
    }

    function toggleWeatherLayer(layerCode) {
        // Removes the current tile layer and adds a new one based on selection
        if (currentWeatherLayer) {
            map.removeLayer(currentWeatherLayer);
            currentWeatherLayer = null;
        }

        if (layerCode && layerCode !== 'none') {
            const tileUrl = `${TILE_URL_BASE}${layerCode}/{z}/{x}/{y}?appid=${API_KEY}`;
            
            currentWeatherLayer = L.tileLayer(tileUrl, {
                maxZoom: 18,
                opacity: 0.6, // Semi-transparent for better base map visibility
                attribution: TILE_ATTRIBUTION
            }).addTo(map);

            if (typeof showToast !== 'undefined') {
                showToast(`Displaying ${layerCode} map overlay.`, false);
            }
        }
    }
    // --- End Weather Layer Toggle Logic ---


    // --- Marker Management (addCityMarker) ---
    function addCityMarker(cityData) {
        // Adds or updates a circular marker showing AQI/Weather data via popup
        if (!cityData || cityData.aqi === 'N/A' || !cityData.geo || !cityData.geo[0]) return;

        let aqi = 'N/A';
        try {
            if (typeof cityData.aqi === 'number') aqi = cityData.aqi;
            else if (typeof cityData.aqi === 'string') aqi = parseInt(cityData.aqi);
            else throw new Error("AQI not number or string");
            if (isNaN(aqi)) aqi = -1;
        } catch (e) { aqi = -1; console.warn(`[addCityMarker] AQI parse error: ${e}.`); }

        const coords = cityData.geo;
        const style = getAqiStyle(aqi);
        const weather = cityData.weather;
        const officialCityName = cityData.city || 'Unknown City';

        // Weather Handling for Popup
        let weatherContent;
        if (weather && !weather.error) {
            const temp = weather.temp !== undefined ? `${weather.temp}°C` : 'N/A';
            const description = weather.description || 'N/A';
            const humidity = weather.humidity !== undefined ? `${weather.humidity}%` : 'N/A';
            const wind = weather.wind_speed !== undefined ? `${weather.wind_speed} km/h` : 'N/A';
            weatherContent = `<p><b>Weather:</b> ${description}</p><p><b>Temp:</b> ${temp}</p><p><b>Humidity:</b> ${humidity}</p><p><b>Wind:</b> ${wind}</p>`;
        } else {
            weatherContent = `<p class="text-slate-400 italic">Weather data unavailable.</p>`;
        }

        const popupContent = `
            <div class="p-1 text-sm">
                <h3 class="font-bold text-base mb-1">${officialCityName}</h3>
                <p><b>AQI:</b> <span class="font-bold text-lg" style="color:${style.color};">${cityData.aqi}</span> (${getAqiCategoryInfo(aqi).category})</p>
                <hr class="my-1 border-slate-600">
                ${weatherContent}
            </div>
        `;

        // Calculate visual properties
        let displayAqiNum = (typeof aqi === 'number' && aqi >= 0) ? aqi : 50;
        let radius = 6 + (displayAqiNum / 25);
        radius = Math.max(6, Math.min(radius, 20));

        const markerOptions = { ...style, radius: radius, weight: 1, opacity: 1, fillOpacity: 0.7 };
        const cityNameKey = officialCityName.toLowerCase();
        const existingMarkerData = markers[cityNameKey];

        if (existingMarkerData && existingMarkerData.marker) {
            existingMarkerData.marker.setLatLng(coords);
            existingMarkerData.marker.setStyle(markerOptions);
            existingMarkerData.marker.setRadius(radius);
            existingMarkerData.marker.setPopupContent(popupContent);
        } else {
            const marker = L.circleMarker(coords, markerOptions).addTo(map);
            marker.bindPopup(popupContent);
            marker.on('mouseover', function (e) { this.openPopup(); this.setStyle({ weight: 3 }); });
            marker.on('mouseout', function (e) { this.closePopup(); this.setStyle({ weight: 1 }); });
            markers[cityNameKey] = { marker, coords };
        }
    } 
    // --- End Marker Management ---


    // --- Data Fetching Logic ---
    async function loadInitialCities() {
        try {
            const response = await fetch('/api/map_cities_data');
            if (!response.ok) {
                // If endpoint doesn't exist, silently skip initial load
                if (response.status === 404) {
                    console.warn('Initial cities endpoint not found. Skipping.');
                    return;
                }
                throw new Error(`Failed to fetch initial map data: ${response.statusText}`);
            }
            const cities = await response.json();
            if (Array.isArray(cities) && cities.length > 0) {
                 cities.forEach(addCityMarker);
            }
        } catch (error) {
            console.error('Error in loadInitialCities:', error);
            // Don't show toast for missing endpoint
            if (error.message.includes('404')) return;
            if (typeof showToast !== 'undefined') { 
                showToast(error.message || "Could not load initial city data.", true); 
            }
        }
    }

    const searchInput = document.getElementById('mapSearchInput');
    const searchButton = document.getElementById('mapSearchButton');

    const performSearch = async () => {
        const cityName = searchInput.value.trim();
        if (!cityName) return;

        searchInput.disabled = true;
        if (searchButton) searchButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

        try {
            const response = await fetch(`/api/city_data/${encodeURIComponent(cityName)}`);
            
            // Check if response is JSON
            const contentType = response.headers.get("content-type");
            if (!contentType || !contentType.includes("application/json")) {
                throw new Error(`Server returned non-JSON response for "${cityName}"`);
            }
            
            const cityData = await response.json();

            if (!response.ok || cityData.error) {
                throw new Error(cityData.error || `Data fetch failed: ${response.statusText}`);
            }
            
            addCityMarker(cityData); 
            if (cityData.geo) map.flyTo(cityData.geo, 10);

        } catch (error) {
            console.error("Error during performSearch:", error);
            if (typeof showToast !== 'undefined') { 
                showToast(error.message || `Could not find data for "${cityName}".`, true); 
            }

            // Fallback: Try to fly to location using Geocoding API
            try {
                 if (!API_KEY) return;
                 // CHANGED: http → https
                 const geoResponse = await fetch(`https://api.openweathermap.org/geo/1.0/direct?q=${encodeURIComponent(cityName)}&limit=1&appid=${API_KEY}`);
                 if (geoResponse.ok) {
                     const geoData = await geoResponse.json();
                     if (geoData && geoData.length > 0) {
                         map.flyTo([geoData[0].lat, geoData[0].lon], 10);
                     }
                 }
            } catch (geoError) { 
                console.error("Geocoding fallback failed:", geoError); 
            }
        } finally {
             searchInput.disabled = false;
             if (searchButton) searchButton.innerHTML = '<i class="fas fa-search"></i>';
        }
    };

    // --- Event Listeners ---
    if (searchInput) {
        searchInput.addEventListener('keypress', async (e) => {
            if (e.key === 'Enter') { e.preventDefault(); await performSearch(); }
        });
    }
    if (searchButton) {
        searchButton.addEventListener('click', async (e) => { e.preventDefault(); await performSearch(); });
    }

    loadInitialCities(); 
});