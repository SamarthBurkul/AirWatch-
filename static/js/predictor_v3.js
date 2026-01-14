// static/js/predictor_v3.js  (FULL UPDATED FILE)
document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const predictForm = document.getElementById('predict-form');
    const predictionResultArea = document.getElementById('prediction-result-area');
    const predictionResultText = document.getElementById('prediction-result');
    const predictBtn = predictForm ? predictForm.querySelector('button[type="submit"]') : null;
    const predictBtnText = document.getElementById('predict-btn-text');
    const predictSpinner = document.getElementById('predict-spinner');
    const useCurrentDataBtn = document.getElementById('use-current-data-btn');
    const geolocateIcon = document.getElementById('geolocate-icon');
    const geolocateSpinner = document.getElementById('geolocate-spinner');
    const geolocateText = document.getElementById('geolocate-text');

    let resultGaugeChart = null;
    let contributionChart = null;

    // --- Provide a robust getAqiCategoryInfo implementation (prevents undefined errors) ---
    // This mirrors the shape the backend returns (textColor, borderColor, bgColor, chartColor)
    function getAqiCategoryInfo(aqi) {
        let aqiVal = Number(aqi);
        if (Number.isNaN(aqiVal)) aqiVal = null;

        if (aqiVal === null) {
            return {
                category: 'N/A',
                description: 'AQI data invalid.',
                textColor: 'text-slate-400',
                borderColor: 'border-slate-500',
                bgColor: 'bg-slate-500/10',
                chartColor: '#64748b'
            };
        }
        if (aqiVal <= 50) {
            return { category: 'Good', description: 'Minimal impact.', textColor: 'text-green-400', borderColor: 'border-green-500', bgColor: 'bg-green-500/20', chartColor: '#34d399' };
        }
        if (aqiVal <= 100) {
            return { category: 'Satisfactory', description: 'Minor breathing discomfort.', textColor: 'text-yellow-400', borderColor: 'border-yellow-500', bgColor: 'bg-yellow-500/20', chartColor: '#f59e0b' };
        }
        if (aqiVal <= 200) {
            return { category: 'Moderate', description: 'Breathing discomfort to sensitive groups.', textColor: 'text-orange-400', borderColor: 'border-orange-500', bgColor: 'bg-orange-500/20', chartColor: '#f97316' };
        }
        if (aqiVal <= 300) {
            return { category: 'Poor', description: 'Breathing discomfort to most people.', textColor: 'text-red-400', borderColor: 'border-red-500', bgColor: 'bg-red-500/20', chartColor: '#ef4444' };
        }
        if (aqiVal <= 400) {
            return { category: 'Very Poor', description: 'Respiratory illness on prolonged exposure.', textColor: 'text-purple-400', borderColor: 'border-purple-500', bgColor: 'bg-purple-500/20', chartColor: '#a855f7' };
        }
        return { category: 'Severe', description: 'Serious health effects.', textColor: 'text-rose-700', borderColor: 'border-rose-700', bgColor: 'bg-rose-800/20', chartColor: '#be123c' };
    }

    // --- Chart plugin safe registration (uses getAqiCategoryInfo above) ---
    const gaugeTextPlugin = {
        id: 'gaugeText',
        beforeDraw: (chart) => {
            if (chart.config.type !== 'doughnut' || chart.canvas.id !== 'resultGaugeChart') return;
            const { ctx, width, height } = chart;
            const aqi = chart.config.data.datasets?.[0]?.data?.[0];
            if (aqi === undefined || aqi === null) return;
            const categoryInfo = getAqiCategoryInfo(aqi);
            ctx.restore();
            const fontSizeTitle = (height / 114).toFixed(2);
            const fontSizeCategory = (height / 220).toFixed(2);
            const colorMap = {
                'text-green-400': '#34d399',
                'text-yellow-400': '#f59e0b',
                'text-orange-400': '#f97316',
                'text-red-400': '#ef4444',
                'text-purple-400': '#a855f7',
                'text-rose-700': '#be123c',
                'text-slate-400': '#9ca3af'
            };
            ctx.font = `bold ${fontSizeTitle}rem Poppins, sans-serif`;
            ctx.textBaseline = 'middle';
            ctx.textAlign = 'center';
            ctx.fillStyle = colorMap[categoryInfo.textColor] || '#9ca3af';
            ctx.fillText(aqi, width / 2, height / 2 - 10);
            ctx.font = `600 ${fontSizeCategory}rem Poppins, sans-serif`;
            ctx.fillStyle = '#9ca3af';
            ctx.fillText("AQI", width / 2, height / 2 + 20);
            ctx.save();
        }
    };

    if (typeof Chart !== 'undefined' && Chart.register) {
        try { Chart.register(gaugeTextPlugin); } catch (e) { console.warn("Could not register gaugeText plugin.", e); }
    }

    // --- Utility: safe JSON parsing to avoid "Unexpected end of JSON input" ---
    async function safeParseJsonResponse(response) {
        // Reads the response text and attempts parse; returns object {parsed, error, rawText}
        const ct = response.headers ? (response.headers.get('content-type') || '') : '';
        const text = await response.text();
        if (!text || text.trim().length === 0) {
            return { parsed: null, error: `Empty response (status ${response.status})`, rawText: '' };
        }
        // Prefer JSON.parse when it looks like JSON
        const looksLikeJson = ct.includes('application/json') || text.trim().startsWith('{') || text.trim().startsWith('[');
        if (looksLikeJson) {
            try {
                const parsed = JSON.parse(text);
                return { parsed, error: null, rawText: text };
            } catch (err) {
                return { parsed: null, error: 'Invalid JSON received from server', rawText: text };
            }
        }
        // Non-JSON but non-empty
        return { parsed: null, error: `Unexpected content-type (${ct || 'unknown'})`, rawText: text };
    }

    // --- Chart draw helpers ---
    function drawResultGauge(aqiValue) {
        const gaugeCtx = document.getElementById('resultGaugeChart')?.getContext('2d');
        if (!gaugeCtx) { console.error("Result Gauge Chart canvas not found."); return; }
        if (resultGaugeChart) resultGaugeChart.destroy();

        let displayAqi = 0;
        let categoryInfo = getAqiCategoryInfo(-1);
        if (aqiValue !== null && aqiValue !== undefined && !isNaN(aqiValue)) {
            displayAqi = parseInt(aqiValue);
            categoryInfo = getAqiCategoryInfo(displayAqi);
        }

        resultGaugeChart = new Chart(gaugeCtx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [displayAqi, Math.max(0, 500 - displayAqi)],
                    backgroundColor: [categoryInfo.chartColor, 'rgba(255, 255, 255, 0.1)'],
                    borderWidth: 0, cutout: '80%'
                }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { enabled: false }, gaugeText: {} } }
        });
    }

    function drawContributionChart(subindices) {
        const contribCtx = document.getElementById('contributionChart')?.getContext('2d');
        if (!contribCtx) { console.error("Contribution Chart canvas not found."); return; }
        if (contributionChart) contributionChart.destroy();

        const chartData = Object.entries(subindices || {})
            .filter(([k, v]) => v !== null && v !== undefined && v > 0)
            .sort(([, a], [, b]) => b - a);

        if (chartData.length === 0) {
            contribCtx.clearRect(0, 0, contribCtx.canvas.width, contribCtx.canvas.height);
            contribCtx.fillStyle = '#9ca3af';
            contribCtx.textAlign = 'center';
            contribCtx.fillText('No significant pollutant contribution detected', contribCtx.canvas.width / 2, contribCtx.canvas.height / 2);
            return;
        }

        const labels = chartData.map(([k]) => k);
        const values = chartData.map(([, v]) => v);
        const backgroundColors = ['#38bdf8', '#8b5cf6', '#f97316', '#ef4444', '#eab308', '#22c55e', '#6366f1'].slice(0, labels.length);

        contributionChart = new Chart(contribCtx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{ label: 'Sub-Index Value', data: values, backgroundColor: backgroundColors, borderColor: backgroundColors, borderWidth: 1 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: false } },
                scales: { y: { title: { display: true, text: 'Calculated Sub-Index' }, beginAtZero: true } }
            }
        });
    }

    // --- Geolocation & fill form with current pollutants ---
    async function fillWithCurrentData() {
        if (!navigator.geolocation) { showToast("Geolocation is not supported by this browser.", true); return; }
        if (!useCurrentDataBtn || !geolocateIcon || !geolocateSpinner || !geolocateText) { console.error("Geolocation button elements missing."); return; }

        useCurrentDataBtn.disabled = true;
        geolocateIcon.classList.add('hidden');
        geolocateSpinner.classList.remove('hidden');
        geolocateText.textContent = 'Getting Location...';

        try {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 8000, enableHighAccuracy: false, maximumAge: 300000 });
            });
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            geolocateText.textContent = 'Fetching Pollutants...';
            console.log(`Geolocation acquired: ${lat}, ${lon}`);

            // fetch backend current pollutants
            const response = await fetch(`/api/current_pollutants?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`, { method: 'GET' });
            if (!response.ok) {
                const parsed = await safeParseJsonResponse(response);
                const msg = parsed.error || `Failed to fetch pollutant data: ${response.status} ${response.statusText}`;
                throw new Error(msg);
            }

            const parsed = await safeParseJsonResponse(response);
            if (parsed.error) throw new Error(parsed.error);

            const data = parsed.parsed;
            if (!data) throw new Error("No pollutant data returned.");

            console.log("Fetched current pollutant data:", data);

            const featuresToFill = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'];
            let filledCount = 0;
            featuresToFill.forEach(feature => {
                const input = predictForm && predictForm.elements ? predictForm.elements[feature] : null;
                if (input) {
                    const value = data[feature];
                    if (value === null || typeof value === 'undefined') {
                        input.value = '';
                    } else {
                        const n = Number(value);
                        input.value = Number.isFinite(n) ? n.toFixed(2) : String(value);
                        if (Number.isFinite(n)) filledCount++;
                    }
                    input.style.borderColor = '';
                }
            });

            if (filledCount > 0) {
                showToast('Form filled with available location data. Some fields may be unavailable.', false);
            } else {
                showToast('Could not fetch pollutant values to fill the form for your location.', true);
            }
        } catch (err) {
            console.error("Error getting current data:", err);
            let userMessage = `Error: ${err.message || String(err)}`;
            // Geolocation API error codes
            if (err && err.code === 1) userMessage = "Geolocation permission denied.";
            else if (err && err.code === 2) userMessage = "Could not determine location (position unavailable).";
            else if (err && err.code === 3) userMessage = "Geolocation request timed out.";
            showToast(userMessage, true);
        } finally {
            useCurrentDataBtn.disabled = false;
            geolocateIcon.classList.remove('hidden');
            geolocateSpinner.classList.add('hidden');
            geolocateText.textContent = 'Use My Location Data';
        }
    }

    if (useCurrentDataBtn) useCurrentDataBtn.addEventListener('click', fillWithCurrentData);

    // --- Prediction form submit ---
    if (predictForm && predictBtn && predictionResultArea) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log("Predict form submitted.");

            // Validate required numeric inputs (inputs with required attribute)
            let isValid = true;
            const requiredInputs = predictForm.querySelectorAll('input[type="number"][required]');
            requiredInputs.forEach((input) => {
                if (!input || typeof input.value === 'undefined' || !String(input.value).trim()) {
                    isValid = false;
                    if (input) input.style.borderColor = 'red';
                } else {
                    if (input) input.style.borderColor = '';
                }
            });
            if (!isValid) { showToast('Please fill all fields with valid numbers.', true); return; }

            // UI state
            predictBtn.disabled = true;
            if (predictBtnText) predictBtnText.classList.add('hidden');
            if (predictSpinner) predictSpinner.classList.remove('hidden');
            if (predictionResultArea) predictionResultArea.classList.add('hidden');

            try {
                const formData = new FormData(predictForm);
                const features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'];
                const payload = {};
                features.forEach(f => { payload[f] = formData.get(f); });

                // Send prediction request to backend
                const response = await fetch('/api/predict_aqi', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    // no-cache helps avoid stale responses during debug
                });

                if (!response.ok) {
                    // Parse response (if any) safely to show a useful message
                    const parsed = await safeParseJsonResponse(response);
                    const msg = parsed && parsed.parsed && parsed.parsed.error ? parsed.parsed.error : (parsed.error || `Request failed with status ${response.status}`);
                    throw new Error(msg);
                }

                const parsedOK = await safeParseJsonResponse(response);
                if (parsedOK.error) throw new Error(parsedOK.error);
                const data = parsedOK.parsed;
                if (!data || data.success === false) {
                    const errText = (data && (data.error || data.message)) || 'Prediction failed: unknown server response.';
                    throw new Error(errText);
                }

                const aqi = data.predicted_aqi;
                const categoryInfo = data.category_info || getAqiCategoryInfo(aqi);
                const subindices = data.subindices || {};

                // Update UI summary
                if (predictionResultText) {
                    predictionResultText.innerHTML = `<span class="text-2xl font-semibold ${categoryInfo.textColor}">${categoryInfo.category}</span><br><span class="text-sm text-slate-400">${categoryInfo.description}</span>`;
                    predictionResultText.className = `mt-4 text-center p-4 rounded-lg border-t-4 ${categoryInfo.borderColor} ${categoryInfo.bgColor}`;
                }

                drawResultGauge(aqi);
                drawContributionChart(subindices);

                if (predictionResultArea) predictionResultArea.classList.remove('hidden');
                console.log("Prediction successful:", data);

            } catch (err) {
                console.error("Prediction submit error:", err);
                const message = err && err.message ? err.message : String(err);
                if (predictionResultText) {
                    predictionResultText.textContent = `Error: ${message}`;
                    predictionResultText.className = 'mt-4 text-center p-4 rounded-lg border-l-4 bg-red-500/20 text-red-300 border-red-500';
                }
                if (predictionResultArea) predictionResultArea.classList.remove('hidden');
                // Destroy charts to avoid stale visuals
                if (resultGaugeChart) { resultGaugeChart.destroy(); resultGaugeChart = null; }
                if (contributionChart) { contributionChart.destroy(); contributionChart = null; }
            } finally {
                predictBtn.disabled = false;
                if (predictBtnText) predictBtnText.classList.remove('hidden');
                if (predictSpinner) predictSpinner.classList.add('hidden');
            }
        });
    } else {
        console.warn("Could not initialize prediction form - missing elements.", {
            predictForm: !!predictForm, predictBtn: !!predictBtn, predictBtnText: !!predictBtnText,
            predictSpinner: !!predictSpinner, predictionResultText: !!predictionResultText, predictionResultArea: !!predictionResultArea
        });
    }

    // --- Fallback showToast if not provided by main.js ---
    if (typeof window.showToast === 'undefined') {
        window.showToast = (message, isError = false) => {
            // Lightweight fallback; in production prefer site toast
            alert(`${isError ? 'Error' : 'Info'}: ${message}`);
        };
    }
}); // DOMContentLoaded end
