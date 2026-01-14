// predictor_v3.js - updated for robust JSON handling and safe fallbacks
document.addEventListener('DOMContentLoaded', () => {

    // ------------------------
    // 1) EARLY FALLBACKS (MUST come first)
    // ------------------------
    // Provide a safe global getAqiCategoryInfo so charts/plugins can call it immediately.
    if (typeof window.getAqiCategoryInfo === 'undefined') {
        window.getAqiCategoryInfo = function(aqi) {
            // Normalize input
            const a = (aqi === null || aqi === undefined || isNaN(Number(aqi))) ? -1 : Number(aqi);
            // Conservative mapping used by UI (textColor, borderColor, bgColor, chartColor)
            if (a <= 50 && a >= 0) {
                return { category: "Good", description: "Minimal impact.", textColor: "text-green-400", borderColor: "border-green-500", bgColor: "bg-green-500/20", chartColor: "#34d399" };
            }
            if (a <= 100) { return { category: "Satisfactory", description: "Minor breathing discomfort.", textColor: "text-yellow-400", borderColor: "border-yellow-500", bgColor: "bg-yellow-500/20", chartColor: "#f59e0b" }; }
            if (a <= 200) { return { category: "Moderate", description: "Breathing discomfort to sensitive groups.", textColor: "text-orange-400", borderColor: "border-orange-500", bgColor: "bg-orange-500/20", chartColor: "#f97316" }; }
            if (a <= 300) { return { category: "Poor", description: "Breathing discomfort to most people.", textColor: "text-red-400", borderColor: "border-red-500", bgColor: "bg-red-500/20", chartColor: "#ef4444" }; }
            if (a <= 400) { return { category: "Very Poor", description: "Respiratory illness on prolonged exposure.", textColor: "text-purple-400", borderColor: "border-purple-500", bgColor: "bg-purple-500/20", chartColor: "#a855f7" }; }
            if (a > 400) { return { category: "Severe", description: "Serious health effects.", textColor: "text-rose-700", borderColor: "border-rose-700", bgColor: "bg-rose-800/20", chartColor: "#be123c" }; }
            return { category: "N/A", description: "", textColor: "text-slate-400", borderColor: "border-slate-500", bgColor: "bg-slate-500/10", chartColor: "#64748b" };
        };
    }
    // Provide showToast fallback if main app toast isn't loaded
    if (typeof window.showToast === 'undefined') {
        window.showToast = function(message, isError = false) {
            // Non-blocking by default — console log + optional alert for dev
            if (isError) console.error('[Toast]', message);
            else console.log('[Toast]', message);
            // Uncomment below to show a blocking alert during dev:
            // alert((isError ? 'ERROR: ' : '') + message);
        };
    }

    // ------------------------
    // 2) DOM ELEMENTS
    // ------------------------
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

    // Chart instances
    let resultGaugeChart = null;
    let contributionChart = null;

    // ------------------------
    // 3) Gauge text plugin (safe register)
    // ------------------------
    const gaugeTextPlugin = {
        id: 'gaugeText',
        beforeDraw: (chart) => {
            if (chart.config.type !== 'doughnut' || !chart.canvas || chart.canvas.id !== 'resultGaugeChart') return;
            const { ctx, width, height } = chart;
            const aqi = chart.config.data.datasets?.[0]?.data?.[0];
            if (aqi === undefined || aqi === null) return;
            // Use the safe global helper
            const categoryInfo = (typeof getAqiCategoryInfo === 'function') ? getAqiCategoryInfo(aqi) : window.getAqiCategoryInfo(aqi);
            ctx.restore();
            const fontSizeTitle = (height / 114).toFixed(2);
            const fontSizeCategory = (height / 220).toFixed(2);
            const colorMap = {
                'text-green-400': '#34d399', 'text-yellow-400': '#f59e0b', 'text-orange-400': '#f97316',
                'text-red-400': '#ef4444', 'text-purple-400': '#a855f7', 'text-rose-700': '#be123c', 'text-slate-400': '#9ca3af'
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
        try { Chart.register(gaugeTextPlugin); } catch (e) { console.warn('Could not register gaugeText plugin', e); }
    }

    // ------------------------
    // 4) CHART DRAW FUNCTIONS
    // ------------------------
    function drawResultGauge(aqiValue) {
        const gaugeCtx = document.getElementById('resultGaugeChart')?.getContext('2d');
        if (!gaugeCtx) { console.error("Result Gauge Chart canvas not found."); return; }
        if (resultGaugeChart) resultGaugeChart.destroy();

        let displayAqi = 0;
        let categoryInfo = window.getAqiCategoryInfo(-1);
        if (aqiValue !== null && aqiValue !== undefined && !isNaN(aqiValue)) {
            displayAqi = parseInt(aqiValue);
            categoryInfo = window.getAqiCategoryInfo(displayAqi);
        }

        resultGaugeChart = new Chart(gaugeCtx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [displayAqi, Math.max(0, 500 - displayAqi)],
                    backgroundColor: [categoryInfo.chartColor || '#64748b', 'rgba(255, 255, 255, 0.1)'],
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
            .filter(([k, v]) => v !== null && v !== undefined && !isNaN(v) && Number(v) > 0)
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
            data: { labels: labels, datasets: [{ label: 'Sub-Index Value', data: values, backgroundColor: backgroundColors, borderColor: backgroundColors, borderWidth: 1 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: false } },
                scales: { y: { title: { display: true, text: 'Calculated Sub-Index' }, beginAtZero: true } }
            }
        });
    }

    // ------------------------
    // 5) GEOLOCATION & PRE-FILL
    // ------------------------
    async function fillWithCurrentData() {
        if (!navigator.geolocation) { showToast("Geolocation is not supported by this browser.", true); return; }

        if (!useCurrentDataBtn || !geolocateIcon || !geolocateSpinner || !geolocateText) { console.error("Geolocation button elements missing."); return; }
        useCurrentDataBtn.disabled = true; geolocateIcon.classList.add('hidden');
        geolocateSpinner.classList.remove('hidden'); geolocateText.textContent = 'Getting Location...';

        try {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 8000, enableHighAccuracy: false, maximumAge: 300000 });
            });
            const lat = position.coords.latitude; const lon = position.coords.longitude;
            geolocateText.textContent = 'Fetching Pollutants...'; console.log(`Geolocation acquired: ${lat}, ${lon}`);

            const response = await fetch(`/api/current_pollutants?lat=${lat}&lon=${lon}`);
            if (!response.ok) {
                let errorMsg = `Failed to fetch pollutant data: ${response.statusText}`;
                try { const errorData = await response.json(); if (errorData.error) errorMsg = errorData.error; } catch (e) {}
                throw new Error(errorMsg);
            }
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            console.log("Fetched current pollutant data:", data);

            const featuresToFill = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'];
            let filledCount = 0;
            featuresToFill.forEach(feature => {
                const input = predictForm.elements[feature];
                if (input) {
                    const value = data[feature];
                    if (value === null || value === undefined || value === '') {
                        input.value = '';
                    } else {
                        try {
                            input.value = Number(value).toFixed(2);
                            filledCount++;
                        } catch (e) {
                            input.value = value;
                        }
                    }
                    input.style.borderColor = '';
                }
            });

            if (filledCount > 0) showToast('Form filled with available location data. Some fields may be unavailable.', false);
            else showToast('Could not fetch pollutant data for your location.', true);

        } catch (error) {
            console.error("Error getting current data:", error);
            let userMessage = `Error: ${error.message}`;
            if (error.code === 1) userMessage = "Geolocation permission denied.";
            else if (error.code === 2) userMessage = "Could not determine location (position unavailable).";
            else if (error.code === 3) userMessage = "Geolocation request timed out.";
            showToast(userMessage, true);
        } finally {
            useCurrentDataBtn.disabled = false;
            geolocateIcon.classList.remove('hidden');
            geolocateSpinner.classList.add('hidden');
            geolocateText.textContent = 'Use My Location Data';
        }
    }
    if (useCurrentDataBtn) useCurrentDataBtn.addEventListener('click', fillWithCurrentData);

    // ------------------------
    // 6) PREDICT FORM SUBMISSION (with robust JSON handling)
    // ------------------------
    if (predictForm && predictBtn && predictionResultArea) {

        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log("Predict form submitted.");

            // Basic validation
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

            // UI loading state
            predictBtn.disabled = true; if (predictBtnText) predictBtnText.classList.add('hidden');
            if (predictSpinner) predictSpinner.classList.remove('hidden'); if (predictionResultArea) predictionResultArea.classList.add('hidden');

            try {
                // Gather form data
                const formData = new FormData(predictForm); const dataObject = {};
                const features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'];
                features.forEach(feature => { dataObject[feature] = formData.get(feature); });

                // POST to server
                const response = await fetch('/api/predict_aqi', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(dataObject)
                });

                // Robust parsing: read text first to handle empty/non-JSON bodies
                const text = await response.text();
                let parsed;
                try {
                    parsed = text ? JSON.parse(text) : null;
                } catch (err) {
                    console.error('Invalid JSON from /api/predict_aqi:', text);
                    throw new Error(parsed?.error || parsed?.detail || text || `Request failed with status ${response.status}`);
                }

                if (!response.ok || !parsed || !parsed.success) {
                    const errMsg = (parsed && (parsed.error || parsed.detail)) || `Request failed with status ${response.status}`;
                    throw new Error(errMsg);
                }

                const data = parsed;

                // Map server category_info to UI-friendly keys (backwards-compatible)
                const sc = data.category_info || {};
                const categoryInfo = {
                    category: sc.category || 'N/A',
                    description: sc.description || sc.desc || '',
                    textColor: sc.textColor || (sc.color_class ? extractTextColorFromClass(sc.color_class) : 'text-slate-400'),
                    borderColor: sc.borderColor || sc.color_class || 'border-slate-500',
                    bgColor: sc.bgColor || 'bg-slate-500/10',
                    chartColor: sc.chartColor || '#64748b'
                };

                // Update result area UI
                const aqi = data.predicted_aqi;
                predictionResultText.innerHTML = `<span class="text-2xl font-semibold ${categoryInfo.textColor}">${categoryInfo.category}</span><br><span class="text-sm text-slate-400">${categoryInfo.description}</span>`;
                predictionResultText.className = `mt-4 text-center p-4 rounded-lg border-t-4 ${categoryInfo.borderColor} ${categoryInfo.bgColor}`;

                // Draw charts
                drawResultGauge(aqi);
                drawContributionChart(data.subindices || {});

                predictionResultArea.classList.remove('hidden');
                console.log('Prediction successful:', data);

            } catch (error) {
                console.error('Prediction submit error:', error);
                // Show nice error message to user
                predictionResultText.textContent = `Error: ${error.message || 'Unknown error'}`;
                predictionResultText.className = 'mt-4 text-center p-4 rounded-lg border-l-4 bg-red-500/20 text-red-300 border-red-500';
                predictionResultArea.classList.remove('hidden');
                if (resultGaugeChart) { try { resultGaugeChart.destroy(); } catch (e) {} resultGaugeChart = null; }
                if (contributionChart) { try { contributionChart.destroy(); } catch (e) {} contributionChart = null; }
            } finally {
                predictBtn.disabled = false;
                if (predictBtnText) predictBtnText.classList.remove('hidden');
                if (predictSpinner) predictSpinner.classList.add('hidden');
            }
        });
    } else {
        console.warn("Could not initialize prediction form listener. Check HTML IDs:", {
            predictForm: !!predictForm, predictBtn: !!predictBtn, predictBtnText: !!predictBtnText,
            predictSpinner: !!predictSpinner, predictionResultText: !!predictionResultText, predictionResultArea: !!predictionResultArea
        });
    }

    // ------------------------
    // 7) Small helpers
    // ------------------------
    function extractTextColorFromClass(colorClass) {
        // colorClass might be something like "bg-red-500/20 text-red-300 border-red-500"
        if (!colorClass || typeof colorClass !== 'string') return 'text-slate-400';
        const m = colorClass.match(/text-[-\w]+/);
        return m ? m[0] : 'text-slate-400';
    }

}); // end DOMContentLoaded
