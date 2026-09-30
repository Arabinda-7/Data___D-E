/**
 * Weather Prediction System - 100% Offline Client Script
 * Zero External Dependencies (No Google Fonts, No CDN scripts)
 */

document.addEventListener('DOMContentLoaded', () => {
    let currentUnit = 'C'; // 'C' or 'F'
    let lastPredictionData = null;
    let debounceTimer = null;
    let hoverIndex = null;

    // Sliders
    const inputHour = document.getElementById('input-hour');
    const inputMonth = document.getElementById('input-month');
    const inputSeason = document.getElementById('input-season');
    const inputDayName = document.getElementById('input-day-name');
    const inputTempLag = document.getElementById('input-temp-lag');
    const inputHumidity = document.getElementById('input-humidity');
    const inputPressure = document.getElementById('input-pressure');
    const inputWind = document.getElementById('input-wind');
    const inputCloud = document.getElementById('input-cloud');
    const inputRain = document.getElementById('input-rain');

    // Number Inputs
    const numTempLag = document.getElementById('num-temp-lag');
    const numHumidity = document.getElementById('num-humidity');
    const numPressure = document.getElementById('num-pressure');
    const numWind = document.getElementById('num-wind');
    const numCloud = document.getElementById('num-cloud');
    const numRain = document.getElementById('num-rain');

    // Display labels
    const valHour = document.getElementById('val-hour');
    const valMonth = document.getElementById('val-month');

    // Results elements
    const heroTempVal = document.getElementById('hero-temp-val');
    const heroUnitLabel = document.getElementById('hero-unit-label');
    const heroBadge = document.getElementById('hero-badge');
    const heroFeelsLike = document.getElementById('hero-feels-like');
    const heroIcon = document.getElementById('hero-icon');
    const heroAdvice = document.getElementById('hero-advice');

    const metricHumidity = document.getElementById('metric-humidity');
    const metricHumiditySub = document.getElementById('metric-humidity-sub');
    const metricPressure = document.getElementById('metric-pressure');
    const metricWind = document.getElementById('metric-wind');
    const metricWindSub = document.getElementById('metric-wind-sub');
    const metricCloud = document.getElementById('metric-cloud');
    const metricRainSub = document.getElementById('metric-rain-sub');

    // Historical banner
    const historicalBanner = document.getElementById('historical-banner');
    const histTime = document.getElementById('hist-time');
    const histActual = document.getElementById('hist-actual');
    const histError = document.getElementById('hist-error');

    // Buttons
    const btnRunForecast = document.getElementById('btn-run-forecast');
    const btnReset = document.getElementById('btn-reset-inputs');
    const btnRandomSample = document.getElementById('btn-random-sample');
    const btnCelsius = document.getElementById('btn-celsius');
    const btnFahrenheit = document.getElementById('btn-fahrenheit');
    const presetButtons = document.querySelectorAll('.btn-preset[data-preset]');

    // Canvas
    const canvas = document.getElementById('diurnalCanvas');
    const ctx = canvas.getContext('2d');

    const monthNames = [
        "", "Jan (Month 1)", "Feb (Month 2)", "Mar (Month 3)", "Apr (Month 4)", "May (Month 5)", "Jun (Month 6)",
        "Jul (Month 7)", "Aug (Month 8)", "Sep (Month 9)", "Oct (Month 10)", "Nov (Month 11)", "Dec (Month 12)"
    ];

    const presetsData = {
        'summer_heat': {
            hour: 14, month: 5, season: 'Summer', day_name: 'Monday',
            temp_1h_ago: 33.5, relative_humidity_2m: 30.0, surface_pressure: 910.0,
            wind_speed_10m: 16.0, cloud_cover: 10.0, precipitation: 0.0
        },
        'monsoon_storm': {
            hour: 18, month: 8, season: 'Monsoon', day_name: 'Friday',
            temp_1h_ago: 22.0, relative_humidity_2m: 92.0, surface_pressure: 913.5,
            wind_speed_10m: 24.0, cloud_cover: 95.0, precipitation: 18.0
        },
        'winter_dawn': {
            hour: 6, month: 1, season: 'Winter', day_name: 'Sunday',
            temp_1h_ago: 14.5, relative_humidity_2m: 75.0, surface_pressure: 923.0,
            wind_speed_10m: 5.0, cloud_cover: 5.0, precipitation: 0.0
        },
        'pleasant_evening': {
            hour: 19, month: 3, season: 'Summer', day_name: 'Wednesday',
            temp_1h_ago: 27.0, relative_humidity_2m: 52.0, surface_pressure: 916.0,
            wind_speed_10m: 12.0, cloud_cover: 25.0, precipitation: 0.0
        },
        'post_monsoon': {
            hour: 13, month: 10, season: 'Post_Monsoon', day_name: 'Saturday',
            temp_1h_ago: 28.0, relative_humidity_2m: 58.0, surface_pressure: 915.0,
            wind_speed_10m: 9.0, cloud_cover: 40.0, precipitation: 0.0
        }
    };

    // Synchronize slider and number input
    function bindSliderAndNumber(slider, numberInput) {
        slider.addEventListener('input', () => {
            numberInput.value = slider.value;
            triggerDebouncedPrediction();
        });
        numberInput.addEventListener('input', () => {
            slider.value = numberInput.value;
            triggerDebouncedPrediction();
        });
    }

    bindSliderAndNumber(inputTempLag, numTempLag);
    bindSliderAndNumber(inputHumidity, numHumidity);
    bindSliderAndNumber(inputPressure, numPressure);
    bindSliderAndNumber(inputWind, numWind);
    bindSliderAndNumber(inputCloud, numCloud);
    bindSliderAndNumber(inputRain, numRain);

    function updateSliderLabels() {
        const h = parseInt(inputHour.value);
        const ampm = h >= 12 ? (h === 12 ? '12 PM' : `${h - 12} PM`) : (h === 0 ? '12 AM' : `${h} AM`);
        valHour.textContent = `${String(h).padStart(2, '0')}:00 (${ampm})`;
        valMonth.textContent = monthNames[parseInt(inputMonth.value)] || `Month ${inputMonth.value}`;
    }

    inputHour.addEventListener('input', () => {
        updateSliderLabels();
        triggerDebouncedPrediction();
    });

    inputMonth.addEventListener('input', () => {
        updateSliderLabels();
        triggerDebouncedPrediction();
    });

    inputSeason.addEventListener('change', triggerDebouncedPrediction);
    inputDayName.addEventListener('change', triggerDebouncedPrediction);

    function gatherFormData() {
        return {
            hour: parseInt(inputHour.value),
            month: parseInt(inputMonth.value),
            season: inputSeason.value,
            day_name: inputDayName.value,
            temp_1h_ago: parseFloat(inputTempLag.value),
            relative_humidity_2m: parseFloat(inputHumidity.value),
            surface_pressure: parseFloat(inputPressure.value),
            wind_speed_10m: parseFloat(inputWind.value),
            cloud_cover: parseFloat(inputCloud.value),
            precipitation: parseFloat(inputRain.value)
        };
    }

    function triggerDebouncedPrediction() {
        updateSliderLabels();
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            fetchPrediction(gatherFormData());
        }, 120);
    }

    async function fetchPrediction(payload) {
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const json = await res.json();
            if (json.status === 'success') {
                lastPredictionData = json.data;
                renderPredictionResults(json.data);
            }
        } catch (err) {
            console.error('Prediction error:', err);
        }
    }

    function renderPredictionResults(data) {
        const isC = currentUnit === 'C';
        const temp = isC ? data.predicted_temperature_c : data.predicted_temperature_f;
        const feels = isC ? data.feels_like_c : data.feels_like_f;
        const unitStr = isC ? '°C' : '°F';

        heroTempVal.textContent = temp.toFixed(1);
        heroUnitLabel.textContent = unitStr;
        heroBadge.textContent = data.comfort_status;
        heroFeelsLike.textContent = `${feels.toFixed(1)} ${unitStr}`;
        heroIcon.textContent = data.icon;
        heroAdvice.textContent = data.advice;

        metricHumidity.textContent = data.summary_metrics.relative_humidity;
        const humNum = parseFloat(data.summary_metrics.relative_humidity);
        metricHumiditySub.textContent = humNum > 75 ? 'Humid' : (humNum < 40 ? 'Dry' : 'Moderate');

        metricPressure.textContent = data.summary_metrics.surface_pressure;
        metricWind.textContent = data.summary_metrics.wind_speed;
        const windNum = parseFloat(data.summary_metrics.wind_speed);
        metricWindSub.textContent = windNum > 20 ? 'Breezy' : 'Calm';

        metricCloud.textContent = `${data.summary_metrics.cloud_cover} • ${data.summary_metrics.precipitation}`;
        const rainNum = parseFloat(data.summary_metrics.precipitation);
        metricRainSub.textContent = rainNum > 5 ? 'Rain Expected' : (parseFloat(data.summary_metrics.cloud_cover) > 70 ? 'Overcast' : 'Clear');

        drawDiurnalCanvasChart(data.diurnal_forecast);
    }

    // =========================================================================
    // PURE OFFLINE HTML5 CANVAS CHART RENDERER (Zero Dependencies)
    // =========================================================================
    function drawDiurnalCanvasChart(forecast) {
        if (!forecast || forecast.length === 0) return;

        const isC = currentUnit === 'C';
        const unit = isC ? '°C' : '°F';
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const width = rect.width;
        const height = rect.height;
        const padding = { top: 25, right: 30, bottom: 35, left: 45 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        ctx.clearRect(0, 0, width, height);

        const temps = forecast.map(d => isC ? d.temp_c : d.temp_f);
        const minTemp = Math.floor(Math.min(...temps) - 1.5);
        const maxTemp = Math.ceil(Math.max(...temps) + 1.5);
        const currentHour = parseInt(inputHour.value);

        // Draw Y Gridlines & Labels
        const ySteps = 4;
        ctx.font = '11px system-ui, -apple-system, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.strokeStyle = '#f1f5f9';
        ctx.lineWidth = 1;

        for (let i = 0; i <= ySteps; i++) {
            const val = minTemp + ((maxTemp - minTemp) / ySteps) * i;
            const y = padding.top + plotHeight - (i / ySteps) * plotHeight;
            
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();

            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(`${val.toFixed(0)}${unit}`, padding.left - 8, y);
        }

        // Draw X Gridlines & Hour Labels
        const xStep = plotWidth / (forecast.length - 1);
        const points = [];

        forecast.forEach((item, i) => {
            const x = padding.left + i * xStep;
            const tempVal = isC ? item.temp_c : item.temp_f;
            const y = padding.top + plotHeight - ((tempVal - minTemp) / (maxTemp - minTemp)) * plotHeight;
            points.push({ x, y, hour: item.hour, temp: tempVal });

            if (i % 4 === 0 || i === forecast.length - 1) {
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillStyle = '#94a3b8';
                ctx.fillText(item.hour, x, height - padding.bottom + 8);
            }
        });

        // Draw Filled Area Gradient
        const grad = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotHeight);
        grad.addColorStop(0, 'rgba(37, 99, 235, 0.18)');
        grad.addColorStop(1, 'rgba(37, 99, 235, 0.01)');

        ctx.beginPath();
        ctx.moveTo(points[0].x, padding.top + plotHeight);
        ctx.lineTo(points[0].x, points[0].y);

        for (let i = 0; i < points.length - 1; i++) {
            const xc = (points[i].x + points[i + 1].x) / 2;
            const yc = (points[i].y + points[i + 1].y) / 2;
            ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
        }
        ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
        ctx.lineTo(points[points.length - 1].x, padding.top + plotHeight);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        // Draw Line Curve
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 0; i < points.length - 1; i++) {
            const xc = (points[i].x + points[i + 1].x) / 2;
            const yc = (points[i].y + points[i + 1].y) / 2;
            ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
        }
        ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Draw Points & Highlight Current Hour
        points.forEach((p, idx) => {
            const isCur = idx === currentHour;
            const isHover = idx === hoverIndex;

            if (isCur || isHover) {
                // Highlight point
                ctx.beginPath();
                ctx.arc(p.x, p.y, isCur ? 6 : 5, 0, Math.PI * 2);
                ctx.fillStyle = isCur ? '#ef4444' : '#2563eb';
                ctx.fill();
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();

                // Value Tag
                const tagText = `${p.temp.toFixed(1)}${unit}`;
                ctx.font = 'bold 11px system-ui, -apple-system, sans-serif';
                const tagWidth = ctx.measureText(tagText).width + 10;
                const tagY = p.y - 18;

                ctx.fillStyle = isCur ? '#1e293b' : '#2563eb';
                ctx.beginPath();
                ctx.roundRect ? ctx.roundRect(p.x - tagWidth / 2, tagY - 10, tagWidth, 18, 4) : ctx.rect(p.x - tagWidth / 2, tagY - 10, tagWidth, 18);
                ctx.fill();

                ctx.fillStyle = '#ffffff';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(tagText, p.x, tagY - 1);
            } else if (idx % 3 === 0) {
                ctx.beginPath();
                ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = '#2563eb';
                ctx.fill();
            }
        });
    }

    // Canvas Interactive Hover Tracking
    canvas.addEventListener('mousemove', (e) => {
        if (!lastPredictionData || !lastPredictionData.diurnal_forecast) return;
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const paddingLeft = 45;
        const paddingRight = 30;
        const plotWidth = rect.width - paddingLeft - paddingRight;
        const xStep = plotWidth / (lastPredictionData.diurnal_forecast.length - 1);
        
        const index = Math.round((mouseX - paddingLeft) / xStep);
        if (index >= 0 && index < lastPredictionData.diurnal_forecast.length) {
            hoverIndex = index;
            drawDiurnalCanvasChart(lastPredictionData.diurnal_forecast);
        }
    });

    canvas.addEventListener('mouseleave', () => {
        hoverIndex = null;
        if (lastPredictionData) {
            drawDiurnalCanvasChart(lastPredictionData.diurnal_forecast);
        }
    });

    window.addEventListener('resize', () => {
        if (lastPredictionData) {
            drawDiurnalCanvasChart(lastPredictionData.diurnal_forecast);
        }
    });

    function applyFormData(data) {
        if (data.hour !== undefined) inputHour.value = data.hour;
        if (data.month !== undefined) inputMonth.value = data.month;
        if (data.season !== undefined) inputSeason.value = data.season;
        if (data.day_name !== undefined) inputDayName.value = data.day_name;
        
        if (data.temp_1h_ago !== undefined) {
            inputTempLag.value = data.temp_1h_ago;
            numTempLag.value = data.temp_1h_ago;
        }
        if (data.relative_humidity_2m !== undefined) {
            inputHumidity.value = data.relative_humidity_2m;
            numHumidity.value = data.relative_humidity_2m;
        }
        if (data.surface_pressure !== undefined) {
            inputPressure.value = data.surface_pressure;
            numPressure.value = data.surface_pressure;
        }
        if (data.wind_speed_10m !== undefined) {
            inputWind.value = data.wind_speed_10m;
            numWind.value = data.wind_speed_10m;
        }
        if (data.cloud_cover !== undefined) {
            inputCloud.value = data.cloud_cover;
            numCloud.value = data.cloud_cover;
        }
        if (data.precipitation !== undefined) {
            inputRain.value = data.precipitation;
            numRain.value = data.precipitation;
        }

        updateSliderLabels();
    }

    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            presetButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            historicalBanner.style.display = 'none';

            const key = btn.getAttribute('data-preset');
            const data = presetsData[key];
            if (data) {
                applyFormData(data);
                fetchPrediction(gatherFormData());
            }
        });
    });

    btnRandomSample.addEventListener('click', async () => {
        try {
            presetButtons.forEach(b => b.classList.remove('active'));
            const res = await fetch('/api/random-sample');
            const json = await res.json();
            if (json.status === 'success') {
                applyFormData(json.raw_inputs);
                
                histTime.textContent = json.timestamp;
                histActual.textContent = `${json.actual_temperature_c.toFixed(2)} °C`;
                histError.textContent = `${json.absolute_error_c.toFixed(2)} °C`;
                historicalBanner.style.display = 'flex';

                lastPredictionData = json.prediction;
                renderPredictionResults(json.prediction);
            }
        } catch (err) {
            console.error('Error fetching sample:', err);
        }
    });

    btnCelsius.addEventListener('click', () => {
        if (currentUnit !== 'C') {
            currentUnit = 'C';
            btnCelsius.classList.add('active');
            btnFahrenheit.classList.remove('active');
            if (lastPredictionData) renderPredictionResults(lastPredictionData);
        }
    });

    btnFahrenheit.addEventListener('click', () => {
        if (currentUnit !== 'F') {
            currentUnit = 'F';
            btnFahrenheit.classList.add('active');
            btnCelsius.classList.remove('active');
            if (lastPredictionData) renderPredictionResults(lastPredictionData);
        }
    });

    btnReset.addEventListener('click', () => {
        historicalBanner.style.display = 'none';
        applyFormData(presetsData['summer_heat']);
        presetButtons.forEach(b => b.classList.remove('active'));
        document.querySelector('[data-preset="summer_heat"]').classList.add('active');
        fetchPrediction(gatherFormData());
    });

    btnRunForecast.addEventListener('click', () => {
        historicalBanner.style.display = 'none';
        fetchPrediction(gatherFormData());
    });

    // Initial run
    updateSliderLabels();
    fetchPrediction(gatherFormData());
});
