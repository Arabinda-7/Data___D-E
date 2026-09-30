"""
=============================================================================
WeatherAI - Flask Web Application for XGBoost Temperature Prediction
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
MODEL_PATH = os.path.join(BASE_DIR, 'xgboost_weather_model.joblib')
META_PATH = os.path.join(BASE_DIR, 'xgboost_model_metadata.json')
DEFAULTS_PATH = os.path.join(BASE_DIR, 'feature_defaults.json')
DATASET_PATH = os.path.join(BASE_DIR, 'bengaluru_preprocessed.csv')

# Global cached variables
model = None
metadata = {}
feature_cols = []
feature_defaults = {}
sample_df = None


def initialize_app_artifacts():
    """Load model, metadata, and feature baselines into memory."""
    global model, metadata, feature_cols, feature_defaults, sample_df
    
    print("Loading XGBoost Model and Metadata...")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("[+] Model loaded successfully from joblib.")
    else:
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            feature_cols = metadata.get('features', [])
            print(f"[+] Metadata loaded ({len(feature_cols)} features registered).")
            
    if os.path.exists(DEFAULTS_PATH):
        with open(DEFAULTS_PATH, 'r', encoding='utf-8') as f:
            feature_defaults = json.load(f)
            print("[+] Feature defaults loaded.")
    else:
        feature_defaults = {}
        
    # Pre-cache a lightweight slice of preprocessed dataset for fast sample lookup
    if os.path.exists(DATASET_PATH):
        try:
            df_full = pd.read_csv(DATASET_PATH)
            sample_df = df_full.sample(n=min(2000, len(df_full)), random_state=42).reset_index(drop=True)
            print(f"[+] Cached sample pool with {len(sample_df)} real observations.")
        except Exception as e:
            print(f"! Warning: Could not pre-cache sample dataset: {e}")
            sample_df = None


initialize_app_artifacts()


def compute_prediction_for_input(user_dict):
    """
    Transforms user inputs into complete model features and predicts temperature.
    """
    row = feature_defaults.copy()
    
    hour = int(user_dict.get('hour', 12))
    month = int(user_dict.get('month', 6))
    day = int(user_dict.get('day', 15))
    year = int(user_dict.get('year', 2026))
    season = str(user_dict.get('season', 'Summer'))
    day_name = str(user_dict.get('day_name', 'Monday'))
    
    # 1. Update basic meteorological inputs
    keys_map = [
        'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'cloud_cover',
        'precipitation', 'rain', 'weather_code', 'pressure_msl', 'wind_speed_100m',
        'wind_direction_10m', 'wind_direction_100m', 'wind_gusts_10m',
        'soil_temperature_7_to_28cm', 'soil_temperature_28_to_100cm', 'soil_temperature_100_to_255cm',
        'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm',
        'et0_fao_evapotranspiration', 'vapour_pressure_deficit'
    ]
    for k in keys_map:
        if k in user_dict and user_dict[k] is not None:
            try:
                row[k] = float(user_dict[k])
            except (ValueError, TypeError):
                pass
                
    # 2. Temperature Inertia / Lags
    temp_lag_1h = float(user_dict.get('temperature_2m_lag_1h', user_dict.get('temp_1h_ago', 25.0)))
    temp_lag_24h = float(user_dict.get('temperature_2m_lag_24h', user_dict.get('temp_yesterday', temp_lag_1h)))
    temp_rolling_mean = float(user_dict.get('temp_rolling_mean_24h', temp_lag_1h))
    temp_rolling_std = float(user_dict.get('temp_rolling_std_24h', 3.0))
    
    row['temperature_2m_lag_1h'] = temp_lag_1h
    row['temperature_2m_lag_24h'] = temp_lag_24h
    row['temp_rolling_mean_24h'] = temp_rolling_mean
    row['temp_rolling_std_24h'] = temp_rolling_std
    
    # Lag variables for other features
    hum = row.get('relative_humidity_2m', 60.0)
    row['relative_humidity_2m_lag_1h'] = float(user_dict.get('relative_humidity_2m_lag_1h', hum))
    row['relative_humidity_2m_lag_24h'] = float(user_dict.get('relative_humidity_2m_lag_24h', hum))
    row['humidity_rolling_mean_24h'] = float(user_dict.get('humidity_rolling_mean_24h', hum))
    
    pres = row.get('surface_pressure', 915.0)
    row['surface_pressure_lag_1h'] = float(user_dict.get('surface_pressure_lag_1h', pres))
    row['surface_pressure_lag_24h'] = float(user_dict.get('surface_pressure_lag_24h', pres))
    
    wspd = row.get('wind_speed_10m', 10.0)
    row['wind_speed_10m_lag_1h'] = float(user_dict.get('wind_speed_10m_lag_1h', wspd))
    row['wind_speed_10m_lag_24h'] = float(user_dict.get('wind_speed_10m_lag_24h', wspd))
    
    # 3. Cyclical Temporal Calculations
    row['hour'] = hour
    row['month'] = month
    row['day'] = day
    row['year'] = year
    row['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    row['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    row['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    row['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    
    doy = int((month - 1) * 30.4 + day)
    row['day_of_year'] = doy
    row['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    row['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    
    row['day_of_week'] = int(user_dict.get('day_of_week', 2))
    row['is_weekend'] = 1.0 if day_name in ['Saturday', 'Sunday'] else 0.0
    row['quarter'] = int((month - 1) // 3 + 1)
    
    precip = float(row.get('precipitation', 0.0))
    row['log_precipitation'] = np.log1p(precip)
    
    # 4. One-Hot Categorical Flags
    for s in ['Post_Monsoon', 'Summer', 'Winter']:
        row[f'season_{s}'] = 1.0 if season.lower() == s.lower() else 0.0
    for d in ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']:
        row[f'day_name_{d}'] = 1.0 if day_name.lower() == d.lower() else 0.0
        
    # Align features exactly as trained
    input_df = pd.DataFrame([row])[feature_cols]
    pred_temp_c = float(model.predict(input_df)[0])
    pred_temp_f = pred_temp_c * 9/5 + 32
    
    # Calculate Heat Index / Apparent Feels Like (Steadman & Rothfusz formula approximation)
    humidity_val = row.get('relative_humidity_2m', 60.0)
    wind_val = row.get('wind_speed_10m', 10.0)
    
    if pred_temp_c >= 26:
        hi = pred_temp_c + (0.5555 * (6.11 * np.exp(5417.7530 * (1/273.16 - 1/(273.15 + pred_temp_c))) * (humidity_val/100) - 10))
        feels_like_c = 0.6 * pred_temp_c + 0.4 * hi
    elif pred_temp_c <= 18:
        feels_like_c = 13.12 + 0.6215 * pred_temp_c - 11.37 * (wind_val**0.16) + 0.3965 * pred_temp_c * (wind_val**0.16)
    else:
        feels_like_c = pred_temp_c - 0.5 * (wind_val / 25.0) + 0.5 * ((humidity_val - 50) / 50.0)
        
    feels_like_c = round(float(feels_like_c), 2)
    feels_like_f = round(feels_like_c * 9/5 + 32, 2)
    
    # Comfort Classification & Visual Metadata
    if pred_temp_c < 15:
        comfort = "Chilly & Crisp"
        theme = "cold"
        icon = "❄️"
        advice = "Wear a warm jacket or sweater. Early morning chill expected."
    elif pred_temp_c < 22:
        comfort = "Cool & Pleasant"
        theme = "pleasant"
        icon = "🍃"
        advice = "Ideal pleasant weather. Great for outdoor walks and sports."
    elif pred_temp_c < 28:
        comfort = "Warm & Comfortable"
        theme = "warm"
        icon = "☀️"
        advice = "Moderate comfortable conditions with clear daylight."
    elif pred_temp_c < 34:
        comfort = "Hot & Sunny"
        theme = "hot"
        icon = "🔥"
        advice = "Warm afternoon conditions. Stay hydrated and seek shade."
    else:
        comfort = "Extreme Heatwave"
        theme = "extreme"
        icon = "🚨"
        advice = "Intense heat peak. Avoid direct midday sun and drink plenty of water."
        
    # Generate 24-Hour Diurnal Forecast Simulation
    diurnal_curve = []
    base_row = row.copy()
    for h in range(24):
        h_row = base_row.copy()
        h_row['hour'] = h
        h_row['hour_sin'] = np.sin(2 * np.pi * h / 24.0)
        h_row['hour_cos'] = np.cos(2 * np.pi * h / 24.0)
        solar_factor = np.sin((h - 5) * np.pi / 14) if 6 <= h <= 18 else -0.3
        h_row['temperature_2m_lag_1h'] = temp_rolling_mean + solar_factor * 4.0
        
        h_df = pd.DataFrame([h_row])[feature_cols]
        h_pred = float(model.predict(h_df)[0])
        diurnal_curve.append({
            'hour': f"{h:02d}:00",
            'temp_c': round(h_pred, 2),
            'temp_f': round(h_pred * 9/5 + 32, 2)
        })
        
    return {
        'predicted_temperature_c': round(pred_temp_c, 2),
        'predicted_temperature_f': round(pred_temp_f, 2),
        'feels_like_c': feels_like_c,
        'feels_like_f': feels_like_f,
        'comfort_status': comfort,
        'theme': theme,
        'icon': icon,
        'advice': advice,
        'diurnal_forecast': diurnal_curve,
        'summary_metrics': {
            'relative_humidity': f"{humidity_val:.1f}%",
            'surface_pressure': f"{pres:.1f} hPa",
            'wind_speed': f"{wind_val:.1f} km/h",
            'cloud_cover': f"{row.get('cloud_cover', 20.0):.1f}%",
            'precipitation': f"{precip:.1f} mm",
            'prior_1h_temp': f"{temp_lag_1h:.1f} °C",
            'season': season,
            'time': f"{hour:02d}:00"
        }
    }


# =============================================================================
# API ROUTES
# =============================================================================
@app.route('/')
def index():
    """Render main interactive dashboard."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict temperature from user inputs."""
    try:
        data = request.get_json() or {}
        result = compute_prediction_for_input(data)
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/presets', methods=['GET'])
def api_presets():
    """Return list of weather presets."""
    presets = [
        {
            'id': 'summer_heat',
            'name': '☀️ Summer Heatwave',
            'description': 'Scorching May afternoon (2:00 PM) with dry breeze',
            'data': {
                'hour': 14, 'month': 5, 'season': 'Summer', 'day_name': 'Monday',
                'temp_1h_ago': 33.5, 'relative_humidity_2m': 30.0, 'surface_pressure': 910.0,
                'wind_speed_10m': 16.0, 'cloud_cover': 10.0, 'precipitation': 0.0
            }
        },
        {
            'id': 'monsoon_storm',
            'name': '🌧️ Monsoon Downpour',
            'description': 'Heavy August evening rain (6:00 PM) with high humidity',
            'data': {
                'hour': 18, 'month': 8, 'season': 'Monsoon', 'day_name': 'Friday',
                'temp_1h_ago': 22.0, 'relative_humidity_2m': 92.0, 'surface_pressure': 913.5,
                'wind_speed_10m': 24.0, 'cloud_cover': 95.0, 'precipitation': 18.0
            }
        },
        {
            'id': 'winter_dawn',
            'name': '❄️ Winter Dawn Chill',
            'description': 'Crisp January morning (6:00 AM) with calm wind',
            'data': {
                'hour': 6, 'month': 1, 'season': 'Winter', 'day_name': 'Sunday',
                'temp_1h_ago': 14.5, 'relative_humidity_2m': 75.0, 'surface_pressure': 923.0,
                'wind_speed_10m': 5.0, 'cloud_cover': 5.0, 'precipitation': 0.0
            }
        },
        {
            'id': 'pleasant_evening',
            'name': '🍃 Pleasant Spring Evening',
            'description': 'Comfortable March dusk (7:00 PM) with gentle breeze',
            'data': {
                'hour': 19, 'month': 3, 'season': 'Summer', 'day_name': 'Wednesday',
                'temp_1h_ago': 27.0, 'relative_humidity_2m': 52.0, 'surface_pressure': 916.0,
                'wind_speed_10m': 12.0, 'cloud_cover': 25.0, 'precipitation': 0.0
            }
        },
        {
            'id': 'post_monsoon',
            'name': '🌤️ Post-Monsoon Afternoon',
            'description': 'Clear October midday (1:00 PM) with moderate humidity',
            'data': {
                'hour': 13, 'month': 10, 'season': 'Post_Monsoon', 'day_name': 'Saturday',
                'temp_1h_ago': 28.0, 'relative_humidity_2m': 58.0, 'surface_pressure': 915.0,
                'wind_speed_10m': 9.0, 'cloud_cover': 40.0, 'precipitation': 0.0
            }
        }
    ]
    return jsonify({'status': 'success', 'presets': presets})


@app.route('/api/random-sample', methods=['GET'])
def api_random_sample():
    """Fetch a random historical record from dataset to test accuracy against true value."""
    if sample_df is not None and not sample_df.empty:
        row = sample_df.sample(n=1).iloc[0].to_dict()
        cleaned_row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        actual_temp = cleaned_row.get('temperature_2m')
        
        user_dict = {
            'hour': cleaned_row.get('hour', 12),
            'month': cleaned_row.get('month', 6),
            'day': cleaned_row.get('day', 15),
            'year': cleaned_row.get('year', 2024),
            'season': cleaned_row.get('season', 'Summer'),
            'day_name': cleaned_row.get('day_name', 'Monday'),
            'temp_1h_ago': cleaned_row.get('temperature_2m', 25.0),
            'relative_humidity_2m': cleaned_row.get('relative_humidity_2m', 60.0),
            'surface_pressure': cleaned_row.get('surface_pressure', 915.0),
            'wind_speed_10m': cleaned_row.get('wind_speed_10m', 10.0),
            'cloud_cover': cleaned_row.get('cloud_cover', 20.0),
            'precipitation': cleaned_row.get('precipitation', 0.0)
        }
        
        prediction = compute_prediction_for_input(user_dict)
        error = abs(actual_temp - prediction['predicted_temperature_c']) if actual_temp is not None else None
        
        return jsonify({
            'status': 'success',
            'timestamp': str(cleaned_row.get('time', 'Historical Record')),
            'actual_temperature_c': round(actual_temp, 2) if actual_temp is not None else None,
            'prediction': prediction,
            'absolute_error_c': round(error, 2) if error is not None else None,
            'raw_inputs': user_dict
        })
    else:
        return jsonify({'status': 'error', 'message': 'Historical dataset sample not available'}), 404


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """Return model specifications and evaluation benchmark metrics."""
    return jsonify({
        'status': 'success',
        'model_name': metadata.get('model_name', 'XGBoost Regressor'),
        'target': metadata.get('target', 'temperature_2m'),
        'feature_count': len(feature_cols),
        'metrics': metadata.get('test_metrics', {})
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n==================================================")
    print(f"  WeatherAI Flask Server Running on http://localhost:{port}")
    print(f"==================================================\n")
    app.run(host='0.0.0.0', port=port, debug=True)
