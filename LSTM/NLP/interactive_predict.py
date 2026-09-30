"""
=============================================================================
Interactive Weather Prediction Terminal for Deep BiLSTM Neural Network
Model: Deep Stacked BiLSTM + Temporal Attention Neural Network
Target: 2-meter Air Temperature (temperature_2m)
=============================================================================
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn

# =============================================================================
# MODEL ARCHITECTURE DEFINITION
# =============================================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs):
        scores = self.attention_weights(rnn_outputs)
        attn_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attn_weights * rnn_outputs, dim=1)
        return context_vector, attn_weights


class WeatherLSTMNeuralNetwork(nn.Module):
    def __init__(self, input_dim=47, hidden_dim=96, num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        lstm_out_dim = hidden_dim * self.num_directions
        self.attention = TemporalAttention(hidden_dim=lstm_out_dim)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        proj = self.input_layer(x)
        lstm_out, _ = self.lstm(proj)
        context, attn_weights = self.attention(lstm_out)
        out = self.mlp_head(context)
        return out


# =============================================================================
# INFERENCE ENGINE & ARTIFACT LOADER
# =============================================================================
class WeatherPredictor:
    def __init__(self, model_dir="NLP"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_path = os.path.join(model_dir, "lstm_best_model.pt")
        self.scaler_path = os.path.join(model_dir, "lstm_scalers.joblib")
        self.meta_path = os.path.join(model_dir, "lstm_metadata.json")
        self.csv_path = os.path.join(model_dir, "bengaluru_preprocessed.csv")
        
        # Fallback to local root if NLP folder prefix not found
        if not os.path.exists(self.model_path) and os.path.exists("lstm_best_model.pt"):
            self.model_path = "lstm_best_model.pt"
            self.scaler_path = "lstm_scalers.joblib"
            self.meta_path = "lstm_metadata.json"
            self.csv_path = "bengaluru_preprocessed.csv"
            
        self._load_artifacts()
        self._load_dataset()

    def _load_artifacts(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Model artifacts not found in '{self.model_dir}'. Please train the model first.")
            
        # Load Scalers
        self.scaler_bundle = joblib.load(self.scaler_path)
        self.feature_scaler = self.scaler_bundle['feature_scaler']
        self.target_scaler = self.scaler_bundle['target_scaler']
        self.feature_cols = self.scaler_bundle['feature_cols']
        self.all_feature_names = self.scaler_bundle['all_feature_names']
        self.seq_len = self.scaler_bundle.get('seq_len', 24)
        
        # Load Metadata
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
        # Initialize Neural Network
        self.model = WeatherLSTMNeuralNetwork(
            input_dim=len(self.all_feature_names),
            hidden_dim=96,
            num_layers=2,
            dropout=0.2,
            bidirectional=True
        ).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def _load_dataset(self):
        if os.path.exists(self.csv_path):
            self.df_raw = pd.read_csv(self.csv_path)
            self.df_raw['time'] = pd.to_datetime(self.df_raw['time'])
            self.df_raw = self.df_raw.sort_values('time').reset_index(drop=True)
            self.df_featured = self._engineer_features(self.df_raw)
        else:
            self.df_raw = None
            self.df_featured = None

    def _engineer_features(self, df):
        data = df.copy()
        hour = data['time'].dt.hour
        day_of_year = data['time'].dt.dayofyear
        month = data['time'].dt.month
        
        data['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        data['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        data['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        data['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        data['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12.0)
        data['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12.0)
        
        if 'wind_speed_10m' in data.columns and 'wind_direction_10m' in data.columns:
            rad_10 = np.radians(data['wind_direction_10m'])
            data['wind_u_10m'] = -data['wind_speed_10m'] * np.sin(rad_10)
            data['wind_v_10m'] = -data['wind_speed_10m'] * np.cos(rad_10)
            
        if 'wind_speed_100m' in data.columns and 'wind_direction_100m' in data.columns:
            rad_100 = np.radians(data['wind_direction_100m'])
            data['wind_u_100m'] = -data['wind_speed_100m'] * np.sin(rad_100)
            data['wind_v_100m'] = -data['wind_speed_100m'] * np.cos(rad_100)

        if 'dew_point_2m' in data.columns and 'temperature_2m' in data.columns:
            data['dew_point_depression'] = data['temperature_2m'] - data['dew_point_2m']
            
        if 'soil_temperature_0_to_7cm' in data.columns and 'soil_temperature_100_to_255cm' in data.columns:
            data['soil_thermal_gradient'] = data['soil_temperature_0_to_7cm'] - data['soil_temperature_100_to_255cm']

        if 'surface_pressure' in data.columns and 'pressure_msl' in data.columns:
            data['pressure_deficit'] = data['pressure_msl'] - data['surface_pressure']

        cols_to_drop = ['season', 'day_name', 'year', 'day_of_week']
        for c in cols_to_drop:
            if c in data.columns:
                data = data.drop(columns=[c])
        return data

    def predict_from_dataframe_window(self, df_window):
        """
        Runs neural network inference on a 24-row DataFrame window.
        Returns predicted temperature in degrees Celsius.
        """
        target_col = self.all_feature_names[0]
        feature_cols = self.all_feature_names[1:]
        
        target_scaled = self.target_scaler.transform(df_window[[target_col]])
        features_scaled = self.feature_scaler.transform(df_window[feature_cols])
        
        matrix = np.hstack([target_scaled, features_scaled])
        tensor_in = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.model(tensor_in).cpu().numpy()
            pred_celsius = self.target_scaler.inverse_transform(pred_scaled).item()
            
        return pred_celsius

    def multi_step_forecast(self, df_window, hours_ahead=6):
        """
        Auto-regressive multi-step forecast for 1 to N hours into the future.
        """
        current_window = df_window[self.all_feature_names].copy().reset_index(drop=True)
        forecasts = []
        
        for step in range(1, hours_ahead + 1):
            next_temp = self.predict_from_dataframe_window(current_window)
            forecasts.append(next_temp)
            
            # Roll window forward by 1 step
            new_row = current_window.iloc[-1:].copy()
            new_row['temperature_2m'] = next_temp
            # Advance cyclical time by 1 hour
            curr_h = (int(new_row['hour'].values[0]) + 1) % 24
            new_row['hour'] = curr_h
            new_row['hour_sin'] = np.sin(2 * np.pi * curr_h / 24.0)
            new_row['hour_cos'] = np.cos(2 * np.pi * curr_h / 24.0)
            
            current_window = pd.concat([current_window.iloc[1:], new_row], ignore_index=True)
            
        return forecasts


# =============================================================================
# INTERACTIVE CLI USER INTERFACE
# =============================================================================
def print_banner():
    print("""
=============================================================================
*   BENGALURU WEATHER FORECASTING: INTERACTIVE LSTM NEURAL NETWORK PREDICTOR *
*   Architecture: Deep Stacked BiLSTM + Temporal Attention (24-Hour Window)  *
=============================================================================
    """)

def prompt_float(prompt_text, default_val, min_val=None, max_val=None):
    while True:
        user_input = input(f"{prompt_text} [Default: {default_val}]: ").strip()
        if not user_input:
            return default_val
        try:
            val = float(user_input)
            if min_val is not None and val < min_val:
                print(f"  [!] Value must be >= {min_val}. Please re-enter.")
                continue
            if max_val is not None and val > max_val:
                print(f"  [!] Value must be <= {max_val}. Please re-enter.")
                continue
            return val
        except ValueError:
            print("  [!] Invalid number. Please enter a numeric value.")

def prompt_int(prompt_text, default_val, min_val=None, max_val=None):
    while True:
        user_input = input(f"{prompt_text} [Default: {default_val}]: ").strip()
        if not user_input:
            return default_val
        try:
            val = int(user_input)
            if min_val is not None and val < min_val:
                print(f"  [!] Value must be >= {min_val}. Please re-enter.")
                continue
            if max_val is not None and val > max_val:
                print(f"  [!] Value must be <= {max_val}. Please re-enter.")
                continue
            return val
        except ValueError:
            print("  [!] Invalid integer. Please enter an integer.")

def mode_historical_date_query(predictor):
    print("\n" + "=" * 70)
    print(" MODE 1: HISTORICAL DATE & TIME QUERY (WITH GROUND TRUTH VERIFICATION)")
    print("=" * 70)
    
    if predictor.df_raw is None:
        print("[!] Dataset not available for historical lookup.")
        return
        
    min_date = predictor.df_raw['time'].min()
    max_date = predictor.df_raw['time'].max()
    print(f"--> Available Date Range: {min_date} to {max_date}")
    print("Examples:")
    print("  - 2024-05-15 14:00:00 (Hot Summer Afternoon)")
    print("  - 2024-07-22 08:00:00 (Monsoon Morning)")
    print("  - 2025-01-10 06:00:00 (Winter Dawn)")
    print("  - Type 'RANDOM' to select a random point from the test set")
    
    choice = input("\nEnter target timestamp (YYYY-MM-DD HH:MM) or 'RANDOM' [Default: RANDOM]: ").strip()
    
    if not choice or choice.upper() == 'RANDOM':
        # Select random index from test range
        test_start_idx = int(len(predictor.df_featured) * 0.9)
        idx = np.random.randint(test_start_idx + 24, len(predictor.df_featured) - 25)
    else:
        try:
            target_dt = pd.to_datetime(choice)
            matches = predictor.df_raw.index[predictor.df_raw['time'] == target_dt].tolist()
            if not matches:
                print(f"[!] Timestamp '{target_dt}' not found in dataset. Using nearest matching timestamp.")
                time_diffs = (predictor.df_raw['time'] - target_dt).abs()
                idx = time_diffs.idxmin()
            else:
                idx = matches[0]
                
            if idx < 24:
                print("[!] Target timestamp must have at least 24 hours of prior history. Setting idx = 24.")
                idx = 24
        except Exception as e:
            print(f"[!] Error parsing date: {e}. Falling back to random test sample.")
            idx = np.random.randint(int(len(predictor.df_featured) * 0.9) + 24, len(predictor.df_featured) - 25)

    # Extract 24-hour lookback window
    window_df = predictor.df_featured.iloc[idx - 24:idx]
    actual_next_temp = predictor.df_featured.iloc[idx]['temperature_2m']
    target_timestamp = predictor.df_raw.iloc[idx]['time']
    window_start_time = predictor.df_raw.iloc[idx - 24]['time']
    window_end_time = predictor.df_raw.iloc[idx - 1]['time']
    
    # Run Prediction
    pred_temp = predictor.predict_from_dataframe_window(window_df)
    abs_error = abs(actual_next_temp - pred_temp)
    accuracy_pct = max(0, 100 - (abs_error / max(1.0, abs(actual_next_temp)) * 100))
    
    print("\n" + "-" * 70)
    print(f" Lookback Window : {window_start_time} to {window_end_time} (24 Hours)")
    print(f" Target Timestamp: {target_timestamp}")
    print(f" Past 24h Mean T : {window_df['temperature_2m'].mean():.2f} deg C (Min: {window_df['temperature_2m'].min():.2f} deg C, Max: {window_df['temperature_2m'].max():.2f} deg C)")
    print(f" Past 24h Mean RH: {window_df['relative_humidity_2m'].mean():.1f} %")
    print(f" Past 24h Mean P : {window_df['surface_pressure'].mean():.1f} hPa")
    print("-" * 70)
    print(f" [Ground Truth Actual] : {actual_next_temp:.2f} deg C")
    print(f" [LSTM Model Forecast] : {pred_temp:.2f} deg C")
    print(f" [Absolute Error]      : {abs_error:.3f} deg C")
    print(f" [Prediction Accuracy] : {accuracy_pct:.2f} %")
    print("-" * 70)
    
    # Multi-Step Rollout Option
    multi_choice = input("\nWould you like a multi-hour rollout forecast (1 to 12 hours ahead)? (y/n) [Default: y]: ").strip().lower()
    if multi_choice != 'n':
        hours = prompt_int("How many hours ahead to forecast (1-24)?", default_val=6, min_val=1, max_val=24)
        rollout = predictor.multi_step_forecast(window_df, hours_ahead=hours)
        print("\n" + "=" * 70)
        print(f"{'STEP':<8} | {'FORECAST TIME':<22} | {'PREDICTED TEMP':<18} | {'GROUND TRUTH':<15}")
        print("=" * 70)
        for step_i, temp_val in enumerate(rollout, start=1):
            f_time = target_timestamp + pd.Timedelta(hours=step_i - 1)
            gt_val = "N/A"
            if idx + step_i - 1 < len(predictor.df_raw):
                gt_temp = predictor.df_raw.iloc[idx + step_i - 1]['temperature_2m']
                gt_val = f"{gt_temp:.2f} deg C"
            print(f"T+{step_i:<5} | {str(f_time):<22} | {temp_val:<18.2f} deg C | {gt_val:<15}")
        print("=" * 70)

def mode_custom_user_inputs(predictor):
    print("\n" + "=" * 70)
    print(" MODE 2: CUSTOM WEATHER SENSOR ENTRY & LIVE INFERENCE")
    print("=" * 70)
    print("Enter the current atmospheric readings. Default values represent realistic Bengaluru conditions.\n")
    
    current_temp = prompt_float("1. Current Air Temperature (deg C)", default_val=26.5, min_val=5.0, max_val=48.0)
    humidity = prompt_float("2. Relative Humidity (%)", default_val=65.0, min_val=5.0, max_val=100.0)
    pressure = prompt_float("3. Surface Pressure (hPa)", default_val=915.0, min_val=850.0, max_val=1050.0)
    wind_speed = prompt_float("4. Wind Speed at 10m (km/h)", default_val=12.0, min_val=0.0, max_val=120.0)
    wind_dir = prompt_float("5. Wind Direction (0-360 degrees)", default_val=240.0, min_val=0.0, max_val=360.0)
    cloud_cover = prompt_float("6. Total Cloud Cover (%)", default_val=40.0, min_val=0.0, max_val=100.0)
    precip = prompt_float("7. Precipitation (mm)", default_val=0.0, min_val=0.0, max_val=200.0)
    hour = prompt_int("8. Current Hour of Day (0 to 23)", default_val=14, min_val=0, max_val=23)
    month = prompt_int("9. Current Month (1 to 12)", default_val=5, min_val=1, max_val=12)
    
    # Atmospheric approximations for missing sub-measurements
    dew_point = current_temp - ((100.0 - humidity) / 5.0)  # Magnus-Tetens approximation
    apparent_temp = current_temp + 0.33 * (humidity / 100.0 * 6.105 * np.exp(17.27 * current_temp / (237.7 + current_temp))) - 0.70 * (wind_speed / 3.6) - 4.0
    pressure_msl = pressure + 105.0  # ~900m altitude adjustment for Bengaluru
    vpd = (1.0 - humidity / 100.0) * (0.61078 * np.exp(17.27 * current_temp / (current_temp + 237.3)))
    
    # Construct a realistic 24-hour synthetic historical trajectory using diurnal sinusoidal curve
    hours_seq = [(hour - 23 + i) % 24 for i in range(24)]
    temp_profile = [current_temp - 4.0 * np.cos(2 * np.pi * (h - 14) / 24.0) for h in hours_seq]
    temp_profile[-1] = current_temp  # Ensure exact final value matches user input
    
    # Build 24-row DataFrame
    rows = []
    for step_i, h_val in enumerate(hours_seq):
        t_val = temp_profile[step_i]
        rh_val = np.clip(humidity + 15.0 * np.cos(2 * np.pi * (h_val - 14) / 24.0), 10, 100)
        dp_val = t_val - ((100.0 - rh_val) / 5.0)
        
        row_dict = {
            'temperature_2m': t_val,
            'relative_humidity_2m': rh_val,
            'apparent_temperature': t_val - 0.5,
            'dew_point_2m': dp_val,
            'precipitation': precip if step_i >= 20 else 0.0,
            'rain': precip if step_i >= 20 else 0.0,
            'weather_code': 1 if precip == 0 else 51,
            'pressure_msl': pressure_msl,
            'surface_pressure': pressure,
            'cloud_cover': cloud_cover,
            'cloud_cover_low': cloud_cover * 0.4,
            'cloud_cover_mid': cloud_cover * 0.3,
            'cloud_cover_high': cloud_cover * 0.3,
            'et0_fao_evapotranspiration': max(0.05, 0.3 * np.sin(np.pi * max(0, h_val - 6) / 12.0)),
            'vapour_pressure_deficit': max(0.1, vpd),
            'wind_speed_10m': wind_speed,
            'wind_speed_100m': wind_speed * 1.3,
            'wind_direction_10m': wind_dir,
            'wind_direction_100m': wind_dir,
            'wind_gusts_10m': wind_speed * 1.5,
            'soil_temperature_0_to_7cm': t_val + 1.2,
            'soil_temperature_7_to_28cm': t_val + 0.8,
            'soil_temperature_28_to_100cm': t_val + 0.2,
            'soil_temperature_100_to_255cm': 24.5,
            'soil_moisture_28_to_100cm': 0.25,
            'soil_moisture_0_to_7cm': 0.22,
            'soil_moisture_7_to_28cm': 0.24,
            'soil_moisture_100_to_255cm': 0.28,
            'month': month,
            'day': 15,
            'hour': h_val,
            'is_weekend': 0,
            'quarter': (month - 1) // 3 + 1,
            'log_precipitation': np.log1p(precip if step_i >= 20 else 0.0),
            'hour_sin': np.sin(2 * np.pi * h_val / 24.0),
            'hour_cos': np.cos(2 * np.pi * h_val / 24.0),
            'doy_sin': np.sin(2 * np.pi * (month * 30) / 365.25),
            'doy_cos': np.cos(2 * np.pi * (month * 30) / 365.25),
            'month_sin': np.sin(2 * np.pi * (month - 1) / 12.0),
            'month_cos': np.cos(2 * np.pi * (month - 1) / 12.0),
            'wind_u_10m': -wind_speed * np.sin(np.radians(wind_dir)),
            'wind_v_10m': -wind_speed * np.cos(np.radians(wind_dir)),
            'wind_u_100m': -(wind_speed * 1.3) * np.sin(np.radians(wind_dir)),
            'wind_v_100m': -(wind_speed * 1.3) * np.cos(np.radians(wind_dir)),
            'dew_point_depression': t_val - dp_val,
            'soil_thermal_gradient': (t_val + 1.2) - 24.5,
            'pressure_deficit': pressure_msl - pressure
        }
        rows.append(row_dict)
        
    custom_df = pd.DataFrame(rows)
    pred_next = predictor.predict_from_dataframe_window(custom_df)
    
    print("\n" + "=" * 70)
    print(f" [Current Timestamp]    : Hour {hour:02d}:00 (Month {month})")
    print(f" [Current Input Temp]   : {current_temp:.2f} deg C")
    print(f" [Predicted Temp at T+1]: {pred_next:.2f} deg C (Delta: {pred_next - current_temp:+.2f} deg C)")
    print("=" * 70)
    
    # 6-Hour Forecast Rollout
    rollout = predictor.multi_step_forecast(custom_df, hours_ahead=6)
    print("\n--- 6-Hour Hourly Weather Trajectory Forecast ---")
    for step_i, temp_val in enumerate(rollout, start=1):
        target_h = (hour + step_i) % 24
        print(f"  Hour {target_h:02d}:00 (T+{step_i}h): {temp_val:.2f} deg C")
    print("=" * 70)

def mode_preset_scenarios(predictor):
    print("\n" + "=" * 70)
    print(" MODE 3: PRESET METEOROLOGICAL SCENARIOS")
    print("=" * 70)
    print("1. Summer Peak Afternoon (35.5 deg C, 22% RH, Clear Sky, Month 5, Hour 14)")
    print("2. Monsoon Torrential Rain (21.0 deg C, 94% RH, Overcast, Rain, Month 7, Hour 17)")
    print("3. Crisp Winter Dawn (13.5 deg C, 88% RH, Calm Winds, Month 1, Hour 06)")
    print("4. Pre-Monsoon Sudden Storm (28.0 deg C, 70% RH, High Wind Gusts, Month 4, Hour 16)")
    
    choice = prompt_int("\nSelect Preset Scenario (1-4)", default_val=1, min_val=1, max_val=4)
    
    presets = {
        1: {'name': 'Summer Peak Afternoon', 't': 35.5, 'rh': 22.0, 'p': 912.0, 'ws': 14.0, 'cc': 15.0, 'pr': 0.0, 'h': 14, 'm': 5},
        2: {'name': 'Monsoon Torrential Rain', 't': 21.0, 'rh': 94.0, 'p': 908.0, 'ws': 22.0, 'cc': 100.0, 'pr': 15.0, 'h': 17, 'm': 7},
        3: {'name': 'Crisp Winter Dawn', 't': 13.5, 'rh': 88.0, 'p': 920.0, 'ws': 5.0, 'cc': 5.0, 'pr': 0.0, 'h': 6, 'm': 1},
        4: {'name': 'Pre-Monsoon Sudden Storm', 't': 28.0, 'rh': 70.0, 'p': 905.0, 'ws': 38.0, 'cc': 85.0, 'pr': 8.5, 'h': 16, 'm': 4},
    }
    
    p = presets[choice]
    print(f"\n[Running Simulation for: {p['name']}]")
    
    # Build synthetic 24h trajectory
    hours_seq = [(p['h'] - 23 + i) % 24 for i in range(24)]
    temp_profile = [p['t'] - 5.0 * np.cos(2 * np.pi * (h - 14) / 24.0) for h in hours_seq]
    temp_profile[-1] = p['t']
    
    rows = []
    for step_i, h_val in enumerate(hours_seq):
        t_val = temp_profile[step_i]
        rh_val = np.clip(p['rh'] + 15.0 * np.cos(2 * np.pi * (h_val - 14) / 24.0), 10, 100)
        dp_val = t_val - ((100.0 - rh_val) / 5.0)
        
        row_dict = {
            'temperature_2m': t_val,
            'relative_humidity_2m': rh_val,
            'apparent_temperature': t_val - 0.5,
            'dew_point_2m': dp_val,
            'precipitation': p['pr'] if step_i >= 20 else 0.0,
            'rain': p['pr'] if step_i >= 20 else 0.0,
            'weather_code': 1 if p['pr'] == 0 else 51,
            'pressure_msl': p['p'] + 105.0,
            'surface_pressure': p['p'],
            'cloud_cover': p['cc'],
            'cloud_cover_low': p['cc'] * 0.4,
            'cloud_cover_mid': p['cc'] * 0.3,
            'cloud_cover_high': p['cc'] * 0.3,
            'et0_fao_evapotranspiration': max(0.05, 0.3 * np.sin(np.pi * max(0, h_val - 6) / 12.0)),
            'vapour_pressure_deficit': 1.2,
            'wind_speed_10m': p['ws'],
            'wind_speed_100m': p['ws'] * 1.3,
            'wind_direction_10m': 220.0,
            'wind_direction_100m': 220.0,
            'wind_gusts_10m': p['ws'] * 1.5,
            'soil_temperature_0_to_7cm': t_val + 1.0,
            'soil_temperature_7_to_28cm': t_val + 0.5,
            'soil_temperature_28_to_100cm': t_val,
            'soil_temperature_100_to_255cm': 24.0,
            'soil_moisture_28_to_100cm': 0.25,
            'soil_moisture_0_to_7cm': 0.22,
            'soil_moisture_7_to_28cm': 0.24,
            'soil_moisture_100_to_255cm': 0.28,
            'month': p['m'],
            'day': 15,
            'hour': h_val,
            'is_weekend': 0,
            'quarter': (p['m'] - 1) // 3 + 1,
            'log_precipitation': np.log1p(p['pr'] if step_i >= 20 else 0.0),
            'hour_sin': np.sin(2 * np.pi * h_val / 24.0),
            'hour_cos': np.cos(2 * np.pi * h_val / 24.0),
            'doy_sin': np.sin(2 * np.pi * (p['m'] * 30) / 365.25),
            'doy_cos': np.cos(2 * np.pi * (p['m'] * 30) / 365.25),
            'month_sin': np.sin(2 * np.pi * (p['m'] - 1) / 12.0),
            'month_cos': np.cos(2 * np.pi * (p['m'] - 1) / 12.0),
            'wind_u_10m': -p['ws'] * np.sin(np.radians(220.0)),
            'wind_v_10m': -p['ws'] * np.cos(np.radians(220.0)),
            'wind_u_100m': -(p['ws'] * 1.3) * np.sin(np.radians(220.0)),
            'wind_v_100m': -(p['ws'] * 1.3) * np.cos(np.radians(220.0)),
            'dew_point_depression': t_val - dp_val,
            'soil_thermal_gradient': (t_val + 1.0) - 24.0,
            'pressure_deficit': 105.0
        }
        rows.append(row_dict)
        
    df_sim = pd.DataFrame(rows)
    pred_next = predictor.predict_from_dataframe_window(df_sim)
    
    print("\n" + "-" * 70)
    print(f" Scenario Name          : {p['name']}")
    print(f" Current Temperature    : {p['t']:.2f} deg C (Hour {p['h']:02d}:00)")
    print(f" LSTM Next-Hour Forecast: {pred_next:.2f} deg C (Change: {pred_next - p['t']:+.2f} deg C)")
    print("-" * 70)
    
    rollout = predictor.multi_step_forecast(df_sim, hours_ahead=8)
    print("\n--- 8-Hour Forward Evolution ---")
    for step_i, temp_val in enumerate(rollout, start=1):
        target_h = (p['h'] + step_i) % 24
        print(f"  Hour {target_h:02d}:00 (+{step_i}h): {temp_val:.2f} deg C")
    print("-" * 70)

def main():
    print_banner()
    print("--> Initializing Deep LSTM Neural Network & Loading Scaler Pipelines...")
    
    predictor = WeatherPredictor(model_dir="NLP")
    print("--> Model and scalers loaded successfully into memory!\n")
    
    while True:
        print("\n" + "=" * 70)
        print(" SELECT INTERACTIVE MODE:")
        print("=" * 70)
        print(" 1. Query Historical Date & Verify with Ground Truth")
        print(" 2. Enter Custom Weather Sensor Readings (Live Simulator)")
        print(" 3. Run Preset Meteorological Scenarios (Summer, Monsoon, Winter, Storm)")
        print(" 4. Exit")
        print("=" * 70)
        
        choice = prompt_int("Enter selection (1-4)", default_val=1, min_val=1, max_val=4)
        
        if choice == 1:
            mode_historical_date_query(predictor)
        elif choice == 2:
            mode_custom_user_inputs(predictor)
        elif choice == 3:
            mode_preset_scenarios(predictor)
        elif choice == 4:
            print("\nExiting Weather Predictor. Have a great day!")
            break
            
        cont = input("\nWould you like to run another prediction? (y/n) [Default: y]: ").strip().lower()
        if cont == 'n':
            print("\nExiting Weather Predictor. Have a great day!")
            break


if __name__ == "__main__":
    main()
