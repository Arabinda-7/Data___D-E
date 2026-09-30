"""
=============================================================================
XGBoost Weather Prediction Model - Training, Testing, Evaluation & Live Inference
Dataset: bengaluru_preprocessed.csv
Target: 2-meter Air Temperature (temperature_2m)
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)

# Configure warnings and aesthetics
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150


# =============================================================================
# 1. DATA LOADING & AUDIT
# =============================================================================
def load_data(file_path='bengaluru_preprocessed.csv'):
    """Load the preprocessed dataset and display basic diagnostics."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset '{file_path}' not found. Please ensure it is in the working directory.")
    
    print("=" * 70)
    print(" STEP 1: LOADING DATASET")
    print("=" * 70)
    df = pd.read_csv(file_path)
    print(f"Loaded dataset successfully from: {file_path}")
    print(f"Initial Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# =============================================================================
# 2. FEATURE ENGINEERING & MISSING PREPROCESSING STEPS
# =============================================================================
def preprocess_and_engineer_features(df, target_col='temperature_2m'):
    """
    1. Parses datetime correctly and sorts chronologically.
    2. Encodes cyclical temporal features (sine/cosine for hour & month & doy).
    3. Generates lag and rolling window features for temporal momentum.
    4. One-Hot encodes remaining categorical columns (season, day_name).
    5. Cleans and structures feature matrices without lookahead data leakage.
    """
    print("\n" + "=" * 70)
    print(" STEP 2: ADVANCED FEATURE ENGINEERING & PREPROCESSING AUDIT")
    print("=" * 70)
    
    data = df.copy()
    
    # 2.1 Datetime parsing and chronological sorting
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values('time').reset_index(drop=True)
    print(f"Time series span: {data['time'].min()} to {data['time'].max()}")
    
    # 2.2 Cyclical Temporal Features (Sine & Cosine transformations)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24.0)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24.0)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12.0)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12.0)
    data['day_of_year'] = data['time'].dt.dayofyear
    data['doy_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
    data['doy_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
    print("Added cyclical trigonometric features: hour_sin/cos, month_sin/cos, doy_sin/cos")
    
    # 2.3 Lag features for key meteorological variables (captures thermal inertia)
    lag_cols = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m']
    lag_cols = [c for c in lag_cols if c in data.columns]
    
    for col in lag_cols:
        data[f'{col}_lag_1h'] = data[col].shift(1)
        data[f'{col}_lag_24h'] = data[col].shift(24)
        
    # 2.4 Rolling statistical aggregations (past 24h rolling mean and std)
    data['temp_rolling_mean_24h'] = data['temperature_2m'].shift(1).rolling(window=24).mean()
    data['temp_rolling_std_24h'] = data['temperature_2m'].shift(1).rolling(window=24).std()
    data['humidity_rolling_mean_24h'] = data['relative_humidity_2m'].shift(1).rolling(window=24).mean()
    
    print("Added temporal lag features (1h, 24h) and 24-hour rolling statistics")
    
    # Drop rows with NaN resulting from lag / rolling operations
    initial_rows = len(data)
    data = data.dropna().reset_index(drop=True)
    print(f"Dropped {initial_rows - len(data)} rows with warm-up NaNs from lag features.")
    
    # 2.5 Categorical One-Hot Encoding
    categorical_cols = [c for c in ['season', 'day_name'] if c in data.columns]
    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=float)
        print(f"One-hot encoded categorical variables: {categorical_cols}")
        
    return data


# =============================================================================
# 3. TIME-SERIES TRAIN-VALIDATION-TEST SPLIT
# =============================================================================
def split_data(data, target_col='temperature_2m', train_ratio=0.8, val_ratio=0.1):
    """
    Performs a strict chronological split (Time-Series aware) to prevent
    lookahead data leakage.
    """
    print("\n" + "=" * 70)
    print(" STEP 3: TIME-SERIES DATA SPLITTING (CHRONOLOGICAL)")
    print("=" * 70)
    
    # Drop non-feature columns
    drop_cols = ['time', target_col]
    
    # Prevent direct target leakage if highly collinear same-hour derivatives exist
    leaking_cols = ['apparent_temperature', 'dew_point_2m', 'soil_temperature_0_to_7cm']
    drop_cols.extend([c for c in leaking_cols if c in data.columns and c != target_col])
    
    feature_cols = [c for c in data.columns if c not in drop_cols]
    
    X = data[feature_cols]
    y = data[target_col]
    timestamps = data['time']
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    time_train = (timestamps.iloc[0], timestamps.iloc[train_end - 1])
    time_val = (timestamps.iloc[train_end], timestamps.iloc[val_end - 1])
    time_test = (timestamps.iloc[val_end], timestamps.iloc[-1])
    
    feature_defaults = X_train.median().to_dict()
    
    print(f"Total Samples     : {n:,}")
    print(f"Training Set (80%): {len(X_train):,} samples | Period: {time_train[0].date()} -> {time_train[1].date()}")
    print(f"Validation Set(10%): {len(X_val):,} samples | Period: {time_val[0].date()} -> {time_val[1].date()}")
    print(f"Test Set (10%)    : {len(X_test):,} samples | Period: {time_test[0].date()} -> {time_test[1].date()}")
    print(f"Number of Features: {X.shape[1]}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, timestamps.iloc[val_end:].values, feature_cols, feature_defaults


# =============================================================================
# 4. MODEL INITIALIZATION & TRAINING
# =============================================================================
def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Initializes and trains the XGBoost Regressor with early stopping."""
    print("\n" + "=" * 70)
    print(" STEP 4: TRAINING XGBOOST REGRESSOR")
    print("=" * 70)
    
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'gamma': 0.1,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'early_stopping_rounds': 50
    }
    
    print("Model Hyperparameters:")
    for k, v in params.items():
        print(f"  • {k:<22}: {v}")
    
    model = xgb.XGBRegressor(**params)
    
    print("\nTraining in progress...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    best_iteration = model.best_iteration
    best_score = model.best_score
    print(f"\nTraining Complete!")
    print(f"Best Iteration: {best_iteration} | Validation RMSE: {best_score:.4f}")
    
    return model


# =============================================================================
# 5. MODEL EVALUATION & PERFORMANCE METRICS
# =============================================================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Calculates and prints comprehensive evaluation metrics on train and test sets."""
    print("\n" + "=" * 70)
    print(" STEP 5: COMPREHENSIVE PERFORMANCE EVALUATION")
    print("=" * 70)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    def calculate_metrics(y_true, y_pred, split_name="Test"):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        evs = explained_variance_score(y_true, y_pred)
        
        return {
            'Split': split_name,
            'RMSE (°C)': rmse,
            'MAE (°C)': mae,
            'MSE (°C²)': mse,
            'R² Score': r2,
            'MAPE (%)': mape,
            'Explained Variance': evs
        }
    
    metrics_train = calculate_metrics(y_train, y_pred_train, "Train")
    metrics_test = calculate_metrics(y_test, y_pred_test, "Test (Unseen)")
    
    metrics_df = pd.DataFrame([metrics_train, metrics_test]).set_index('Split')
    print("\nPerformance Comparison Table:")
    print(metrics_df.to_string())
    
    return y_pred_test, metrics_df


# =============================================================================
# 6. VISUALIZATION & DIAGNOSTICS
# =============================================================================
def generate_evaluation_plots(model, y_test, y_pred_test, test_timestamps, feature_names, X_test, output_dir='.'):
    """Generates comprehensive multi-panel evaluation plots and detailed error diagnostics."""
    print("\n" + "=" * 70)
    print(" STEP 6: GENERATING MULTI-PANEL EVALUATION DIAGNOSTICS & PLOTS")
    print("=" * 70)
    
    # ----------------- FIGURE 1: CORE 4-PANEL DIAGNOSTICS -----------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Learning Curves
    evals_result = model.evals_result()
    train_rmse = evals_result['validation_0']['rmse']
    val_rmse = evals_result['validation_1']['rmse']
    epochs = range(len(train_rmse))
    
    axes[0, 0].plot(epochs, train_rmse, label='Train RMSE', color='#1f77b4', lw=2)
    axes[0, 0].plot(epochs, val_rmse, label='Validation RMSE', color='#ff7f0e', lw=2)
    axes[0, 0].axvline(model.best_iteration, color='red', linestyle='--', alpha=0.7, label=f'Best Iter ({model.best_iteration})')
    axes[0, 0].set_title('Training & Validation RMSE Learning Curve', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Boosting Iterations')
    axes[0, 0].set_ylabel('RMSE (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted Zoom
    zoom_len = min(300, len(y_test))
    axes[0, 1].plot(range(zoom_len), y_test.iloc[-zoom_len:].values, label='Actual Temperature', color='#2ca02c', lw=2)
    axes[0, 1].plot(range(zoom_len), y_pred_test[-zoom_len:], label='XGBoost Predicted', color='#d62728', linestyle='--', lw=2)
    axes[0, 1].set_title(f'Actual vs. Predicted Temperature (Recent {zoom_len} Hours Zoom)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Recent Hourly Time Steps')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residual Distribution
    residuals = y_test.values - y_pred_test
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='#9467bd', bins=40)
    axes[1, 0].axvline(0, color='black', linestyle='--', lw=1.5)
    axes[1, 0].set_title(f'Residuals Distribution (Mean={residuals.mean():.3f}, Std={residuals.std():.3f})', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Prediction Error (Actual - Predicted) °C')
    axes[1, 0].set_ylabel('Density / Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Top 15 Feature Importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Top 15 Feature Importance (Gain)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Relative Importance Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file_1 = os.path.join(output_dir, 'xgboost_evaluation_plots.png')
    plt.savefig(plot_file_1, bbox_inches='tight')
    plt.close()
    print(f"Saved primary evaluation plots to: {plot_file_1}")
    
    # ----------------- FIGURE 2: ADVANCED MULTI-ANGLE DIAGNOSTICS -----------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))
    
    # (A) Actual vs Predicted Scatter with Confidence Bands
    sample_size = min(10000, len(y_test))
    sample_indices = np.random.choice(len(y_test), size=sample_size, replace=False)
    axes2[0, 0].scatter(y_test.iloc[sample_indices], y_pred_test[sample_indices], alpha=0.25, color='#1f77b4', s=10, label='Test Sample')
    
    min_v = min(y_test.min(), y_pred_test.min()) - 1
    max_v = max(y_test.max(), y_pred_test.max()) + 1
    axes2[0, 0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Fit Line (y=x)')
    axes2[0, 0].fill_between([min_v, max_v], [min_v-1, max_v-1], [min_v+1, max_v+1], color='green', alpha=0.12, label='±1.0°C Tolerance Band')
    axes2[0, 0].set_title('Actual vs Predicted Scatter (10k Sample)', fontsize=13, fontweight='bold')
    axes2[0, 0].set_xlabel('Actual Temperature (°C)')
    axes2[0, 0].set_ylabel('Predicted Temperature (°C)')
    axes2[0, 0].legend()
    axes2[0, 0].grid(True, alpha=0.3)
    
    # (B) Diurnal 24-Hour Cycle
    eval_df = pd.DataFrame({
        'hour': X_test['hour'].values,
        'month': X_test['month'].values,
        'actual': y_test.values,
        'predicted': y_pred_test,
        'abs_error': np.abs(residuals)
    })
    
    hourly_avg = eval_df.groupby('hour')[['actual', 'predicted']].mean()
    hourly_std = eval_df.groupby('hour')[['actual', 'predicted']].std()
    
    axes2[0, 1].plot(hourly_avg.index, hourly_avg['actual'], 'o-', color='#2ca02c', lw=2.5, label='Actual Avg Temp')
    axes2[0, 1].plot(hourly_avg.index, hourly_avg['predicted'], 's--', color='#d62728', lw=2.5, label='Predicted Avg Temp')
    axes2[0, 1].fill_between(hourly_avg.index, hourly_avg['actual'] - hourly_std['actual'], hourly_avg['actual'] + hourly_std['actual'], color='#2ca02c', alpha=0.12, label='Actual ±1 Std')
    axes2[0, 1].set_title('Diurnal (24-Hour) Average Temperature Curve', fontsize=13, fontweight='bold')
    axes2[0, 1].set_xlabel('Hour of Day (00:00 to 23:00)')
    axes2[0, 1].set_ylabel('Temperature (°C)')
    axes2[0, 1].set_xticks(range(0, 24, 2))
    axes2[0, 1].legend()
    axes2[0, 1].grid(True, alpha=0.3)
    
    # (C) Monthly vs Hourly MAE Heatmap
    pivot_mae = eval_df.pivot_table(index='month', columns='hour', values='abs_error', aggfunc='mean')
    sns.heatmap(pivot_mae, cmap='YlOrRd', ax=axes2[1, 0], cbar_kws={'label': 'Mean Absolute Error (°C)'})
    axes2[1, 0].set_title('MAE Heatmap Across Month & Hour', fontsize=13, fontweight='bold')
    axes2[1, 0].set_xlabel('Hour of Day')
    axes2[1, 0].set_ylabel('Month')
    
    # (D) Cumulative Accuracy CDF Curve
    sorted_err = np.sort(eval_df['abs_error'].values)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    axes2[1, 1].plot(sorted_err, cdf, color='#1f77b4', lw=2.5)
    axes2[1, 1].axvline(0.5, color='green', linestyle=':', label=f'≤ 0.5°C: {(sorted_err <= 0.5).mean()*100:.1f}%')
    axes2[1, 1].axvline(1.0, color='orange', linestyle=':', label=f'≤ 1.0°C: {(sorted_err <= 1.0).mean()*100:.1f}%')
    axes2[1, 1].set_title('Cumulative Error Distribution (CDF)', fontsize=13, fontweight='bold')
    axes2[1, 1].set_xlabel('Absolute Prediction Error (°C)')
    axes2[1, 1].set_ylabel('Percentage of Predictions (%)')
    axes2[1, 1].set_xlim(0, 3.0)
    axes2[1, 1].legend(loc='lower right')
    axes2[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file_2 = os.path.join(output_dir, 'xgboost_detailed_diagnostics.png')
    plt.savefig(plot_file_2, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed diagnostics plots to: {plot_file_2}")


# =============================================================================
# 7. MODEL PERSISTENCE
# =============================================================================
def save_model_and_artifacts(model, feature_names, metrics_df, output_dir='.'):
    """Saves the trained model, feature metadata, and performance summary."""
    print("\n" + "=" * 70)
    print(" STEP 7: SAVING MODEL & METADATA ARTIFACTS")
    print("=" * 70)
    
    model_joblib_path = os.path.join(output_dir, 'xgboost_weather_model.joblib')
    model_json_path = os.path.join(output_dir, 'xgboost_weather_model.json')
    meta_path = os.path.join(output_dir, 'xgboost_model_metadata.json')
    
    joblib.dump(model, model_joblib_path)
    model.save_model(model_json_path)
    
    metadata = {
        'model_name': 'XGBoost Regressor (Weather Forecast)',
        'target_variable': 'temperature_2m',
        'n_features': len(feature_names),
        'features': feature_names,
        'test_metrics': metrics_df.loc['Test (Unseen)'].to_dict()
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Saved model (joblib)  : {model_joblib_path}")
    print(f"Saved model (JSON)    : {model_json_path}")
    print(f"Saved metadata (JSON) : {meta_path}")


# =============================================================================
# 8. USER INPUT PREDICTION ENGINE & SCENARIO SIMULATOR
# =============================================================================
def predict_custom_weather(user_input_dict, model, feature_names, defaults):
    """
    Takes custom user-specified weather and temporal parameters, derives cyclical
    and rolling features, fills missing inputs from defaults, and computes prediction.
    """
    input_row = defaults.copy()
    
    hour = user_input_dict.get('hour', 12)
    month = user_input_dict.get('month', 6)
    day = user_input_dict.get('day', 15)
    year = user_input_dict.get('year', 2026)
    day_name = user_input_dict.get('day_name', 'Monday')
    season = user_input_dict.get('season', 'Summer')
    
    # Direct sensor readings
    for key in ['relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'cloud_cover',
                'precipitation', 'rain', 'weather_code', 'pressure_msl', 'wind_speed_100m',
                'wind_direction_10m', 'wind_direction_100m', 'wind_gusts_10m',
                'soil_temperature_7_to_28cm', 'soil_temperature_28_to_100cm', 'soil_temperature_100_to_255cm',
                'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm',
                'et0_fao_evapotranspiration', 'vapour_pressure_deficit']:
        if key in user_input_dict:
            input_row[key] = user_input_dict[key]
            
    # Thermal momentum & lag
    temp_lag_1h = user_input_dict.get('temperature_2m_lag_1h', user_input_dict.get('temp_1h_ago', 25.0))
    temp_lag_24h = user_input_dict.get('temperature_2m_lag_24h', user_input_dict.get('temp_yesterday', temp_lag_1h))
    temp_rolling_mean = user_input_dict.get('temp_rolling_mean_24h', temp_lag_1h)
    temp_rolling_std = user_input_dict.get('temp_rolling_std_24h', 3.0)
    
    input_row['temperature_2m_lag_1h'] = temp_lag_1h
    input_row['temperature_2m_lag_24h'] = temp_lag_24h
    input_row['temp_rolling_mean_24h'] = temp_rolling_mean
    input_row['temp_rolling_std_24h'] = temp_rolling_std
    
    hum = input_row.get('relative_humidity_2m', 60.0)
    input_row['relative_humidity_2m_lag_1h'] = user_input_dict.get('relative_humidity_2m_lag_1h', hum)
    input_row['relative_humidity_2m_lag_24h'] = user_input_dict.get('relative_humidity_2m_lag_24h', hum)
    input_row['humidity_rolling_mean_24h'] = user_input_dict.get('humidity_rolling_mean_24h', hum)
    
    pres = input_row.get('surface_pressure', 915.0)
    input_row['surface_pressure_lag_1h'] = user_input_dict.get('surface_pressure_lag_1h', pres)
    input_row['surface_pressure_lag_24h'] = user_input_dict.get('surface_pressure_lag_24h', pres)
    
    wspd = input_row.get('wind_speed_10m', 10.0)
    input_row['wind_speed_10m_lag_1h'] = user_input_dict.get('wind_speed_10m_lag_1h', wspd)
    input_row['wind_speed_10m_lag_24h'] = user_input_dict.get('wind_speed_10m_lag_24h', wspd)
    
    # Cyclical trigonometric encodings
    input_row['hour'] = hour
    input_row['month'] = month
    input_row['day'] = day
    input_row['year'] = year
    input_row['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    input_row['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    input_row['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    input_row['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    
    approx_doy = int((month - 1) * 30.4 + day)
    input_row['day_of_year'] = approx_doy
    input_row['doy_sin'] = np.sin(2 * np.pi * approx_doy / 365.25)
    input_row['doy_cos'] = np.cos(2 * np.pi * approx_doy / 365.25)
    
    input_row['day_of_week'] = user_input_dict.get('day_of_week', 2)
    input_row['is_weekend'] = 1.0 if day_name in ['Saturday', 'Sunday'] else 0.0
    input_row['quarter'] = int((month - 1) // 3 + 1)
    precip = input_row.get('precipitation', 0.0)
    input_row['log_precipitation'] = np.log1p(precip)
    
    # Categorical OHE flags
    for s in ['Post_Monsoon', 'Summer', 'Winter']:
        input_row[f'season_{s}'] = 1.0 if season.lower() == s.lower() else 0.0
    for d in ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']:
        input_row[f'day_name_{d}'] = 1.0 if day_name.lower() == d.lower() else 0.0
        
    input_df = pd.DataFrame([input_row])[feature_names]
    pred_temp = float(model.predict(input_df)[0])
    
    if pred_temp < 15:
        category = "❄️ Chilly / Cold"
    elif pred_temp < 22:
        category = "🍃 Cool & Pleasant"
    elif pred_temp < 28:
        category = "☀️ Warm & Comfortable"
    elif pred_temp < 34:
        category = "🔥 Hot"
    else:
        category = "🚨 Very Hot / Heatwave"
        
    return {
        'predicted_temperature_c': round(pred_temp, 2),
        'predicted_temperature_f': round(pred_temp * 9/5 + 32, 2),
        'thermal_category': category,
        'input_summary': {
            'Time': f"{hour:02d}:00",
            'Month': month,
            'Season': season,
            'Prior Temp (1h ago)': f"{temp_lag_1h:.1f} °C",
            'Humidity': f"{input_row.get('relative_humidity_2m', 60):.1f}%",
            'Surface Pressure': f"{input_row.get('surface_pressure', 915):.1f} hPa",
            'Wind Speed': f"{input_row.get('wind_speed_10m', 10):.1f} km/h",
            'Cloud Cover': f"{input_row.get('cloud_cover', 20):.1f}%"
        }
    }


def display_weather_card(res):
    """Displays a clean ASCII dashboard of prediction results."""
    print("\n" + "=" * 60)
    print(" 🌦️  XGBOOST WEATHER FORECAST SUMMARY")
    print("=" * 60)
    print(f"  🌡️  PREDICTED TEMPERATURE : {res['predicted_temperature_c']:.2f} °C ({res['predicted_temperature_f']:.2f} °F)")
    print(f"  🏷️  COMFORT CLASSIFICATION: {res['thermal_category']}")
    print("-" * 60)
    print("  📋 INPUT CONDITIONS:")
    for k, v in res['input_summary'].items():
        print(f"     • {k:<22}: {v}")
    print("=" * 60)


def run_interactive_inference_demo(model, feature_names, defaults):
    """Demonstrates custom user input prediction and presets."""
    print("\n" + "=" * 70)
    print(" STEP 8: CUSTOM USER INPUT PREDICTION & SCENARIO DEMONSTRATIONS")
    print("=" * 70)
    
    # 1. Custom User Sandbox Example
    custom_input = {
        'hour': 14,
        'month': 5,
        'season': 'Summer',
        'temp_1h_ago': 32.5,
        'relative_humidity_2m': 38.0,
        'surface_pressure': 911.0,
        'wind_speed_10m': 14.5,
        'cloud_cover': 15.0
    }
    print("Running Prediction on Custom Input Sandbox (May Summer Afternoon):")
    res = predict_custom_weather(custom_input, model, feature_names, defaults)
    display_weather_card(res)
    
    # 2. Preset Scenarios Benchmark
    scenarios = {
        "Summer Afternoon": {'hour': 14, 'month': 5, 'season': 'Summer', 'temp_1h_ago': 33.0, 'relative_humidity_2m': 30.0},
        "Monsoon Evening":  {'hour': 18, 'month': 8, 'season': 'Monsoon', 'temp_1h_ago': 22.0, 'relative_humidity_2m': 90.0, 'precipitation': 15.0},
        "Winter Morning":   {'hour': 6,  'month': 1, 'season': 'Winter', 'temp_1h_ago': 14.5, 'relative_humidity_2m': 75.0, 'surface_pressure': 922.0}
    }
    
    print("\nSimulating Benchmark Weather Presets:")
    for name, scen in scenarios.items():
        r = predict_custom_weather(scen, model, feature_names, defaults)
        print(f"  • {name:<18}: {r['predicted_temperature_c']:>5.2f} °C | Feel: {r['thermal_category']}")


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    print("#####################################################################")
    print("#      XGBOOST WEATHER FORECASTING & EVALUATION PIPELINE            #")
    print("#####################################################################")
    
    # 1. Load data
    df = load_data('bengaluru_preprocessed.csv')
    
    # 2. Feature Engineering & Preprocessing Audit
    data = preprocess_and_engineer_features(df, target_col='temperature_2m')
    
    # 3. Train-Val-Test Chronological Split
    X_train, y_train, X_val, y_val, X_test, y_test, test_timestamps, feature_names, defaults = split_data(data)
    
    # 4. Train Model
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # 5. Evaluate Model
    y_pred_test, metrics_df = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # 6. Generate Multi-Panel Visual Plots
    generate_evaluation_plots(model, y_test, y_pred_test, test_timestamps, feature_names, X_test)
    
    # 7. Save Model Artifacts
    save_model_and_artifacts(model, feature_names, metrics_df)
    
    # 8. User Input & Scenario Inference Demo
    run_interactive_inference_demo(model, feature_names, defaults)
    
    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
