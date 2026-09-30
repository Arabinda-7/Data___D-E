"""
=============================================================================
Deep Learning LSTM Neural Network for Weather Time-Series Forecasting
Dataset: NLP/bengaluru_preprocessed.csv
Architecture: Deep Stacked Bidirectional LSTM + Multi-Head Temporal Attention + Residual MLP Head
Target: 2-meter Air Temperature (temperature_2m)
=============================================================================
"""

import os
import sys
import json
import time
import math
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)

# ---------------------------------------------------------------------------
# Aesthetics and Reproducibility Setup
# ---------------------------------------------------------------------------
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(42)

# =============================================================================
# STEP 1: DATA INGESTION & AUDIT
# =============================================================================
def load_and_audit_data(csv_path="NLP/bengaluru_preprocessed.csv"):
    """
    Load time-series weather data, verify datetime ordering and check data consistency.
    """
    print("=" * 80)
    print(" STEP 1: DATA INGESTION & TIME-SERIES AUDIT")
    print("=" * 80)
    
    if not os.path.exists(csv_path):
        # Fallback to current folder if NLP is not prefixed
        if os.path.exists("bengaluru_preprocessed.csv"):
            csv_path = "bengaluru_preprocessed.csv"
        else:
            raise FileNotFoundError(f"Could not find dataset at {csv_path}")

    print(f"--> Reading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"--> Raw dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Chronological sorting
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"--> Date coverage: {df['time'].min()} to {df['time'].max()}")
    print(f"--> Total time span: {(df['time'].max() - df['time'].min()).days} days")
    print(f"--> Missing values check: {df.isnull().sum().sum()} total nulls across dataset")
    
    return df


# =============================================================================
# STEP 2: METEOROLOGICAL & CYCLICAL FEATURE ENGINEERING
# =============================================================================
def engineer_meteorological_features(df):
    """
    Derive domain-rich meteorological indicators and continuous cyclical temporal signals:
    1. Cyclical Sine/Cosine transformations for Hour (24h), Day of Year (365.25d), Month (12m)
    2. Wind Vector Decomposition (U = East-West, V = North-South)
    3. Dew Point Depression (temperature_2m - dew_point_2m)
    4. Vapor Pressure Deficit / Relative humidity interaction
    5. Soil Heat Transmission Gradients (surface soil temp - deep soil temp)
    """
    print("\n" + "=" * 80)
    print(" STEP 2: ADVANCED FEATURE ENGINEERING & PHYSICS-INFORMED TRANSFORMS")
    print("=" * 80)
    
    data = df.copy()
    
    # 1. Cyclical Time Transforms
    hour = data['time'].dt.hour
    day_of_year = data['time'].dt.dayofyear
    month = data['time'].dt.month
    
    data['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    data['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    data['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    data['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    data['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12.0)
    data['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12.0)
    
    # 2. Wind Direction Trigonometric Components (U, V vectors)
    if 'wind_speed_10m' in data.columns and 'wind_direction_10m' in data.columns:
        rad_10 = np.radians(data['wind_direction_10m'])
        data['wind_u_10m'] = -data['wind_speed_10m'] * np.sin(rad_10)
        data['wind_v_10m'] = -data['wind_speed_10m'] * np.cos(rad_10)
        
    if 'wind_speed_100m' in data.columns and 'wind_direction_100m' in data.columns:
        rad_100 = np.radians(data['wind_direction_100m'])
        data['wind_u_100m'] = -data['wind_speed_100m'] * np.sin(rad_100)
        data['wind_v_100m'] = -data['wind_speed_100m'] * np.cos(rad_100)

    # 3. Meteorological Physics Indicators
    if 'dew_point_2m' in data.columns and 'temperature_2m' in data.columns:
        data['dew_point_depression'] = data['temperature_2m'] - data['dew_point_2m']
        
    if 'soil_temperature_0_to_7cm' in data.columns and 'soil_temperature_100_to_255cm' in data.columns:
        data['soil_thermal_gradient'] = data['soil_temperature_0_to_7cm'] - data['soil_temperature_100_to_255cm']

    if 'surface_pressure' in data.columns and 'pressure_msl' in data.columns:
        data['pressure_deficit'] = data['pressure_msl'] - data['surface_pressure']

    # Drop raw non-numeric / string columns (e.g. season names, day names)
    cols_to_drop = ['season', 'day_name', 'year', 'day_of_week']
    for col in cols_to_drop:
        if col in data.columns:
            data = data.drop(columns=[col])
            
    print(f"--> Feature engineering complete. Total feature columns: {len(data.columns) - 1}")
    return data


# =============================================================================
# STEP 3: DATA PREPARATION & NORMALIZATION
# =============================================================================
def prepare_and_scale_data(df, target_col='temperature_2m', test_ratio=0.10, val_ratio=0.10):
    """
    Split time-series strictly chronologically without data leakage.
    Fit robust MinMax scalers on the training partition only.
    """
    print("\n" + "=" * 80)
    print(" STEP 3: TIME-SERIES CHRONOLOGICAL SPLITTING & SCALING")
    print("=" * 80)
    
    feature_cols = [c for c in df.columns if c not in ['time', target_col]]
    
    total_len = len(df)
    test_size = int(total_len * test_ratio)
    val_size = int(total_len * val_ratio)
    train_size = total_len - val_size - test_size
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size + val_size].reset_index(drop=True)
    test_df = df.iloc[train_size + val_size:].reset_index(drop=True)
    
    print(f"--> Train partition: {len(train_df):,} records ({train_df['time'].min().date()} to {train_df['time'].max().date()})")
    print(f"--> Val partition:   {len(val_df):,} records ({val_df['time'].min().date()} to {val_df['time'].max().date()})")
    print(f"--> Test partition:  {len(test_df):,} records ({test_df['time'].min().date()} to {test_df['time'].max().date()})")
    
    # Feature Scaling (Fit on Train ONLY)
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    train_features_scaled = feature_scaler.fit_transform(train_df[feature_cols])
    val_features_scaled = feature_scaler.transform(val_df[feature_cols])
    test_features_scaled = feature_scaler.transform(test_df[feature_cols])
    
    train_target_scaled = target_scaler.fit_transform(train_df[[target_col]])
    val_target_scaled = target_scaler.transform(val_df[[target_col]])
    test_target_scaled = target_scaler.transform(test_df[[target_col]])
    
    # Combine scaled features + target into arrays for sequence generation
    # Feature matrix will have [Target, Features...] for historical context
    train_matrix = np.hstack([train_target_scaled, train_features_scaled])
    val_matrix = np.hstack([val_target_scaled, val_features_scaled])
    test_matrix = np.hstack([test_target_scaled, test_features_scaled])
    
    all_feature_names = [target_col] + feature_cols
    
    return {
        'train_matrix': train_matrix,
        'val_matrix': val_matrix,
        'test_matrix': test_matrix,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols,
        'all_feature_names': all_feature_names
    }


# =============================================================================
# STEP 4: VECTORIZED SLIDING WINDOW GENERATION & TENSORDATASETS
# =============================================================================
def create_sliding_window_tensors(data_matrix, seq_len=24, target_idx=0):
    """
    Vectorized sliding window generator using numpy strided memory views.
    Constructs 3D tensors [Num_Samples, Sequence_Length, Num_Features] in milliseconds.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    num_samples = len(data_matrix) - seq_len
    # sliding_window_view creates [num_samples + 1, num_features, seq_len]
    windows = sliding_window_view(data_matrix, window_shape=seq_len, axis=0)[:num_samples]
    # Transpose to [num_samples, seq_len, num_features]
    windows_3d = np.transpose(windows, (0, 2, 1)).astype(np.float32)
    y_targets = data_matrix[seq_len:, target_idx:target_idx+1].astype(np.float32)
    
    x_tensor = torch.from_numpy(windows_3d)
    y_tensor = torch.from_numpy(y_targets)
    return torch.utils.data.TensorDataset(x_tensor, y_tensor)


# =============================================================================
# STEP 5: DEEP NEURAL NETWORK & LSTM ARCHITECTURE
# =============================================================================
class TemporalAttention(nn.Module):
    """
    Self-Attention Mechanism across the temporal sequence dimension.
    Learns dynamic weighting over historical timesteps.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs):
        # rnn_outputs: [Batch_Size, Seq_Len, Hidden_Dim]
        scores = self.attention_weights(rnn_outputs)  # [Batch_Size, Seq_Len, 1]
        attn_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attn_weights * rnn_outputs, dim=1)  # [Batch_Size, Hidden_Dim]
        return context_vector, attn_weights


class WeatherLSTMNeuralNetwork(nn.Module):
    """
    State-of-the-art Deep Recurrent Architecture:
    1. Input Projection Layer + LayerNorm
    2. Multi-layer Stacked Bidirectional LSTM
    3. Temporal Attention Pooling
    4. Deep Fully Connected Residual MLP Head with GELU activations and Dropout
    5. Output Layer predicting next-step continuous weather target
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 1. Feature Input Projector
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 2. Deep LSTM Stack
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        lstm_out_dim = hidden_dim * self.num_directions
        
        # 3. Temporal Attention Mechanism
        self.attention = TemporalAttention(hidden_dim=lstm_out_dim)
        
        # 4. Residual Deep MLP Head
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
        # x: [Batch_Size, Seq_Len, Input_Dim]
        proj = self.input_layer(x)  # [Batch, Seq_Len, Hidden_Dim]
        lstm_out, _ = self.lstm(proj)  # [Batch, Seq_Len, Hidden_Dim * directions]
        
        # Attention context
        context, _ = self.attention(lstm_out)  # [Batch, Hidden_Dim * directions]
        
        # Output prediction
        out = self.mlp_head(context)  # [Batch, 1]
        return out


# =============================================================================
# STEP 6: TRAINING & VALIDATION PIPELINE WITH ADAPTIVE SCHEDULING
# =============================================================================
def train_lstm_model(
    model,
    train_loader,
    val_loader,
    target_scaler,
    device,
    epochs=25,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=6,
    save_path="NLP/lstm_best_model.pt"
):
    """
    Supervised Training Loop with Smooth L1 (Huber) Loss, AdamW Optimizer,
    Cosine Annealing Learning Rate Scheduler, Gradient Clipping & Early Stopping.
    """
    print("\n" + "=" * 80)
    print(" STEP 6: NEURAL NETWORK TRAINING & OPTIMIZATION")
    print("=" * 80)
    
    criterion = nn.SmoothL1Loss()  # Robust against extreme outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae_celsius': [],
        'lr': []
    }
    
    print(f"--> Training on device: {device}")
    print(f"--> Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 80)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val MAE (°C)':<14} | {'LR':<10} | {'Status'}")
    print("-" * 80)
    
    start_total_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss_acc = 0.0
        train_batches = 0
        total_batches = len(train_loader)
        
        for b_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            
            # Gradient clipping to stabilize recurrent gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss_acc += loss.item()
            train_batches += 1
            
            if (b_idx + 1) % 150 == 0 or (b_idx + 1) == total_batches:
                print(f"   [Epoch {epoch:02d}] Step {b_idx+1}/{total_batches} | Running Loss: {train_loss_acc / train_batches:.5f}", flush=True)
            
        avg_train_loss = train_loss_acc / train_batches
        
        # 2. Validation Phase
        model.eval()
        val_loss_acc = 0.0
        val_batches = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                
                val_loss_acc += loss.item()
                val_batches += 1
                
                all_val_preds.append(preds.cpu().numpy())
                all_val_targets.append(batch_y.cpu().numpy())
                
        avg_val_loss = val_loss_acc / val_batches
        
        # Calculate real-scale Celsius MAE for physical interpretability
        val_preds_arr = np.vstack(all_val_preds)
        val_targets_arr = np.vstack(all_val_targets)
        val_preds_celsius = target_scaler.inverse_transform(val_preds_arr)
        val_targets_celsius = target_scaler.inverse_transform(val_targets_arr)
        val_mae_c = mean_absolute_error(val_targets_celsius, val_preds_celsius)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae_celsius'].append(val_mae_c)
        history['lr'].append(current_lr)
        
        # Checkpoint Best Model
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_mae_celsius': val_mae_c
            }, save_path)
            status = f"[SAVED] Val Loss: {best_val_loss:.6f}"
        else:
            early_stop_counter += 1
            status = f"Patience {early_stop_counter}/{patience}"
            
        print(f"{epoch:<8} | {avg_train_loss:<12.6f} | {avg_val_loss:<12.6f} | {val_mae_c:<14.3f} | {current_lr:<10.2e} | {status}", flush=True)
        
        if early_stop_counter >= patience:
            print(f"\n[!] Early stopping triggered at Epoch {epoch}. Restoring best weights.")
            break
            
    total_train_time = time.time() - start_total_time
    print(f"\n--> Training finished in {total_train_time:.1f}s. Best Validation Loss: {best_val_loss:.6f}")
    
    # Reload best checkpoint weights
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, history


# =============================================================================
# STEP 7: RIGOROUS EVALUATION & BENCHMARKING
# =============================================================================
def evaluate_lstm_model(model, test_loader, test_df, target_scaler, seq_len, device):
    """
    Perform rigorous evaluation on unseen out-of-sample test partition.
    Calculates RMSE, MAE, R², MAPE, EV, Max Error, and compares against Baseline.
    """
    print("\n" + "=" * 80)
    print(" STEP 7: COMPREHENSIVE OUT-OF-SAMPLE TEST EVALUATION")
    print("=" * 80)
    
    model.eval()
    all_preds = []
    all_actuals = []
    
    start_eval_time = time.time()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.append(preds.cpu().numpy())
            all_actuals.append(batch_y.numpy())
            
    eval_duration = time.time() - start_eval_time
    
    preds_scaled = np.vstack(all_preds)
    actuals_scaled = np.vstack(all_actuals)
    
    # Invert scaling to true physical Celsius scale
    preds_celsius = target_scaler.inverse_transform(preds_scaled).flatten()
    actuals_celsius = target_scaler.inverse_transform(actuals_scaled).flatten()
    
    # Align test timestamps (offset by seq_len)
    test_timestamps = test_df['time'].iloc[seq_len:].reset_index(drop=True)
    
    # Core Metrics
    mse = mean_squared_error(actuals_celsius, preds_celsius)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actuals_celsius, preds_celsius)
    r2 = r2_score(actuals_celsius, preds_celsius)
    mape = mean_absolute_percentage_error(actuals_celsius, preds_celsius) * 100
    ev = explained_variance_score(actuals_celsius, preds_celsius)
    max_err = np.max(np.abs(actuals_celsius - preds_celsius))
    
    # Baseline Persistence Benchmark (Predicting t using t-1)
    baseline_celsius = actuals_celsius[:-1]
    baseline_actuals = actuals_celsius[1:]
    baseline_rmse = math.sqrt(mean_squared_error(baseline_actuals, baseline_celsius))
    baseline_mae = mean_absolute_error(baseline_actuals, baseline_celsius)
    baseline_r2 = r2_score(baseline_actuals, baseline_celsius)
    
    print("\n" + "-" * 70)
    print(f"{'METRIC':<35} | {'LSTM NEURAL NET':<15} | {'PERSISTENCE BASELINE':<15}")
    print("-" * 70)
    print(f"{'Root Mean Squared Error (RMSE)':<35} | {rmse:<15.4f} deg C | {baseline_rmse:<15.4f} deg C")
    print(f"{'Mean Absolute Error (MAE)':<35} | {mae:<15.4f} deg C | {baseline_mae:<15.4f} deg C")
    print(f"{'Coefficient of Determination (R2)':<35} | {r2:<15.4f}       | {baseline_r2:<15.4f}")
    print(f"{'Mean Abs. Pct. Error (MAPE)':<35} | {mape:<15.2f} %     | {'-':<15}")
    print(f"{'Explained Variance (EV)':<35} | {ev:<15.4f}       | {'-':<15}")
    print(f"{'Maximum Absolute Error':<35} | {max_err:<15.4f} deg C | {'-':<15}")
    print(f"{'Inference Latency per Sample':<35} | {eval_duration / len(preds_celsius) * 1000:<15.3f} ms    | {'-':<15}")
    print("-" * 70)
    
    results = {
        'timestamps': test_timestamps,
        'actuals': actuals_celsius,
        'predictions': preds_celsius,
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'explained_variance': float(ev),
            'max_error': float(max_err),
            'baseline_rmse': float(baseline_rmse),
            'baseline_mae': float(baseline_mae),
            'baseline_r2': float(baseline_r2)
        }
    }
    return results


# =============================================================================
# STEP 8: HIGH-RESOLUTION VISUALIZATION & DIAGNOSTICS SUITE
# =============================================================================
def generate_evaluation_visualizations(history, eval_results, output_path="NLP/lstm_evaluation_plots.png"):
    """
    Generate a 6-panel comprehensive diagnostic dashboard:
    1. Training & Validation Loss curves over epochs
    2. Real-scale Validation MAE (°C) progression
    3. Actual vs Predicted 14-Day High-Resolution Temporal Sequence
    4. Scatter Parity Plot with Ideal Fit Line & R²
    5. Residual Distribution (Error Normality & Skewness)
    6. Residual Error vs Predicted Value (Heteroscedasticity Analysis)
    """
    print("\n" + "=" * 80)
    print(" STEP 8: GENERATING HIGH-RESOLUTION DIAGNOSTIC PLOTS")
    print("=" * 80)
    
    fig, axs = plt.subplots(3, 2, figsize=(18, 16))
    fig.patch.set_facecolor('#0f172a')
    
    # Common dark-mode palette styling
    title_color = '#f8fafc'
    text_color = '#cbd5e1'
    grid_color = '#334155'
    
    for ax in axs.flat:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors=text_color, labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.35, color=grid_color)
        for spine in ax.spines.values():
            spine.set_color('#475569')

    # Panel 1: Learning Curves (Loss)
    epochs = range(1, len(history['train_loss']) + 1)
    axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss (Smooth L1)', color='#38bdf8', linewidth=2.2)
    axs[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', color='#f43f5e', linewidth=2.2, linestyle='--')
    axs[0, 0].set_title('1. Learning Curves (Huber Loss)', fontsize=13, fontweight='bold', color=title_color, pad=10)
    axs[0, 0].set_xlabel('Epoch', color=text_color)
    axs[0, 0].set_ylabel('Loss', color=text_color)
    axs[0, 0].legend(frameon=True, facecolor='#0f172a', edgecolor='#475569', labelcolor=text_color)

    # Panel 2: Physical Metric Progression
    axs[0, 1].plot(epochs, history['val_mae_celsius'], label='Validation MAE (°C)', color='#10b981', linewidth=2.2, marker='o')
    axs[0, 1].set_title('2. Physical Validation Error (°C) over Epochs', fontsize=13, fontweight='bold', color=title_color, pad=10)
    axs[0, 1].set_xlabel('Epoch', color=text_color)
    axs[0, 1].set_ylabel('MAE (°C)', color=text_color)
    axs[0, 1].legend(frameon=True, facecolor='#0f172a', edgecolor='#475569', labelcolor=text_color)

    # Panel 3: 14-Day Micro Time-Series Zoom (336 hourly observations)
    zoom_len = min(336, len(eval_results['actuals']))
    zoom_times = eval_results['timestamps'].iloc[:zoom_len]
    zoom_act = eval_results['actuals'][:zoom_len]
    zoom_pred = eval_results['predictions'][:zoom_len]

    axs[1, 0].plot(zoom_times, zoom_act, label='Actual Temperature (°C)', color='#38bdf8', linewidth=2.0, alpha=0.9)
    axs[1, 0].plot(zoom_times, zoom_pred, label='LSTM Predicted (°C)', color='#fbbf24', linewidth=2.0, linestyle='--')
    axs[1, 0].set_title(f'3. Time-Series Tracking (First {zoom_len // 24} Days of Test Set)', fontsize=13, fontweight='bold', color=title_color, pad=10)
    axs[1, 0].set_xlabel('Datetime', color=text_color)
    axs[1, 0].set_ylabel('Temperature (°C)', color=text_color)
    axs[1, 0].tick_params(axis='x', rotation=25)
    axs[1, 0].legend(frameon=True, facecolor='#0f172a', edgecolor='#475569', labelcolor=text_color)

    # Panel 4: Parity Scatter Plot
    act = eval_results['actuals']
    pred = eval_results['predictions']
    r2 = eval_results['metrics']['r2']
    rmse = eval_results['metrics']['rmse']
    
    axs[1, 1].scatter(act, pred, alpha=0.25, color='#a855f7', s=12, edgecolors='none', label='Test Predictions')
    min_val = min(np.min(act), np.min(pred))
    max_val = max(np.max(act), np.max(pred))
    axs[1, 1].plot([min_val, max_val], [min_val, max_val], color='#ef4444', linestyle='--', linewidth=2.0, label='Ideal Perfect Line (1:1)')
    axs[1, 1].set_title(f'4. Parity Correlation (R² = {r2:.4f}, RMSE = {rmse:.2f}°C)', fontsize=13, fontweight='bold', color=title_color, pad=10)
    axs[1, 1].set_xlabel('Ground Truth Actual (°C)', color=text_color)
    axs[1, 1].set_ylabel('LSTM Predicted (°C)', color=text_color)
    axs[1, 1].legend(frameon=True, facecolor='#0f172a', edgecolor='#475569', labelcolor=text_color)

    # Panel 5: Residual Error Distribution
    residuals = act - pred
    sns.histplot(residuals, kde=True, ax=axs[2, 0], color='#06b6d4', bins=50, alpha=0.6, edgecolor='#0f172a')
    axs[2, 0].axvline(0, color='#ef4444', linestyle='--', linewidth=1.5, label='Zero Error')
    axs[2, 0].axvline(np.mean(residuals), color='#fbbf24', linestyle=':', linewidth=1.8, label=f'Mean Bias: {np.mean(residuals):.3f}°C')
    axs[2, 0].set_title('5. Residual Error Distribution (Normality & Bias)', fontsize=13, fontweight='bold', color=title_color, pad=10)
    axs[2, 0].set_xlabel('Residual (Actual - Predicted) [°C]', color=text_color)
    axs[2, 0].set_ylabel('Frequency', color=text_color)
    axs[2, 0].legend(frameon=True, facecolor='#0f172a', edgecolor='#475569', labelcolor=text_color)

    # Panel 6: Residuals vs Predicted
    axs[2, 1].scatter(pred, residuals, alpha=0.25, color='#f59e0b', s=12, edgecolors='none')
    axs[2, 1].axhline(0, color='#ef4444', linestyle='--', linewidth=1.8, label='Zero Residual Line')
    axs[2, 1].axhline(np.std(residuals)*2, color='#94a3b8', linestyle=':', label='±2σ Bound')
    axs[2, 1].axhline(-np.std(residuals)*2, color='#94a3b8', linestyle=':')
    axs[2, 1].set_title('6. Homoscedasticity Analysis (Residual vs Predicted)', fontsize=13, fontweight='bold', color=title_color, pad=10)
    axs[2, 1].set_xlabel('Predicted Temperature (°C)', color=text_color)
    axs[2, 1].set_ylabel('Residual Error (°C)', color=text_color)
    axs[2, 1].legend(frameon=True, facecolor='#0f172a', edgecolor='#475569', labelcolor=text_color)

    plt.suptitle('Bengaluru Weather Forecasting — Deep BiLSTM Neural Network Evaluation Suite',
                 fontsize=16, fontweight='bold', color='#ffffff', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"--> Diagnostic plots saved successfully to: {output_path}")


# =============================================================================
# STEP 9: ARTIFACT SERIALIZATION & METADATA EXPORT
# =============================================================================
def export_model_artifacts(
    model,
    feature_scaler,
    target_scaler,
    feature_cols,
    all_feature_names,
    seq_len,
    eval_metrics,
    output_dir="NLP"
):
    """
    Save all necessary assets for offline/online deployment:
    1. Scaler pipelines (joblib)
    2. Comprehensive metadata schema (json)
    3. PyTorch architecture config and weights
    """
    print("\n" + "=" * 80)
    print(" STEP 9: SERIALIZING MODEL ARTIFACTS & SERVING METADATA")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Scalers
    scaler_bundle = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols,
        'all_feature_names': all_feature_names,
        'seq_len': seq_len
    }
    scaler_path = os.path.join(output_dir, "lstm_scalers.joblib")
    joblib.dump(scaler_bundle, scaler_path)
    print(f"--> Scalers saved to: {scaler_path}")
    
    # Save Metadata
    metadata = {
        'model_name': 'Deep Stacked BiLSTM Neural Network with Attention',
        'target_variable': 'temperature_2m',
        'sequence_length': seq_len,
        'num_input_features': len(all_feature_names),
        'input_features': all_feature_names,
        'metrics': eval_metrics,
        'architecture_config': {
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.2,
            'attention': True
        }
    }
    meta_path = os.path.join(output_dir, "lstm_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"--> Metadata saved to: {meta_path}")


# =============================================================================
# STEP 10: STANDALONE INFERENCE SIMULATOR
# =============================================================================
def run_sample_inference(model, sample_sequence_raw, feature_scaler, target_scaler, all_feature_names, device):
    """
    Given an unscaled DataFrame or 2D array of the past `seq_len` timesteps,
    scales features, runs forward pass through the Neural Network, and inverts target.
    """
    model.eval()
    
    # Assume sample_sequence_raw is a DataFrame containing `all_feature_names`
    # or array of shape [seq_len, num_features]
    target_col = all_feature_names[0]
    feature_cols = all_feature_names[1:]
    
    target_scaled = target_scaler.transform(sample_sequence_raw[[target_col]])
    features_scaled = feature_scaler.transform(sample_sequence_raw[feature_cols])
    
    matrix = np.hstack([target_scaled, features_scaled])
    tensor_in = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)  # [1, seq_len, input_dim]
    
    with torch.no_grad():
        pred_scaled = model(tensor_in).cpu().numpy()
        pred_celsius = target_scaler.inverse_transform(pred_scaled).item()
        
    return pred_celsius


# =============================================================================
# MASTER EXECUTION PIPELINE
# =============================================================================
def main():
    print("""
    #########################################################################
    #   BENGALURU WEATHER FORECASTING: DEEP LSTM NEURAL NETWORK PIPELINE    #
    #########################################################################
    """)
    
    # Configure Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Hardware Compute Device: {device}")
    
    # 1. Load Data
    raw_df = load_and_audit_data(csv_path="NLP/bengaluru_preprocessed.csv")
    
    # 2. Feature Engineering
    engineered_df = engineer_meteorological_features(raw_df)
    
    # 3. Data Splitting & Scaling
    # Use 24 hours (1 diurnal day) sequence context
    seq_len = 24
    data_bundle = prepare_and_scale_data(
        engineered_df,
        target_col='temperature_2m',
        test_ratio=0.10,
        val_ratio=0.10
    )
    
    # 4. Create PyTorch Datasets & DataLoaders
    print("\n" + "=" * 80)
    print(" STEP 4: BUILDING PYTORCH TENSOR DATASETS & VECTORIZED LOADERS")
    print("=" * 80)
    
    train_dataset = create_sliding_window_tensors(data_bundle['train_matrix'], seq_len=seq_len)
    val_dataset = create_sliding_window_tensors(data_bundle['val_matrix'], seq_len=seq_len)
    test_dataset = create_sliding_window_tensors(data_bundle['test_matrix'], seq_len=seq_len)
    
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = data_bundle['train_matrix'].shape[1]
    print(f"--> Input features per timestep: {input_dim}")
    print(f"--> Lookback Sequence Length: {seq_len} timesteps (48 hours)")
    print(f"--> Train sequences: {len(train_dataset):,}")
    print(f"--> Validation sequences: {len(val_dataset):,}")
    print(f"--> Test sequences: {len(test_dataset):,}")
    
    # 5. Instantiate Deep Neural Network Model
    print("\n" + "=" * 80)
    print(" STEP 5: INITIALIZING DEEP STACKED BiLSTM + ATTENTION NETWORK")
    print("=" * 80)
    
    model = WeatherLSTMNeuralNetwork(
        input_dim=input_dim,
        hidden_dim=96,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    ).to(device)
    
    print(model)
    
    # 6. Train the Neural Network
    best_model_path = "NLP/lstm_best_model.pt"
    trained_model, history = train_lstm_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        target_scaler=data_bundle['target_scaler'],
        device=device,
        epochs=12,
        learning_rate=1e-3,
        patience=4,
        save_path=best_model_path
    )
    
    # 7. Evaluate on Unseen Test Partition
    eval_results = evaluate_lstm_model(
        model=trained_model,
        test_loader=test_loader,
        test_df=data_bundle['test_df'],
        target_scaler=data_bundle['target_scaler'],
        seq_len=seq_len,
        device=device
    )
    
    # 8. Generate Visualizations
    generate_evaluation_visualizations(
        history=history,
        eval_results=eval_results,
        output_path="NLP/lstm_evaluation_plots.png"
    )
    
    # 9. Export Artifacts
    export_model_artifacts(
        model=trained_model,
        feature_scaler=data_bundle['feature_scaler'],
        target_scaler=data_bundle['target_scaler'],
        feature_cols=data_bundle['feature_cols'],
        all_feature_names=data_bundle['all_feature_names'],
        seq_len=seq_len,
        eval_metrics=eval_results['metrics'],
        output_dir="NLP"
    )
    
    # 10. Sample Test Simulation
    print("\n" + "=" * 80)
    print(" STEP 10: SAMPLE LIVE PREDICTION SIMULATION")
    print("=" * 80)
    sample_df = data_bundle['test_df'].iloc[100:100 + seq_len][data_bundle['all_feature_names']]
    true_next_temp = data_bundle['test_df'].iloc[100 + seq_len]['temperature_2m']
    sim_pred = run_sample_inference(
        model=trained_model,
        sample_sequence_raw=sample_df,
        feature_scaler=data_bundle['feature_scaler'],
        target_scaler=data_bundle['target_scaler'],
        all_feature_names=data_bundle['all_feature_names'],
        device=device
    )
    print(f"--> Input 24h Sequence ending at: {data_bundle['test_df'].iloc[100+seq_len-1]['time']}")
    print(f"--> Target Timestep:             {data_bundle['test_df'].iloc[100+seq_len]['time']}")
    print(f"--> Ground Truth Actual Temp:    {true_next_temp:.2f} deg C")
    print(f"--> LSTM Model Prediction:       {sim_pred:.2f} deg C")
    print(f"--> Absolute Difference:         {abs(true_next_temp - sim_pred):.3f} deg C")
    print("\n" + "=" * 80)
    print(" [SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n--> Tip: Run 'python NLP/interactive_predict.py' for the dedicated interactive CLI.")


if __name__ == "__main__":
    if "--predict" in sys.argv or "--interactive" in sys.argv:
        from interactive_predict import main as run_interactive
        run_interactive()
    else:
        main()

