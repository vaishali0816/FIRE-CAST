"""
Wildfire Next-Day Active-Fire Count Prediction Pipeline
Region: California, USA
Target: Next-Day Satellite Active Fire Detections (Count of thermal anomalies)
Comparison: Simple Baselines (Previous Day, 7-Day MA) vs Random Forest / Gradient Boosting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import json
import os

def generate_wildfire_dataset(start_date="2021-01-01", days=1095, seed=42):
    """
    Generates a realistic daily time series reflecting California wildfire dynamics:
    - Summer/Autumn peak wildfire season (July - November)
    - Extreme dry/wind events (Santa Ana / Diablo winds)
    - Satellite cloud obscuration effects (thick clouds hide fires, post-cloud clearance spike)
    - Meteorological drivers: Temperature, Humidity, Wind Speed, Precipitation
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Seasonality (peaks in day ~220 to ~290: August to October)
    day_of_year = dates.dayofyear.values
    seasonal_temp = 16 + 14 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    temp_max = seasonal_temp + np.random.normal(0, 3.5, size=days)
    temp_max = np.clip(temp_max, 5, 43)
    
    # Humidity: Inverse to temperature, occasional drops during wind events
    seasonal_rh = 55 - 25 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    rh = seasonal_rh + np.random.normal(0, 8, size=days)
    rh = np.clip(rh, 8, 90)
    
    # Precipitation: Rare in summer, frequent in winter
    rain_prob = np.where((day_of_year > 140) & (day_of_year < 290), 0.03, 0.28)
    precip_raw = np.random.exponential(scale=6.0, size=days) * (np.random.rand(days) < rain_prob)
    precip = np.round(precip_raw, 1)
    
    # Wind Speed: General baseline + sporadic high-wind offshore events
    wind_base = np.random.gamma(shape=2.5, scale=4.0, size=days)
    # Wind events common in autumn (Santa Ana / Diablo)
    wind_event_prob = np.where((day_of_year > 240) & (day_of_year < 320), 0.15, 0.04)
    wind_gusts = (np.random.rand(days) < wind_event_prob) * np.random.uniform(25, 55, size=days)
    wind_speed = np.round(wind_base + wind_gusts, 1)
    
    # Cloud cover (% of sky covered by clouds in satellite pass)
    # Winter is cloudier; summer is mostly clear
    cloud_base = 35 + 25 * np.cos(2 * np.pi * (day_of_year - 20) / 365)
    cloud_cover = np.clip(cloud_base + np.random.normal(0, 15, size=days), 0, 100)
    
    # True latent fire risk / fire count based on physical fire weather index proxy:
    # High temp, low RH, high wind, low rain
    fwi_proxy = (temp_max / 30.0) * (60.0 / (rh + 5)) * (wind_speed / 15.0)
    fwi_proxy = np.where(precip > 2.0, fwi_proxy * 0.1, fwi_proxy)
    
    # Generate latent active fires with autoregressive persistence (fires don't disappear in 1 day)
    true_active_fires = np.zeros(days)
    current_fires = 12.0
    for t in range(days):
        # Environmental ignition/spread pressure
        growth_factor = 0.82 + 0.18 * fwi_proxy[t]
        new_ignitions = np.random.poisson(lam=max(0.5, 4.0 * (fwi_proxy[t] ** 1.8) if (day_of_year[t] > 180 and day_of_year[t] < 310) else 0.5))
        current_fires = current_fires * growth_factor + new_ignitions
        if precip[t] > 5.0:
            current_fires *= 0.2  # Rain suppresses fires
        current_fires = max(0.0, current_fires + np.random.normal(0, 3))
        true_active_fires[t] = current_fires
        
    # CLOUD OBSCURATION EFFECT:
    # If cloud cover is high (>65%), satellite thermal sensors (MODIS/VIIRS) cannot see ground fires!
    # Obscuration fraction = min(0.95, (cloud_cover / 100)^1.5)
    detection_probability = np.clip(1.0 - (cloud_cover / 100.0) ** 1.6, 0.05, 1.0)
    observed_fire_count = np.random.binomial(n=np.round(true_active_fires).astype(int), p=detection_probability)
    
    df = pd.DataFrame({
        'date': dates,
        'true_latent_fires': np.round(true_active_fires).astype(int),
        'observed_fire_count': observed_fire_count,
        'cloud_cover_pct': np.round(cloud_cover, 1),
        'temp_max_c': np.round(temp_max, 1),
        'rel_humidity_pct': np.round(rh, 1),
        'wind_speed_kmh': np.round(wind_speed, 1),
        'precip_mm': np.round(precip, 1),
        'fwi_proxy': np.round(fwi_proxy, 2)
    })
    return df

def build_features(df):
    """
    Creates temporal features, rolling statistics, and time-lagged predictors
    strictly respecting causality (no future leakage).
    """
    data = df.copy()
    
    # TARGET: Next-day observed fire count (y_{t+1})
    data['target_next_day'] = data['observed_fire_count'].shift(-1)
    
    # 1. Autoregressive Lags
    data['lag1_fire_count'] = data['observed_fire_count']  # Count at day t (previous day relative to t+1)
    data['lag2_fire_count'] = data['observed_fire_count'].shift(1)
    data['lag3_fire_count'] = data['observed_fire_count'].shift(2)
    data['lag7_fire_count'] = data['observed_fire_count'].shift(6)
    
    # 2. Rolling Statistics over past observations
    data['roll_mean_3d'] = data['observed_fire_count'].rolling(window=3).mean()
    data['roll_mean_7d'] = data['observed_fire_count'].rolling(window=7).mean()
    data['roll_max_7d'] = data['observed_fire_count'].rolling(window=7).max()
    data['roll_std_7d'] = data['observed_fire_count'].rolling(window=7).std().fillna(0)
    
    # 3. Fire Count Momentum / Trend
    data['fire_count_diff_1d'] = data['lag1_fire_count'] - data['lag2_fire_count']
    data['fire_count_ratio_7d'] = (data['lag1_fire_count'] + 1.0) / (data['roll_mean_7d'] + 1.0)
    
    # 4. Weather & Remote Sensing Indicators at day t
    data['cloud_cover_pct'] = data['cloud_cover_pct']
    data['cloud_drop_flag'] = (data['cloud_cover_pct'] > 60).astype(int)
    # Cloud difference: if clouds cleared today compared to yesterday, detections may surge!
    data['cloud_delta_1d'] = data['cloud_cover_pct'] - data['cloud_cover_pct'].shift(1)
    
    data['temp_max_c'] = data['temp_max_c']
    data['rel_humidity_pct'] = data['rel_humidity_pct']
    data['wind_speed_kmh'] = data['wind_speed_kmh']
    data['precip_mm'] = data['precip_mm']
    data['dry_wind_interaction'] = data['wind_speed_kmh'] * (100.0 - data['rel_humidity_pct']) / 100.0
    
    # 5. Seasonal Calendar Features (Cyclical encoding)
    day_of_year = data['date'].dt.dayofyear
    data['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    data['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    data['month'] = data['date'].dt.month
    
    # Drop rows with NaN caused by shifting / rolling windows
    data = data.dropna().reset_index(drop=True)
    return data

def run_evaluation():
    # 1. Generate Dataset
    raw_df = generate_wildfire_dataset(start_date="2021-01-01", days=1095, seed=42)
    df = build_features(raw_df)
    
    # 2. Strict Time-Based Split: Train on 2021-2022, Held-Out Test on 2023
    train_mask = df['date'] < '2023-01-01'
    test_mask = df['date'] >= '2023-01-01'
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    features = [
        'lag1_fire_count', 'lag2_fire_count', 'lag3_fire_count', 'lag7_fire_count',
        'roll_mean_3d', 'roll_mean_7d', 'roll_max_7d', 'roll_std_7d',
        'fire_count_diff_1d', 'fire_count_ratio_7d',
        'cloud_cover_pct', 'cloud_drop_flag', 'cloud_delta_1d',
        'temp_max_c', 'rel_humidity_pct', 'wind_speed_kmh', 'precip_mm',
        'dry_wind_interaction', 'doy_sin', 'doy_cos', 'month'
    ]
    
    X_train = train_df[features]
    y_train = train_df['target_next_day']
    
    X_test = test_df[features]
    y_test = test_df['target_next_day']
    
    # 3. Baselines on Held-Out Test Set
    # Baseline 1: Persistence (Previous-Day count = lag1_fire_count)
    pred_baseline_prev = test_df['lag1_fire_count'].values
    
    # Baseline 2: 7-Day Moving Average
    pred_baseline_7d = test_df['roll_mean_7d'].values
    
    # 4. Alternative Method: Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=4, random_state=42)
    rf_model.fit(X_train, y_train)
    pred_rf = np.maximum(0, rf_model.predict(X_test))
    
    # Gradient Boosting alternative
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    gb_model.fit(X_train, y_train)
    pred_gb = np.maximum(0, gb_model.predict(X_test))
    
    # 5. Compute Metrics
    def compute_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        medae = median_absolute_error(y_true, y_pred)
        mean_y = np.mean(y_true)
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
        return {
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MedAE": round(medae, 2),
            "WAPE (%)": round(wape, 2)
        }
        
    metrics = [
        compute_metrics(y_test, pred_baseline_prev, "Baseline 1: Previous-Day (Persistence)"),
        compute_metrics(y_test, pred_baseline_7d, "Baseline 2: 7-Day Moving Average"),
        compute_metrics(y_test, pred_rf, "Alternative: Random Forest Regressor"),
        compute_metrics(y_test, pred_gb, "Alternative: Gradient Boosting Regressor")
    ]
    
    metrics_df = pd.DataFrame(metrics)
    print("=================================================================")
    print("       HELD-OUT TEST RESULTS (Year 2023, 365 Days)               ")
    print("=================================================================")
    print(metrics_df.to_string(index=False))
    print("=================================================================")
    
    # Honest Reporting Assessment:
    rf_mae = metrics_df.loc[metrics_df['Model'] == 'Alternative: Random Forest Regressor', 'MAE'].values[0]
    base_mae = metrics_df.loc[metrics_df['Model'] == 'Baseline 1: Previous-Day (Persistence)', 'MAE'].values[0]
    
    mae_diff = base_mae - rf_mae
    pct_improvement = (mae_diff / base_mae) * 100
    
    print("\n--- HONEST REPORTING & DIAGNOSTIC VERDICT ---")
    if rf_mae < base_mae:
        print(f"Outcome: Random Forest BEAT the Persistence Baseline by {mae_diff:.2f} MAE points ({pct_improvement:.1f}% reduction).")
        print("Why: Random Forest incorporates weather drivers (temperature, wind, and humidity drops) and cloud adjustments that anticipate rapid fire breakouts/suppressions before they show up in lag1 count.")
    else:
        print(f"Outcome: Random Forest did NOT beat the Persistence Baseline (Baseline MAE: {base_mae:.2f} vs RF MAE: {rf_mae:.2f}).")
        print("Why: Wildfire activity exhibits massive autocorrelation during stable periods. Simple persistence avoids regression dilution/over-smoothing.")
        
    # Feature Importances
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop 5 Most Important Features in Random Forest:")
    for feat, imp in importances.head(5).items():
        print(f"  - {feat}: {imp*100:.1f}%")
        
    # Export summary json for dashboard
    export_data = {
        "metrics": metrics,
        "feature_importances": [{"feature": k, "importance": round(v, 4)} for k, v in importances.items()],
        "test_dates": test_df['date'].dt.strftime('%Y-%m-%d').tolist(),
        "ground_truth": test_df['target_next_day'].tolist(),
        "pred_baseline_prev": [round(float(x), 1) for x in pred_baseline_prev],
        "pred_baseline_7d": [round(float(x), 1) for x in pred_baseline_7d],
        "pred_rf": [round(float(x), 1) for x in pred_rf],
        "cloud_cover": test_df['cloud_cover_pct'].tolist(),
        "temp_max": test_df['temp_max_c'].tolist()
    }
    
    export_path = os.path.join(os.path.dirname(__file__), "forecast_results.json")
    with open(export_path, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\nSaved export results to {export_path}")

if __name__ == "__main__":
    run_evaluation()
