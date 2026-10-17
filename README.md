# Wildfire Next-Day Active-Fire Count Prediction

A beginner-friendly, production-grade time-series forecasting solution designed to predict next-day active-fire detection counts for the **State of California**.
🔗 **Live Demo:** [Open Interactive Dashboard](https://<your-username>.github.io/<your-repo-name>/)
---

## 📁 Project Structure

```
wildfire-prediction-project/
│
├── wildfire_forecasting.ipynb   # Complete, runnable Jupyter Notebook
├── dashboard.html               # Interactive visual dashboard (open in Chrome / any browser)
├── forecast_pipeline.py         # Standalone ML pipeline (baselines, RF, evaluation, intervals)
├── fetch_real_data.py           # Live data fetcher (Open-Meteo Weather + NASA FIRMS Satellite Active Fires)
├── requirements.txt             # Project Python dependencies
└── README.md                    # Project documentation & execution guide
```

---

## 🚀 Quick Start

### Option 1: Open the Interactive Visual Dashboard (No installation needed)
Simply open `dashboard.html` in Chrome or any browser:
- **Spatial Hotspots Map:** Explore 10 major California wildfire complexes with seasonal scrubbing and regional filters.
- **Multi-Year Forecasts:** Seamlessly switch between **2023**, **2024**, **2025**, and **2026 (Present Year)**.
- **Interactive Time-Series:** Toggle Ground Truth, Random Forest forecasts, Previous-Day persistence, 7-Day MA, 80% prediction interval bands, and cloud cover overlays.
- **Benchmark & Verdicts:** View MAE, RMSE, MedAE, and Gini feature importances.

### Option 2: Fetch Live Real-World Satellite & Weather Data
Run the automated data fetcher script:
```bash
python fetch_real_data.py 2024-01-01 2025-12-31
```
This queries the free **Open-Meteo Historical Climate API** (for max temperature, humidity, wind gusts, precipitation, and cloud cover) and connects to the **NASA FIRMS Active Fire Satellite API**, saving a combined `california_real_wildfire_dataset.csv`.

### Option 3: Run the Machine Learning Pipeline CLI
```bash
python forecast_pipeline.py
```

### Option 4: Run in Jupyter Notebook
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook wildfire_forecasting.ipynb
   ```
.
