# FunnelIQ — Marketing Funnel Analytics Platform

> A full-stack web application for marketing funnel analysis: conversion tracking, channel performance, drop-off identification, ML-powered lead scoring, and actionable recommendations.

---

## 📸 Features

| Module | Description |
|---|---|
| **Funnel Overview** | Visual pipeline from Visitors → Customers with stage-by-stage conversion rates and drop-off badges |
| **Channel Performance** | Side-by-side ROI, CPA, revenue and conversion rate comparison across 6 acquisition channels |
| **Trend Analysis** | 30-day time series for visitors, leads and customers + rolling 7-day conversion rate |
| **Recommendations** | Auto-generated, priority-ranked action items with estimated revenue impact |
| **What-If Simulator** | Slider-based scenario modeling — see how improving each funnel stage affects customer volume |
| **ML Predictor** | Gradient-boosted model for lead-to-customer probability + budget allocation optimizer |
| **CSV Export** | One-click download of the full funnel + channel report |

---

## 🏗️ Project Structure

```
funnel-analytics/
├── backend/
│   ├── app.py              # Flask REST API (6 endpoints)
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Single-page dashboard (Chart.js, vanilla JS)
├── model/
│   ├── funnel_model.py     # Gradient Boosting Classifier (pure Python + NumPy)
│   └── funnel_model.json   # Serialized trained model weights
└── README.md
```

---

## ⚡ Quick Start

### 1. Backend (Flask API)

```bash
cd backend
pip install -r requirements.txt
python app.py
# → Running on http://localhost:5000
```

### 2. ML Model (optional — pre-trained weights included)

```bash
cd model

# Train from scratch (generates funnel_model.json)
python funnel_model.py --train

# Run sample predictions
python funnel_model.py --predict

# Both
python funnel_model.py
```

### 3. Frontend

Open `frontend/index.html` in any browser.

- **With backend running:** Live data from Flask API
- **Without backend:** Automatically falls back to built-in demo data — the full dashboard is usable offline

---

## 🔌 API Reference

Base URL: `http://localhost:5000/api`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API health check |
| GET | `/funnel/overview` | Aggregate funnel: stages, counts, conversions, biggest drop-off |
| GET | `/funnel/channels` | Per-channel breakdown with KPIs (CPA, ROI, LTV, conv rate) |
| GET | `/funnel/timeseries` | 30-day daily data: visitors, leads, customers |
| GET | `/funnel/recommendations` | Sorted action items with estimated impact |
| POST | `/model/predict` | Budget optimizer — send `{budgets: {channel: amount}}` → predicted customers + ROI |
| POST | `/model/simulate` | What-if simulator — send `{improvements: {stage_idx: pct}}` → simulated counts |

### Example: Budget Optimizer

```bash
curl -X POST http://localhost:5000/api/model/predict \
  -H "Content-Type: application/json" \
  -d '{"budgets": {"Email": 5000, "Organic Search": 3000, "Referral": 2000}}'
```

```json
{
  "total_budget": 10000,
  "predicted_customers": 47,
  "estimated_revenue": 30550,
  "predicted_roi": 205.5,
  "breakdown": {
    "Email": {"budget": 5000, "estimated_customers": 27, "cpa": 185.2, "roi": 340.1},
    ...
  }
}
```

### Example: Scenario Simulator

```bash
curl -X POST http://localhost:5000/api/model/simulate \
  -H "Content-Type: application/json" \
  -d '{"improvements": {"1": 20, "3": 15}}'
```

---

## 🤖 ML Model

### Architecture

A from-scratch **Gradient Boosting Classifier** implemented in pure Python + NumPy (no sklearn dependency for the core model).

```
Input Features (7):
  channel_id        — acquisition source (0–5)
  days_in_funnel    — time since first touch
  pages_visited     — engagement depth
  email_opens       — nurture responsiveness
  form_fills        — intent signals
  demo_booked       — high-intent action
  budget_score      — company size proxy (0–1)

Model:
  100 Decision Stumps (single-split trees)
  Learning rate: 0.08
  Loss: Binary cross-entropy (log-loss)
  Trained on: 3,200 synthetic lead records

Output:
  Probability of conversion (0.0 – 1.0)
```

### Evaluation (test set, n=800)

| Metric | Score |
|---|---|
| Accuracy | 82.5% |
| Precision | 82.5% |
| Recall | 100.0% |
| F1 Score | 0.904 |
| AUC-ROC | 0.575 |

### Training Your Own Model

```python
from model.funnel_model import GradientBoostingClassifier, generate_dataset

X, y = generate_dataset(n=10000, seed=42)
model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05)
model.fit(X, y)
model.save("model/my_model.json")
```

### Using Real Data

Replace `generate_dataset()` with a loader for your CRM export:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("your_leads.csv")
# Required columns: channel, days_in_funnel, pages_visited,
#                   email_opens, form_fills, demo_booked,
#                   budget_score, converted (0/1)
X = df[["channel_id","days_in_funnel","pages_visited",
        "email_opens","form_fills","demo_booked","budget_score"]].values
y = df["converted"].values

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X, y)
model.save("model/production_model.json")
```

---

## 📊 Key Metrics Defined

| Metric | Formula |
|---|---|
| **Conversion Rate** | `Customers / Visitors × 100` |
| **CPA** (Cost per Acquisition) | `Total Spend / Customers` |
| **ROI** | `(Revenue − Spend) / Spend × 100` |
| **LTV** (Lifetime Value) | `Revenue / Customers` |
| **Drop-off %** | `(Stage N − Stage N+1) / Stage N × 100` |
| **MQL** | Marketing Qualified Lead (passed lead score threshold) |
| **SQL** | Sales Qualified Lead (accepted by sales team) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, Flask 3.0, Flask-CORS |
| ML Model | Pure Python + NumPy (custom gradient boosting) |
| Frontend | Vanilla HTML/CSS/JS, Chart.js 4.4 |
| Charts | Chart.js (bar, line, doughnut) |
| Deployment | Gunicorn (production WSGI) |

---

## 🚀 Production Deployment

### With Gunicorn

```bash
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### With Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t funneliq .
docker run -p 5000:5000 funneliq
```

### Environment Variables

```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://...   # for real data persistence
```

---

## 🔮 Extending with Real Data

1. **Connect your CRM** (HubSpot, Salesforce, Pipedrive) via their API
2. **Replace mock data** in `backend/app.py` `generate_funnel_data()` with SQL queries
3. **Retrain the model** on your actual conversion data
4. **Add authentication** (Flask-Login or JWT) for multi-user access
5. **Set up scheduled jobs** (APScheduler or Celery) to refresh data daily

---

## 📄 License

MIT — use freely, attribution appreciated.

---

*Built with FunnelIQ · Marketing Funnel Analytics Platform*
