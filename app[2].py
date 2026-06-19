"""
Marketing Funnel Analytics Backend API
Flask REST API serving funnel data, ML predictions, and recommendations.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import random
import math
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ─── Simulated Funnel Data Store ──────────────────────────────────────────────

CHANNELS = ["Organic Search", "Paid Ads", "Social Media", "Email", "Referral", "Direct"]

def generate_funnel_data(days=30, seed=42):
    random.seed(seed)
    stages = ["Visitors", "Leads", "MQLs", "SQLs", "Opportunities", "Customers"]
    
    channel_data = {}
    for ch in CHANNELS:
        base_visitors = random.randint(800, 5000)
        rates = {
            "Organic Search":  [1.0, 0.22, 0.60, 0.55, 0.45, 0.38],
            "Paid Ads":        [1.0, 0.18, 0.52, 0.48, 0.40, 0.30],
            "Social Media":    [1.0, 0.12, 0.45, 0.40, 0.32, 0.22],
            "Email":           [1.0, 0.35, 0.70, 0.65, 0.55, 0.50],
            "Referral":        [1.0, 0.28, 0.65, 0.60, 0.50, 0.45],
            "Direct":          [1.0, 0.20, 0.58, 0.52, 0.42, 0.35],
        }
        r = rates[ch]
        counts = [int(base_visitors * r[i]) for i in range(len(stages))]
        channel_data[ch] = {
            "stages": stages,
            "counts": counts,
            "cost": random.randint(500, 8000),
            "revenue": counts[-1] * random.randint(200, 1200),
        }

    # Aggregate funnel
    agg = [sum(channel_data[ch]["counts"][i] for ch in CHANNELS) for i in range(len(stages))]

    # Time series (last N days)
    time_series = []
    base_date = datetime.now() - timedelta(days=days)
    for d in range(days):
        date = base_date + timedelta(days=d)
        noise = random.uniform(0.85, 1.15)
        time_series.append({
            "date": date.strftime("%Y-%m-%d"),
            "visitors": int(agg[0] / days * noise),
            "leads": int(agg[1] / days * noise),
            "customers": int(agg[5] / days * noise),
        })

    return {
        "stages": stages,
        "aggregate": agg,
        "channels": channel_data,
        "time_series": time_series,
    }

DATA = generate_funnel_data()

# ─── Helper: Conversion Metrics ───────────────────────────────────────────────

def compute_conversions(counts, stages):
    conversions = []
    for i in range(1, len(counts)):
        rate = (counts[i] / counts[i - 1] * 100) if counts[i - 1] > 0 else 0
        drop = counts[i - 1] - counts[i]
        conversions.append({
            "from": stages[i - 1],
            "to": stages[i],
            "rate": round(rate, 2),
            "drop": drop,
            "drop_pct": round(100 - rate, 2),
        })
    return conversions

def channel_kpis(ch_data):
    counts = ch_data["counts"]
    cost   = ch_data["cost"]
    rev    = ch_data["revenue"]
    cust   = counts[-1] if counts[-1] > 0 else 1
    return {
        "cpa":  round(cost / cust, 2),
        "roi":  round((rev - cost) / cost * 100, 2),
        "ltv":  round(rev / cust, 2),
        "conversion_rate": round(counts[-1] / counts[0] * 100, 3) if counts[0] > 0 else 0,
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

@app.route("/api/funnel/overview")
def funnel_overview():
    stages  = DATA["stages"]
    agg     = DATA["aggregate"]
    convs   = compute_conversions(agg, stages)
    overall_cr = round(agg[-1] / agg[0] * 100, 3) if agg[0] > 0 else 0

    # Biggest drop-off
    biggest_drop = max(convs, key=lambda x: x["drop_pct"])

    return jsonify({
        "stages": stages,
        "counts": agg,
        "conversions": convs,
        "overall_conversion_rate": overall_cr,
        "biggest_dropoff": biggest_drop,
        "total_visitors": agg[0],
        "total_customers": agg[-1],
    })

@app.route("/api/funnel/channels")
def funnel_channels():
    result = {}
    for ch, ch_data in DATA["channels"].items():
        result[ch] = {
            **ch_data,
            "conversions": compute_conversions(ch_data["counts"], ch_data["stages"]),
            "kpis": channel_kpis(ch_data),
        }
    return jsonify(result)

@app.route("/api/funnel/timeseries")
def funnel_timeseries():
    return jsonify(DATA["time_series"])

@app.route("/api/funnel/recommendations")
def recommendations():
    stages = DATA["stages"]
    agg    = DATA["aggregate"]
    convs  = compute_conversions(agg, stages)

    recs = []
    for c in convs:
        if c["drop_pct"] > 50:
            recs.append({
                "priority": "High",
                "stage": f"{c['from']} → {c['to']}",
                "drop_pct": c["drop_pct"],
                "issue": f"Only {c['rate']}% convert from {c['from']} to {c['to']}",
                "recommendation": _recommend(c["from"], c["to"], c["rate"]),
                "impact_estimate": f"+{round(c['drop'] * 0.15)} customers if 15% improvement",
            })
        elif c["drop_pct"] > 30:
            recs.append({
                "priority": "Medium",
                "stage": f"{c['from']} → {c['to']}",
                "drop_pct": c["drop_pct"],
                "issue": f"{c['rate']}% conversion from {c['from']} to {c['to']}",
                "recommendation": _recommend(c["from"], c["to"], c["rate"]),
                "impact_estimate": f"+{round(c['drop'] * 0.10)} customers if 10% improvement",
            })

    # Best channel
    ch_rois = {ch: channel_kpis(d)["roi"] for ch, d in DATA["channels"].items()}
    best_ch = max(ch_rois, key=ch_rois.get)
    recs.insert(0, {
        "priority": "Quick Win",
        "stage": "Channel Mix",
        "drop_pct": 0,
        "issue": "Budget not optimally distributed",
        "recommendation": f"Increase budget allocation to '{best_ch}' — highest ROI at {ch_rois[best_ch]:.0f}%",
        "impact_estimate": "Estimated 8–12% overall conversion lift",
    })

    return jsonify(sorted(recs, key=lambda x: {"High": 0, "Quick Win": 1, "Medium": 2}.get(x["priority"], 3)))

def _recommend(from_stage, to_stage, rate):
    tips = {
        ("Visitors", "Leads"):       "Add exit-intent popups, improve CTA copy, A/B test landing page headlines.",
        ("Leads", "MQLs"):           "Implement lead scoring. Nurture with educational drip campaigns.",
        ("MQLs", "SQLs"):            "Shorten qualification form. Add demo booking widget on pricing page.",
        ("SQLs", "Opportunities"):   "Improve SDR response time. Personalize outreach with account context.",
        ("Opportunities", "Customers"): "Offer free trial extension. Address objections with case studies.",
    }
    return tips.get((from_stage, to_stage), "Review content and CTA alignment for this stage.")

@app.route("/api/model/predict", methods=["POST"])
def predict():
    """
    Simple ML-style prediction endpoint.
    Accepts channel budget allocation and predicts estimated customers.
    """
    body = request.get_json(force=True)
    budgets = body.get("budgets", {})  # {channel: budget}

    total_budget = sum(budgets.values()) or 1
    predicted_customers = 0
    breakdown = {}

    for ch, budget in budgets.items():
        ch_data = DATA["channels"].get(ch)
        if not ch_data:
            continue
        kpis = channel_kpis(ch_data)
        cpa  = kpis["cpa"] if kpis["cpa"] > 0 else 100
        est_customers = round(budget / cpa)
        breakdown[ch] = {
            "budget": budget,
            "estimated_customers": est_customers,
            "cpa": cpa,
            "roi": kpis["roi"],
        }
        predicted_customers += est_customers

    return jsonify({
        "total_budget": total_budget,
        "predicted_customers": predicted_customers,
        "estimated_revenue": predicted_customers * 650,  # avg LTV
        "predicted_roi": round((predicted_customers * 650 - total_budget) / total_budget * 100, 2),
        "breakdown": breakdown,
    })

@app.route("/api/model/simulate", methods=["POST"])
def simulate():
    """Simulate conversion improvement scenarios."""
    body = request.get_json(force=True)
    improvements = body.get("improvements", {})  # {stage_key: pct_improvement}

    stages = DATA["stages"]
    agg    = DATA["aggregate"][:]
    base_customers = agg[-1]

    for stage_key, pct in improvements.items():
        try:
            idx = int(stage_key)
            if 0 < idx < len(agg):
                agg[idx] = min(agg[idx - 1], int(agg[idx] * (1 + pct / 100)))
                # Propagate downstream
                for j in range(idx + 1, len(agg)):
                    ratio = DATA["aggregate"][j] / DATA["aggregate"][j - 1] if DATA["aggregate"][j - 1] else 0
                    agg[j] = int(agg[j - 1] * ratio)
        except (ValueError, IndexError):
            pass

    new_customers = agg[-1]
    lift = new_customers - base_customers

    return jsonify({
        "stages": stages,
        "baseline_counts": DATA["aggregate"],
        "simulated_counts": agg,
        "baseline_customers": base_customers,
        "simulated_customers": new_customers,
        "customer_lift": lift,
        "revenue_impact": lift * 650,
    })

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
