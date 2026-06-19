"""
Funnel Conversion ML Model
===========================
Trains a gradient-boosted classifier to predict lead-to-customer conversion
probability based on channel, stage timing, and engagement signals.

Usage:
    python model/funnel_model.py --train       # Train and save model
    python model/funnel_model.py --predict     # Run sample predictions
    python model/funnel_model.py --report      # Print evaluation report
"""

import argparse
import json
import os
import random
import numpy as np

# ─── Synthetic Dataset Generation ─────────────────────────────────────────────

CHANNELS = ["Organic Search", "Paid Ads", "Social Media", "Email", "Referral", "Direct"]
CHANNEL_IDX = {ch: i for i, ch in enumerate(CHANNELS)}

STAGE_CR = {
    "Organic Search":  [0.22, 0.60, 0.55, 0.45, 0.38],
    "Paid Ads":        [0.18, 0.52, 0.48, 0.40, 0.30],
    "Social Media":    [0.12, 0.45, 0.40, 0.32, 0.22],
    "Email":           [0.35, 0.70, 0.65, 0.55, 0.50],
    "Referral":        [0.28, 0.65, 0.60, 0.50, 0.45],
    "Direct":          [0.20, 0.58, 0.52, 0.42, 0.35],
}

def generate_dataset(n=5000, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    X, y = [], []
    for _ in range(n):
        ch = random.choice(CHANNELS)
        ch_id = CHANNEL_IDX[ch]
        days_in_funnel = random.randint(1, 90)
        pages_visited  = random.randint(1, 20)
        email_opens    = random.randint(0, 10)
        form_fills     = random.randint(0, 3)
        demo_booked    = random.randint(0, 1)
        budget_score   = random.uniform(0, 1)   # proxy for company size/intent

        # Base conversion prob from channel
        base_cr = STAGE_CR[ch][-1]
        # Boost from engagement signals
        engagement = (
            0.05 * min(pages_visited / 10, 1) +
            0.10 * min(email_opens / 5, 1) +
            0.15 * form_fills +
            0.20 * demo_booked +
            0.10 * budget_score
        )
        prob = min(base_cr + engagement + random.gauss(0, 0.05), 1.0)
        prob = max(prob, 0.0)
        converted = int(random.random() < prob)

        X.append([ch_id, days_in_funnel, pages_visited, email_opens,
                  form_fills, demo_booked, budget_score])
        y.append(converted)

    return np.array(X, dtype=float), np.array(y)

# ─── Feature Engineering ──────────────────────────────────────────────────────

FEATURE_NAMES = [
    "channel_id", "days_in_funnel", "pages_visited",
    "email_opens", "form_fills", "demo_booked", "budget_score",
]

# ─── Simple Gradient Boosted Model (pure Python, no sklearn required) ─────────

class DecisionStump:
    """Single-feature threshold split."""
    def __init__(self):
        self.feature = 0
        self.threshold = 0.0
        self.left_val = 0.0
        self.right_val = 0.0

    def fit(self, X, residuals, sample_weight=None):
        n, d = X.shape
        best_loss = float("inf")
        for feat in range(d):
            vals = np.unique(X[:, feat])
            thresholds = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals
            for thr in thresholds:
                left  = residuals[X[:, feat] <= thr]
                right = residuals[X[:, feat] >  thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                loss = np.sum(left ** 2) + np.sum(right ** 2)
                if loss < best_loss:
                    best_loss = loss
                    self.feature   = feat
                    self.threshold = thr
                    self.left_val  = float(np.mean(left))
                    self.right_val = float(np.mean(right))

    def predict(self, X):
        mask = X[:, self.feature] <= self.threshold
        out  = np.where(mask, self.left_val, self.right_val)
        return out


class GradientBoostingClassifier:
    """Minimal gradient boosting for binary classification (log-loss)."""

    def __init__(self, n_estimators=80, learning_rate=0.1):
        self.n_estimators   = n_estimators
        self.learning_rate  = learning_rate
        self.stumps = []
        self.base_score = 0.0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def fit(self, X, y):
        self.base_score = float(np.log(np.mean(y) / (1 - np.mean(y) + 1e-9)))
        F = np.full(len(y), self.base_score)
        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            residuals = y - p
            stump = DecisionStump()
            stump.fit(X, residuals)
            update = stump.predict(X)
            F += self.learning_rate * update
            self.stumps.append(stump)

    def predict_proba(self, X):
        F = np.full(len(X), self.base_score)
        for stump in self.stumps:
            F += self.learning_rate * stump.predict(X)
        proba = self._sigmoid(F)
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save(self, path):
        data = {
            "base_score": self.base_score,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "stumps": [
                {"feature": s.feature, "threshold": s.threshold,
                 "left_val": s.left_val, "right_val": s.right_val}
                for s in self.stumps
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        model = cls(n_estimators=data["n_estimators"], learning_rate=data["learning_rate"])
        model.base_score = data["base_score"]
        for s_data in data["stumps"]:
            s = DecisionStump()
            s.feature   = s_data["feature"]
            s.threshold = s_data["threshold"]
            s.left_val  = s_data["left_val"]
            s.right_val = s_data["right_val"]
            model.stumps.append(s)
        return model

# ─── Evaluation Helpers ───────────────────────────────────────────────────────

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def precision(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def roc_auc(y_true, y_score):
    """Trapezoidal AUC without sklearn."""
    thresholds = np.linspace(0, 1, 101)[::-1]
    tprs, fprs = [], []
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tprs.append(tp / pos if pos else 0)
        fprs.append(fp / neg if neg else 0)
    auc = float(np.trapezoid(tprs, fprs) if hasattr(np, "trapezoid") else np.trapz(tprs, fprs))
    return abs(auc)

# ─── Main ─────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "funnel_model.json")

def train():
    print("Generating dataset …")
    X, y = generate_dataset(n=4000, seed=1)
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    print(f"Training on {len(X_tr)} samples …")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08)
    model.fit(X_tr, y_tr)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    metrics = {
        "accuracy":  round(accuracy(y_te, y_pred), 4),
        "precision": round(precision(y_te, y_pred), 4),
        "recall":    round(recall(y_te, y_pred), 4),
        "f1":        round(f1(y_te, y_pred), 4),
        "roc_auc":   round(roc_auc(y_te, y_proba), 4),
        "test_samples": len(y_te),
        "positive_rate": round(float(np.mean(y_te)), 4),
    }
    print("\n── Evaluation Report ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")

    model.save(MODEL_PATH)

    # Save metrics alongside
    metrics_path = MODEL_PATH.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics

def sample_predict():
    if not os.path.exists(MODEL_PATH):
        print("No saved model found — training first …")
        train()

    model = GradientBoostingClassifier.load(MODEL_PATH)

    samples = [
        {"channel": "Email",         "days_in_funnel": 14, "pages_visited": 8,
         "email_opens": 6, "form_fills": 2, "demo_booked": 1, "budget_score": 0.8},
        {"channel": "Social Media",  "days_in_funnel": 60, "pages_visited": 2,
         "email_opens": 0, "form_fills": 0, "demo_booked": 0, "budget_score": 0.2},
        {"channel": "Organic Search","days_in_funnel": 30, "pages_visited": 12,
         "email_opens": 3, "form_fills": 1, "demo_booked": 1, "budget_score": 0.6},
    ]

    print("\n── Sample Predictions ─────────────────────")
    for s in samples:
        x = np.array([[
            CHANNEL_IDX[s["channel"]], s["days_in_funnel"], s["pages_visited"],
            s["email_opens"], s["form_fills"], s["demo_booked"], s["budget_score"],
        ]])
        prob = model.predict_proba(x)[0, 1]
        label = "CONVERT" if prob >= 0.5 else "CHURN"
        print(f"  [{s['channel']:<16}] prob={prob:.2%}  → {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--report",  action="store_true")
    args = parser.parse_args()

    if args.train or args.report:
        train()
    if args.predict:
        sample_predict()
    if not any(vars(args).values()):
        train()
        sample_predict()
