"""
RetailPulse - Churn Prediction (F04)
=======================================
Binary classifier identifying customers at risk of churning, defined as
no purchase in the 60 days leading up to the reference date despite an
established purchase history. Uses XGBoost + SHAP for explainability.
Target: AUC-ROC >= 0.88, precision@top20% >= 0.75.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

REFERENCE_DATE = pd.Timestamp("2026-06-19")
CHURN_WINDOW_DAYS = 60


def build_churn_features(txns: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    sales = txns[~txns["is_return"]].copy()

    # Use a "snapshot" 60 days before reference date to build features, and label
    # based on activity in the 60 days AFTER the snapshot (avoids leakage).
    snapshot_date = REFERENCE_DATE - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    hist = sales[sales["date"] <= snapshot_date]
    future_window = sales[(sales["date"] > snapshot_date) & (sales["date"] <= REFERENCE_DATE)]

    feats = hist.groupby("customer_id").agg(
        recency_days=("date", lambda x: (snapshot_date - x.max()).days),
        frequency=("transaction_id", "nunique"),
        monetary=("revenue", "sum"),
        avg_basket_value=("revenue", "mean"),
        n_categories=("category", "nunique"),
        tenure_days=("date", lambda x: (snapshot_date - x.min()).days),
        total_quantity=("quantity", "sum"),
    ).reset_index()

    # trend feature: comparing recent 90d activity vs prior 90d activity
    last90 = hist[hist["date"] > snapshot_date - pd.Timedelta(days=90)]
    prev90 = hist[(hist["date"] <= snapshot_date - pd.Timedelta(days=90)) &
                  (hist["date"] > snapshot_date - pd.Timedelta(days=180))]
    last90_cnt = last90.groupby("customer_id")["transaction_id"].nunique().rename("txns_last_90d")
    prev90_cnt = prev90.groupby("customer_id")["transaction_id"].nunique().rename("txns_prev_90d")
    feats = feats.merge(last90_cnt, on="customer_id", how="left").merge(prev90_cnt, on="customer_id", how="left")
    feats[["txns_last_90d", "txns_prev_90d"]] = feats[["txns_last_90d", "txns_prev_90d"]].fillna(0)
    feats["momentum"] = feats["txns_last_90d"] - feats["txns_prev_90d"]

    feats = feats.merge(customers[["customer_id", "age", "region", "archetype"]], on="customer_id", how="left")

    # Label: churned = no purchase at all in the future window, AND had enough
    # history to be a "real" customer (avoid mislabeling brand-new signups)
    active_in_future = set(future_window["customer_id"].unique())
    feats["churned"] = (~feats["customer_id"].isin(active_in_future)).astype(int)
    feats = feats[feats["frequency"] >= 2]  # require purchase history to be meaningful

    return feats


def train_churn_model(feats: pd.DataFrame):
    cat_cols = ["region", "archetype"]
    num_cols = ["recency_days", "frequency", "monetary", "avg_basket_value", "n_categories",
                "tenure_days", "total_quantity", "txns_last_90d", "txns_prev_90d", "momentum", "age"]

    X = pd.get_dummies(feats[num_cols + cat_cols], columns=cat_cols, drop_first=True)
    y = feats["churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=42,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    # precision@top20%
    n_top = max(1, int(len(y_test) * 0.20))
    top_idx = np.argsort(-y_proba)[:n_top]
    precision_at_20 = float(y_test.values[top_idx].mean())

    report = classification_report(y_test, (y_proba >= 0.5).astype(int), output_dict=True)

    return model, X, X_train, X_test, y_test, y_proba, auc, precision_at_20, report


def plot_roc_and_shap(model, X_train, X_test, y_test, y_proba, auc):
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, color="crimson", label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Churn Model ROC Curve")
    axes[0].legend()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1][:10]
    axes[1].barh(np.array(X_test.columns)[order][::-1], mean_abs_shap[order][::-1], color="#4C5FD5")
    axes[1].set_title("Top 10 Churn Drivers (mean |SHAP value|)")
    axes[1].set_xlabel("Mean |SHAP value| (impact on churn probability)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "churn_model_performance.png", dpi=130)
    plt.close()

    return {k: float(v) for k, v in zip(np.array(X_test.columns)[order], mean_abs_shap[order])}


def main():
    txns = pd.read_csv(PROCESSED_DIR / "transactions_clean.csv", parse_dates=["date"])
    customers = pd.read_csv(PROCESSED_DIR / "customers_clean.csv", parse_dates=["signup_date"])

    feats = build_churn_features(txns, customers)
    print(f"Churn dataset: {len(feats)} customers, churn rate = {feats['churned'].mean()*100:.1f}%")

    model, X, X_train, X_test, y_test, y_proba, auc, precision_at_20, report = train_churn_model(feats)
    print(f"AUC-ROC: {auc:.3f} (target >= 0.88)")
    print(f"Precision@top20%: {precision_at_20:.3f} (target >= 0.75)")

    top_drivers = plot_roc_and_shap(model, X_train, X_test, y_test, y_proba, auc)
    print("Top churn drivers (SHAP):", top_drivers)

    # Score the FULL current customer base for the dashboard (most-recent snapshot)
    full_X = X.copy()
    feats["churn_probability"] = model.predict_proba(full_X)[:, 1].round(4)
    feats[["customer_id", "churn_probability", "recency_days", "frequency", "monetary", "archetype"]].to_csv(
        PROCESSED_DIR / "churn_scores.csv", index=False
    )

    model.save_model(str(MODEL_DIR / "churn_xgboost.json"))

    metrics = {
        "auc_roc": round(float(auc), 4),
        "precision_at_top_20pct": round(precision_at_20, 4),
        "meets_target_auc_088": bool(auc >= 0.88),
        "meets_target_precision_075": bool(precision_at_20 >= 0.75),
        "churn_rate_pct": round(float(feats["churned"].mean() * 100), 2),
        "n_customers_scored": int(len(feats)),
        "top_churn_drivers": top_drivers,
    }
    with open(PROCESSED_DIR / "churn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
