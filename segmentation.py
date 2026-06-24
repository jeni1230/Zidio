"""
RetailPulse - Customer Segmentation (F02)
============================================
RFM (Recency, Frequency, Monetary) + behavioral feature engineering,
followed by K-Means clustering (primary) and DBSCAN (anomaly/outlier
cross-check), with business-friendly segment naming.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_DATE = pd.Timestamp("2026-06-19")


def build_rfm(txns: pd.DataFrame) -> pd.DataFrame:
    sales = txns[~txns["is_return"]].copy()

    rfm = sales.groupby("customer_id").agg(
        recency_days=("date", lambda x: (REFERENCE_DATE - x.max()).days),
        frequency=("transaction_id", "nunique"),
        monetary=("revenue", "sum"),
        avg_basket_value=("revenue", "mean"),
        n_categories=("category", "nunique"),
        tenure_days=("date", lambda x: (REFERENCE_DATE - x.min()).days),
    ).reset_index()

    def gap_std(dates):
        d = np.sort(dates.values)
        if len(d) < 3:
            return 0.0
        gaps = np.diff(d).astype("timedelta64[D]").astype(float)
        return float(np.std(gaps))

    cadence = sales.groupby("customer_id")["date"].apply(gap_std).rename("purchase_gap_std")
    rfm = rfm.merge(cadence, on="customer_id")

    return rfm


def rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    df = rfm.copy()
    df["R_score"] = pd.qcut(df["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["F_score"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["M_score"] = pd.qcut(df["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["RFM_score"] = df["R_score"] + df["F_score"] + df["M_score"]
    return df


def cluster_customers(rfm: pd.DataFrame, k=7):
    features = ["recency_days", "frequency", "monetary", "avg_basket_value",
                "n_categories", "purchase_gap_std"]
    X = rfm[features].copy()
    for col in ["monetary", "frequency", "avg_basket_value"]:
        X[col] = np.log1p(X[col].clip(lower=0))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sil_scores = {}
    for k_try in range(4, 10):
        km = KMeans(n_clusters=k_try, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores[k_try] = silhouette_score(X_scaled, labels)

    best_k = max(sil_scores, key=sil_scores.get)
    best_k = min(max(best_k, 6), 8)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_scaled)
    final_sil = silhouette_score(X_scaled, rfm["cluster"])

    dbscan = DBSCAN(eps=1.5, min_samples=10)
    rfm["dbscan_label"] = dbscan.fit_predict(X_scaled)
    n_outliers = int((rfm["dbscan_label"] == -1).sum())

    return rfm, sil_scores, best_k, final_sil, n_outliers


def name_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Rank clusters relative to each other (not absolute medians) so each
    cluster gets a distinct, business-interpretable name even when the
    underlying distributions are skewed."""
    summary = rfm.groupby("cluster").agg(
        recency_days=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        size=("customer_id", "count"),
    )
    k = len(summary)
    summary["recency_rank"] = summary["recency_days"].rank(method="first").astype(int)
    summary["freq_rank"] = summary["frequency"].rank(ascending=False, method="first").astype(int)
    summary["monetary_rank"] = summary["monetary"].rank(ascending=False, method="first").astype(int)

    summary["composite_rank"] = (summary["recency_rank"] + summary["freq_rank"] + summary["monetary_rank"])
    order = summary["composite_rank"].rank(method="first").astype(int)  # 1 = best overall, k = worst overall

    # Standard RFM segment taxonomy, ordered from highest to lowest overall engagement/value.
    # This avoids fragile tie-breaking on individual R/F/M dimensions and matches
    # conventional retail-analytics naming used in practice.
    taxonomy = [
        "Champions", "Loyal Customers", "Potential Loyalists",
        "Needs Attention", "At Risk", "Hibernating / Lost",
        "Dormant", "Lost",
    ]
    rank_to_name = {i + 1: taxonomy[i] if i < len(taxonomy) else f"Segment {i+1}" for i in range(k)}
    summary["segment_name"] = order.map(rank_to_name)

    mapping = summary["segment_name"].to_dict()
    rfm["segment_name"] = rfm["cluster"].map(mapping)
    return rfm, summary


def plot_segments(rfm: pd.DataFrame, summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    counts = rfm["segment_name"].value_counts()
    axes[0].barh(counts.index, counts.values, color="#4C5FD5")
    axes[0].set_title("Customer Count by Segment")
    axes[0].set_xlabel("Number of Customers")
    axes[0].invert_yaxis()

    axes[1].scatter(
        rfm["recency_days"], np.log1p(rfm["monetary"]),
        c=rfm["cluster"], cmap="tab10", alpha=0.5, s=15
    )
    axes[1].set_xlabel("Recency (days since last purchase)")
    axes[1].set_ylabel("log(1 + Monetary Value)")
    axes[1].set_title("Segments: Recency vs Monetary Value")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "customer_segments.png", dpi=130)
    plt.close()


def main():
    txns = pd.read_csv(PROCESSED_DIR / "transactions_clean.csv", parse_dates=["date"])

    rfm = build_rfm(txns)
    rfm = rfm_scores(rfm)
    rfm, sil_scores, best_k, final_sil, n_outliers = cluster_customers(rfm, k=7)
    rfm, summary = name_segments(rfm)

    plot_segments(rfm, summary)

    print(f"Chose k={best_k} clusters (silhouette={final_sil:.3f})")
    print(f"DBSCAN flagged {n_outliers} outlier customers ({n_outliers/len(rfm)*100:.1f}%)")
    print("\nSegment summary:")
    print(summary[["recency_days", "frequency", "monetary", "segment_name"]].round(1).to_string())
    print("\nSegment sizes:")
    print(rfm["segment_name"].value_counts().to_string())

    rfm.to_csv(PROCESSED_DIR / "customer_segments.csv", index=False)

    metrics = {
        "n_clusters": int(best_k),
        "silhouette_score": float(final_sil),
        "dbscan_outliers": int(n_outliers),
        "segment_sizes": rfm["segment_name"].value_counts().to_dict(),
        "silhouette_sweep": {str(k): float(v) for k, v in sil_scores.items()},
    }
    with open(PROCESSED_DIR / "segmentation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved segments to {PROCESSED_DIR / 'customer_segments.csv'}")
    print(f"Saved chart to {FIG_DIR / 'customer_segments.png'}")


if __name__ == "__main__":
    main()
