"""
RetailPulse - Interactive Analytics Dashboard (F06)
======================================================
Streamlit app exposing: demand forecasting + what-if analysis, customer
segmentation, churn risk, and inventory optimization recommendations.
Run with: streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

st.set_page_config(page_title="RetailPulse Analytics", page_icon="📊", layout="wide")


# ----------------------------------------------------------------- cache ----
@st.cache_data
def load_data():
    txns = pd.read_csv(DATA / "transactions_clean.csv", parse_dates=["date"])
    segments = pd.read_csv(DATA / "customer_segments.csv")
    churn = pd.read_csv(DATA / "churn_scores.csv")
    forecast = pd.read_csv(DATA / "demand_forecast_30day.csv", parse_dates=["date"])
    inventory = pd.read_csv(DATA / "inventory_recommendations.csv")

    with open(DATA / "segmentation_metrics.json") as f:
        seg_metrics = json.load(f)
    with open(DATA / "forecast_metrics.json") as f:
        fc_metrics = json.load(f)
    with open(DATA / "churn_metrics.json") as f:
        churn_metrics = json.load(f)
    with open(DATA / "inventory_impact.json") as f:
        inv_metrics = json.load(f)

    return txns, segments, churn, forecast, inventory, seg_metrics, fc_metrics, churn_metrics, inv_metrics


txns, segments, churn, forecast, inventory, seg_metrics, fc_metrics, churn_metrics, inv_metrics = load_data()

# ----------------------------------------------------------------- sidebar --
st.sidebar.title("📊 RetailPulse")
st.sidebar.caption("AI-Powered Customer Analytics & Demand Forecasting")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 Demand Forecasting", "👥 Customer Segmentation",
     "⚠️ Churn Risk", "📦 Inventory Optimization"],
)
st.sidebar.divider()
st.sidebar.caption(f"Data window: {txns['date'].min().date()} → {txns['date'].max().date()}")
st.sidebar.caption(f"{len(txns):,} transactions · {segments['customer_id'].nunique():,} customers · "
                    f"{inventory['product_id'].nunique()} SKUs")
st.sidebar.caption("Built with Streamlit · Prophet · PyTorch LSTM · XGBoost · SHAP")


def kpi(col, label, value, help_text=None, delta=None):
    col.metric(label, value, delta=delta, help=help_text)


# ============================================================== OVERVIEW ====
if page == "🏠 Overview":
    st.title("RetailPulse — Executive Overview")
    st.caption("End-to-end retail analytics: demand forecasting, segmentation, churn, and inventory optimization.")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Demand Forecast MAPE", f"{fc_metrics['ensemble_mape']}%",
        help_text="Target: ≤ 12%. Lower is better.")
    kpi(c2, "Churn Model AUC-ROC", f"{churn_metrics['auc_roc']}",
        help_text="Target: ≥ 0.88")
    kpi(c3, "Stockout Reduction (sim.)", f"{inv_metrics['stockout_reduction_pct']}%",
        help_text="Forecast-driven reorder policy vs. reactive baseline")
    kpi(c4, "Customer Segments", f"{seg_metrics['n_clusters']}",
        help_text="K-Means clusters with business interpretation")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Revenue Trend")
        sales = txns[~txns["is_return"]].copy()
        monthly = sales.groupby(sales["date"].dt.to_period("M"))["revenue"].sum().reset_index()
        monthly["date"] = monthly["date"].dt.to_timestamp()
        fig = px.line(monthly, x="date", y="revenue", markers=True)
        fig.update_layout(yaxis_title="Revenue (₹)", xaxis_title=None, height=350)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Revenue by Category")
        cat_rev = sales.groupby("category")["revenue"].sum().sort_values(ascending=True).reset_index()
        fig = px.bar(cat_rev, x="revenue", y="category", orientation="h")
        fig.update_layout(xaxis_title="Revenue (₹)", yaxis_title=None, height=350)
        st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Model Performance Summary")
    summary_df = pd.DataFrame([
        {"Model": "Demand Forecasting (Prophet+LSTM)", "Metric": "MAPE",
         "Target": "≤ 12%", "Achieved": f"{fc_metrics['ensemble_mape']}%",
         "Status": "✅ Pass" if fc_metrics["meets_target_mape_12pct"] else "❌ Fail"},
        {"Model": "Churn Prediction (XGBoost)", "Metric": "AUC-ROC",
         "Target": "≥ 0.88", "Achieved": f"{churn_metrics['auc_roc']}",
         "Status": "✅ Pass" if churn_metrics["meets_target_auc_088"] else "❌ Fail"},
        {"Model": "Churn Prediction (XGBoost)", "Metric": "Precision@Top20%",
         "Target": "≥ 0.75", "Achieved": f"{churn_metrics['precision_at_top_20pct']}",
         "Status": "✅ Pass" if churn_metrics["meets_target_precision_075"] else "❌ Fail"},
        {"Model": "Inventory Optimization", "Metric": "Stockout Reduction (simulated)",
         "Target": "30–50%", "Achieved": f"{inv_metrics['stockout_reduction_pct']}%",
         "Status": "✅ Pass" if inv_metrics["meets_target_stockout_30_50pct"] else "❌ Fail"},
        {"Model": "Customer Segmentation (K-Means)", "Metric": "Number of segments",
         "Target": "6–8", "Achieved": f"{seg_metrics['n_clusters']}",
         "Status": "✅ Pass" if 6 <= seg_metrics["n_clusters"] <= 8 else "❌ Fail"},
    ])
    st.dataframe(summary_df, width='stretch', hide_index=True)

# ======================================================== FORECASTING ======
elif page == "📈 Demand Forecasting":
    st.title("Demand Forecasting")
    st.caption("Hybrid Prophet + LSTM ensemble · 30-day-ahead daily demand")

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Ensemble MAPE", f"{fc_metrics['ensemble_mape']}%")
    kpi(c2, "Prophet-only MAPE", f"{fc_metrics['prophet_only_mape']}%")
    kpi(c3, "LSTM weight in ensemble", f"{fc_metrics['ensemble_weight_lstm']}")

    sales = txns[~txns["is_return"]].copy()
    daily = sales.groupby("date")["quantity"].sum().reset_index()

    st.subheader("30-Day Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"].iloc[-60:], y=daily["quantity"].iloc[-60:],
                              name="Historical", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["forecasted_demand"],
                              name="Forecast", line=dict(color="crimson", dash="dash")))
    fig.update_layout(height=420, yaxis_title="Units sold / day", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("🔧 What-If Analysis")
    wc1, wc2 = st.columns(2)
    with wc1:
        demand_shock = st.slider("Apply demand shock (e.g. promo / disruption)", -50, 100, 0, step=5,
                                  format="%d%%")
    with wc2:
        horizon = st.slider("Forecast horizon to display (days)", 7, 30, 30)

    adj_forecast = forecast.head(horizon).copy()
    adj_forecast["adjusted_demand"] = (adj_forecast["forecasted_demand"] * (1 + demand_shock / 100)).round()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=adj_forecast["date"], y=adj_forecast["forecasted_demand"],
                               name="Base forecast", line=dict(color="gray", dash="dot")))
    fig2.add_trace(go.Scatter(x=adj_forecast["date"], y=adj_forecast["adjusted_demand"],
                               name=f"Adjusted ({demand_shock:+d}%)", line=dict(color="crimson")))
    fig2.update_layout(height=380, yaxis_title="Units / day")
    st.plotly_chart(fig2, width='stretch')

    total_base = adj_forecast["forecasted_demand"].sum()
    total_adj = adj_forecast["adjusted_demand"].sum()
    st.info(f"Base forecast total: **{total_base:,.0f} units** → Adjusted total: **{total_adj:,.0f} units** "
            f"({'+' if total_adj >= total_base else ''}{total_adj - total_base:,.0f} units over {horizon} days)")

    with st.expander("📋 View forecast table"):
        st.dataframe(adj_forecast, width='stretch', hide_index=True)
        st.download_button("Download forecast CSV", adj_forecast.to_csv(index=False),
                            "demand_forecast.csv", "text/csv")

# ======================================================== SEGMENTATION =====
elif page == "👥 Customer Segmentation":
    st.title("Customer Segmentation")
    st.caption("RFM + behavioral features → K-Means clustering, DBSCAN outlier cross-check")

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Segments identified", seg_metrics["n_clusters"])
    kpi(c2, "Silhouette score", f"{seg_metrics['silhouette_score']:.3f}")
    kpi(c3, "Outliers flagged (DBSCAN)", seg_metrics["dbscan_outliers"])

    seg_choice = st.multiselect("Filter by segment", sorted(segments["segment_name"].unique()),
                                 default=sorted(segments["segment_name"].unique()))
    filtered = segments[segments["segment_name"].isin(seg_choice)]

    col_a, col_b = st.columns(2)
    with col_a:
        counts = filtered["segment_name"].value_counts().reset_index()
        counts.columns = ["segment_name", "count"]
        fig = px.bar(counts, x="count", y="segment_name", orientation="h")
        fig.update_layout(height=380, yaxis_title=None, xaxis_title="Customers")
        st.plotly_chart(fig, width='stretch')
    with col_b:
        fig = px.scatter(filtered, x="recency_days", y="monetary", color="segment_name",
                          size="frequency", opacity=0.6, log_y=True,
                          labels={"recency_days": "Recency (days)", "monetary": "Monetary value (log)"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Segment Profiles")
    profile = filtered.groupby("segment_name").agg(
        customers=("customer_id", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_basket_value=("avg_basket_value", "mean"),
    ).round(1).sort_values("avg_monetary", ascending=False)
    st.dataframe(profile, width='stretch')

    with st.expander("📋 View customer-level segment data"):
        st.dataframe(filtered, width='stretch', hide_index=True)
        st.download_button("Download segments CSV", filtered.to_csv(index=False),
                            "customer_segments.csv", "text/csv")

# ============================================================== CHURN ======
elif page == "⚠️ Churn Risk":
    st.title("Churn Risk")
    st.caption("XGBoost classifier with SHAP explainability")

    c1, c2, c3 = st.columns(3)
    kpi(c1, "AUC-ROC", churn_metrics["auc_roc"])
    kpi(c2, "Precision @ Top 20%", churn_metrics["precision_at_top_20pct"])
    kpi(c3, "Overall churn rate", f"{churn_metrics['churn_rate_pct']}%")

    risk_threshold = st.slider("At-risk probability threshold", 0.0, 1.0, 0.5, 0.05)
    at_risk = churn[churn["churn_probability"] >= risk_threshold].sort_values(
        "churn_probability", ascending=False
    )
    st.warning(f"**{len(at_risk):,}** customers flagged at or above {risk_threshold:.0%} churn probability "
               f"— representing **₹{at_risk['monetary'].sum():,.0f}** in historical revenue at risk.")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(churn, x="churn_probability", nbins=30)
        fig.add_vline(x=risk_threshold, line_color="crimson", line_dash="dash")
        fig.update_layout(height=350, title="Churn Probability Distribution", xaxis_title="Predicted churn probability")
        st.plotly_chart(fig, width='stretch')
    with col_b:
        drivers = pd.Series(churn_metrics["top_churn_drivers"]).sort_values()
        fig = px.bar(drivers, orientation="h")
        fig.update_layout(height=350, title="Top Churn Drivers (mean |SHAP|)", showlegend=False,
                           xaxis_title="Mean |SHAP value|", yaxis_title=None)
        st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("At-Risk Customer List")
    st.dataframe(
        at_risk[["customer_id", "archetype", "churn_probability", "recency_days", "frequency", "monetary"]],
        width='stretch', hide_index=True, height=350,
    )
    st.download_button("Download at-risk customer list", at_risk.to_csv(index=False),
                        "at_risk_customers.csv", "text/csv")

# =========================================================== INVENTORY =====
elif page == "📦 Inventory Optimization":
    st.title("Inventory Optimization")
    st.caption("Reorder-point recommendations driven by 30-day demand forecast")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Critical (reorder now)", inv_metrics["products_critical_reorder"])
    kpi(c2, "Reorder soon", inv_metrics["products_reorder_soon"])
    kpi(c3, "Healthy stock", inv_metrics["products_healthy"])
    kpi(c4, "Stockout reduction (sim.)", f"{inv_metrics['stockout_reduction_pct']}%")

    status_filter = st.multiselect("Filter by status", inventory["status"].unique().tolist(),
                                    default=inventory["status"].unique().tolist())
    cat_filter = st.multiselect("Filter by category", sorted(inventory["category"].unique()),
                                 default=sorted(inventory["category"].unique()))
    filtered_inv = inventory[inventory["status"].isin(status_filter) & inventory["category"].isin(cat_filter)]

    col_a, col_b = st.columns(2)
    with col_a:
        status_counts = filtered_inv["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.pie(status_counts, names="status", values="count", hole=0.45,
                     color="status",
                     color_discrete_map={"Critical - Reorder Immediately": "#d62728",
                                          "Reorder Soon": "#ff7f0e", "Healthy": "#2ca02c"})
        fig.update_layout(height=350, title="Inventory Health")
        st.plotly_chart(fig, width='stretch')
    with col_b:
        fig = px.scatter(filtered_inv, x="days_of_supply", y="recommended_order_qty",
                          color="status", size="current_stock", hover_name="product_name",
                          color_discrete_map={"Critical - Reorder Immediately": "#d62728",
                                               "Reorder Soon": "#ff7f0e", "Healthy": "#2ca02c"})
        fig.update_layout(height=350, title="Days of Supply vs. Recommended Order Qty")
        st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Reorder Recommendations")
    st.dataframe(
        filtered_inv[["product_id", "product_name", "category", "current_stock", "days_of_supply",
                      "reorder_point_calculated", "target_stock_level", "recommended_order_qty", "status"]]
        .sort_values("days_of_supply"),
        width='stretch', hide_index=True, height=400,
    )
    st.download_button("Download inventory recommendations CSV", filtered_inv.to_csv(index=False),
                        "inventory_recommendations.csv", "text/csv")

    with st.expander("ℹ️ How this is calculated"):
        st.markdown(
            "- **Reorder point** = blended forecast/historical daily demand × lead time + safety stock\n"
            "- **Safety stock** = z(95% service level) × demand std-dev × √(lead time)\n"
            "- **Target stock level** = demand × (lead time + 14-day review cycle) + safety stock\n"
            "- Simulated against a reactive 'order only after stockout' baseline over a 90-day "
            "Monte Carlo simulation (200 runs/product) to estimate stockout reduction.\n\n"
            f"_{inv_metrics['overstock_note']}_"
        )
