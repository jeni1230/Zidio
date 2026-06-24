# 📊 RetailPulse — AI-Powered Customer Analytics & Demand Forecasting Platform

End-to-end data science platform for retail: **demand forecasting**, **customer segmentation**, **churn prediction**, and **inventory optimization** — built for the Zidio Development Data Science & Analytics track (March 2026 edition).

> Retailers lose revenue to poor demand forecasting and stock mismanagement. RetailPulse uses ML to predict demand, segment customers, flag churn risk, and recommend reorder quantities — all surfaced through one interactive dashboard.

---

## 🎯 Results at a Glance

| Capability | Model | Target | Achieved |
|---|---|---|---|
| Demand Forecasting | Prophet + LSTM ensemble | MAPE ≤ 12% | **6.97%** ✅ |
| Churn Prediction | XGBoost + SHAP | AUC-ROC ≥ 0.88 | **0.917** ✅ |
| Churn Prediction | XGBoost + SHAP | Precision@Top20% ≥ 0.75 | **0.925** ✅ |
| Customer Segmentation | K-Means + DBSCAN | 6–8 segments | **6 segments**, silhouette 0.29 ✅ |
| Inventory Optimization | Reorder-point simulation | Reduce stockouts 30–50% | **96.9%** (90-day Monte Carlo sim.) ✅ |

*(Exact numbers will vary slightly on re-run since data generation uses a fixed seed but model training has some stochasticity in train/test splits.)*

---

## 🗂️ Project Structure

```
retailpulse/
├── data/
│   ├── raw/                  # Synthetic source data (products, customers, transactions)
│   └── processed/            # Cleaned data + all model outputs (CSV/JSON)
├── src/
│   ├── data_generation.py    # Synthetic retail data generator
│   ├── data_pipeline.py      # F01 - Ingestion, cleaning, data quality checks
│   ├── segmentation.py       # F02 - RFM + K-Means/DBSCAN customer segmentation
│   ├── forecasting.py        # F03 - Prophet + LSTM ensemble demand forecasting
│   ├── churn.py              # F04 - XGBoost churn classifier + SHAP explainability
│   └── inventory.py          # F05 - Reorder-point optimization + policy simulation
├── dashboard/
│   └── app.py                # F06 - Streamlit interactive analytics dashboard
├── models/                   # Saved model artifacts (churn_xgboost.json)
├── reports/figures/          # Generated charts (PNG)
├── tests/
│   └── test_pipeline.py      # Validation tests (pytest)
├── run_pipeline.py           # Runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

```bash
# 1. Set up environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run the full pipeline (generates data, cleans it, trains all models)
python run_pipeline.py            # ~80 seconds end-to-end

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

Or run each stage independently — useful while iterating on one model:

```bash
python src/data_generation.py     # synthetic sales/customer/inventory data
python src/data_pipeline.py       # F01 - cleaning & validation
python src/segmentation.py        # F02 - customer segments
python src/forecasting.py         # F03 - demand forecast
python src/churn.py               # F04 - churn scores
python src/inventory.py           # F05 - reorder recommendations
```

Run tests:
```bash
pytest tests/ -v
```

---

## 🧩 Feature Breakdown

### F01 — Data Ingestion & Cleaning
Loads synthetic POS/CRM/warehouse-style data, runs quality checks (duplicates, missing prices, return handling), imputes missing values at the product/category level, and validates the cleaned output against a set of assertions (Great-Expectations-style).

### F02 — Customer Segmentation
Builds RFM (Recency, Frequency, Monetary) features plus behavioral signals (purchase cadence variability, category breadth), scales and log-transforms skewed features, sweeps K-Means over k=4–9 by silhouette score, settles on a business-friendly cluster count (6–8), and names segments using a standard RFM taxonomy (Champions, Loyal Customers, Potential Loyalists, Needs Attention, At Risk, Hibernating/Lost). DBSCAN runs in parallel as an outlier/anomaly cross-check.

### F03 — Demand Forecasting
- **Prophet** models trend + weekly/yearly seasonality + known promotional windows (festive season, mid-year sale) as custom "holidays."
- **LSTM** (PyTorch, 2-layer, 32 hidden units) is trained on Prophet's *residuals* — it picks up nonlinear patterns the additive model misses, rather than competing with Prophet head-on.
- The two are blended with a weight tuned on a held-out 30-day validation window, then refit on the full history for the actual forward-looking 30-day forecast.
- Backtested MAPE: **6.97%** (target ≤ 12%).

### F04 — Churn Prediction
Customers are labeled churned if they made no purchase in a 60-day window following a snapshot date (with a strict snapshot/future-window split to avoid leakage). Features include recency/frequency/monetary, a 90-day-vs-prior-90-day "momentum" signal, and demographic/archetype data. XGBoost with class-weight balancing achieves **AUC 0.917** and **precision@top-20% of 0.925**. SHAP values identify `recency_days`, `total_quantity`, and recent transaction count as the top churn drivers.

### F05 — Inventory Optimization
Allocates the platform-level 30-day demand forecast down to individual SKUs by historical category/product demand share, then computes:
- **Safety stock** = z(95%) × demand std-dev × √(lead time)
- **Reorder point** = blended forecast/historical daily demand × lead time + safety stock
- **Target stock level** = demand × (lead time + 14-day review cycle) + safety stock

A 90-day, 200-run-per-SKU **Monte Carlo simulation** compares this forecast-driven (s, S) policy against a naive reactive policy (orders only placed after hitting zero stock, with the random over/under-ordering common in manual processes). Result: **96.9% fewer stockout-days**. Overstock reduction from reorder-policy tuning alone is modest (~6%) over this horizon — see the dashboard's inventory page for why, and what additional lever (markdown/clearance) would close that gap.

### F06 — Interactive Analytics Dashboard
5-page Streamlit app: Executive Overview, Demand Forecasting (with a what-if demand-shock slider), Customer Segmentation (filterable scatter + segment profiles), Churn Risk (adjustable probability threshold, SHAP driver chart, downloadable at-risk list), and Inventory Optimization (status breakdown, filterable reorder table, methodology explainer). All tables are CSV-exportable.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data processing | Pandas, NumPy |
| ML | scikit-learn (K-Means, DBSCAN), XGBoost |
| Forecasting | Prophet, PyTorch (LSTM) |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| Testing | pytest |

> **Note on the brief's full production stack** (Kubernetes, full MLflow experiment tracking server, Airflow retraining DAGs, Evidently AI drift monitoring, Prometheus/Grafana, JWT-secured APIs): this repository implements the **modeling, MLOps logic, and dashboard layers** end-to-end and runnable locally. The infra layer (container orchestration, scheduled retraining, live monitoring stack) is documented in the architecture section of the project report as the productionization path, since deploying a live Kubernetes cluster isn't reproducible inside a portfolio submission. `mlflow` is included in requirements.txt and can be wired into each `src/*.py` script with a few `mlflow.log_metric()` calls to track this exact run.

---

## 📈 Data

Since this is a portfolio/training submission rather than a live retail client integration, `src/data_generation.py` produces a **realistic synthetic dataset**: 3,000 customers with six behavioral archetypes (loyal, regular, occasional, at-risk, new, dormant), 60 products across 6 categories, and ~118K transactions spanning 2.5 years with weekly/yearly seasonality, festive-season and mid-year promotional spikes, and injected data-quality issues (duplicates, missing prices, returns) for the cleaning pipeline to handle. Swap in real POS/CRM exports by matching the same column schema in `data/raw/`.

---

## 🔭 Future Roadmap

- Wire `mlflow.log_metric()` / `log_model()` calls into each pipeline stage for full experiment tracking
- Add Evidently AI drift reports comparing weekly data snapshots
- Containerize with Docker; add a GitHub Actions workflow to run `pytest` on every push
- Extend churn model with a survival-analysis (time-to-churn) formulation
- Add markdown/clearance recommendations to close the overstock gap identified in F05

---

## 🙋 Personal Reflection

This project's most interesting engineering problem wasn't the modeling — Prophet, XGBoost, and K-Means are well-trodden tools — it was getting the **business framing right**. The inventory simulation initially "passed" its target by accident (a bug let stockouts spiral), and the first attempt at an overstock-reduction metric returned a number that looked good but didn't mean anything, since most SKUs in the synthetic data simply weren't being reordered within the simulation window. Building a correct (s, S) inventory-position policy and then being honest that overstock-reduction targets need a *complementary* lever (markdown, not just reorder timing) made the result more defensible than just tuning constants until a percentage matched the brief. That's the main practical lesson: a model that fails honestly is worth more than one that's been parameter-tuned to report a target number without the underlying mechanism actually supporting it.

---

*Zidio Development — Data Science & Analytics Domain · March 2026 Edition*
