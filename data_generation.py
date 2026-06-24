"""
RetailPulse - Synthetic Data Generator
========================================
Generates realistic retail transaction, customer, and inventory data with:
- Seasonal demand patterns (weekly + yearly)
- Promotional spikes
- Customer behavioral diversity (loyal, occasional, churned, new)
- Category-level inventory data

This stands in for the "multiple source ingestion" the brief calls for
(POS system, CRM, warehouse management) until real client data is connected.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_CUSTOMERS = 3000
N_PRODUCTS = 60
START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2026-06-19")  # ~2.5 years, ends "today"
CATEGORIES = ["Grocery", "Apparel", "Electronics", "Home & Living", "Beauty", "Sports"]


def generate_products():
    cat_per_product = RNG.choice(CATEGORIES, size=N_PRODUCTS)
    base_price = {
        "Grocery": (3, 25), "Apparel": (10, 120), "Electronics": (25, 800),
        "Home & Living": (8, 250), "Beauty": (5, 60), "Sports": (10, 200),
    }
    rows = []
    for i in range(N_PRODUCTS):
        cat = cat_per_product[i]
        lo, hi = base_price[cat]
        price = round(RNG.uniform(lo, hi), 2)
        rows.append({
            "product_id": f"P{i+1:04d}",
            "product_name": f"{cat[:4].upper()}-{i+1:04d}",
            "category": cat,
            "unit_price": price,
            "unit_cost": round(price * RNG.uniform(0.45, 0.7), 2),
            "lead_time_days": int(RNG.integers(2, 21)),
            "current_stock": int(RNG.integers(20, 500)),
            "reorder_point": int(RNG.integers(10, 80)),
        })
    return pd.DataFrame(rows)


def generate_customers():
    signup_dates = pd.to_datetime(
        RNG.integers(START_DATE.value // 10**9, (END_DATE.value - 86400 * 30) // 10**9, N_CUSTOMERS),
        unit="s",
    )
    # Behavioral archetypes drive purchase frequency & recency patterns
    archetypes = RNG.choice(
        ["loyal", "regular", "occasional", "at_risk", "new", "dormant"],
        size=N_CUSTOMERS,
        p=[0.12, 0.28, 0.25, 0.12, 0.13, 0.10],
    )
    regions = RNG.choice(
        ["North", "South", "East", "West", "Central"], size=N_CUSTOMERS, p=[0.22, 0.24, 0.18, 0.2, 0.16]
    )
    df = pd.DataFrame({
        "customer_id": [f"C{i+1:05d}" for i in range(N_CUSTOMERS)],
        "signup_date": signup_dates,
        "archetype": archetypes,
        "region": regions,
        "age": RNG.integers(18, 70, N_CUSTOMERS),
    })
    return df


ARCHETYPE_PARAMS = {
    # mean days between purchases, recency decay (chance churned), spend multiplier
    "loyal":      dict(gap_days=7,  churn_after=None, spend_mult=1.4),
    "regular":    dict(gap_days=18, churn_after=None, spend_mult=1.0),
    "occasional": dict(gap_days=45, churn_after=None, spend_mult=0.8),
    "at_risk":    dict(gap_days=25, churn_after=120, spend_mult=0.9),  # was active, then stops
    "new":        dict(gap_days=20, churn_after=None, spend_mult=0.9),
    "dormant":    dict(gap_days=30, churn_after=60, spend_mult=0.7),   # stops early, stays churned
}


def seasonal_multiplier(date: pd.Timestamp) -> float:
    """Weekly + yearly seasonality plus known promo spikes (Diwali/holiday, summer sale)."""
    doy = date.dayofyear
    weekday_boost = 1.15 if date.weekday() >= 5 else 1.0  # weekend lift
    yearly = 1 + 0.25 * np.sin(2 * np.pi * (doy - 80) / 365)  # spring/summer lift
    promo = 1.0
    month, day = date.month, date.day
    if (month == 11 and 10 <= day <= 25) or (month == 12 and 15 <= day <= 26):  # festive/holiday season
        promo = 1.8
    elif month == 7:  # mid-year sale
        promo = 1.3
    return weekday_boost * yearly * promo


def generate_transactions(customers: pd.DataFrame, products: pd.DataFrame):
    all_days = pd.date_range(START_DATE, END_DATE, freq="D")
    seasonal = {d: seasonal_multiplier(d) for d in all_days}

    records = []
    txn_id = 1
    for _, cust in customers.iterrows():
        params = ARCHETYPE_PARAMS[cust["archetype"]]
        cur_date = cust["signup_date"] + pd.Timedelta(days=int(RNG.exponential(5)))
        is_churned_flag = False

        while cur_date <= END_DATE:
            if params["churn_after"] is not None and not is_churned_flag:
                if (cur_date - cust["signup_date"]).days > params["churn_after"] + RNG.integers(-15, 15):
                    is_churned_flag = True
                    break  # this customer stops transacting from here on

            mult = seasonal.get(cur_date.normalize(), 1.0)
            # number of items in this basket
            n_items = max(1, int(RNG.poisson(2 * params["spend_mult"] * mult * 0.6)))
            basket_products = products.sample(n=min(n_items, len(products)), random_state=int(RNG.integers(0, 1e6)))

            for _, prod in basket_products.iterrows():
                qty = max(1, int(RNG.poisson(1.5)))
                records.append({
                    "transaction_id": f"T{txn_id:07d}",
                    "customer_id": cust["customer_id"],
                    "product_id": prod["product_id"],
                    "category": prod["category"],
                    "date": cur_date.normalize(),
                    "quantity": qty,
                    "unit_price": prod["unit_price"],
                    "revenue": round(qty * prod["unit_price"], 2),
                })
                txn_id += 1

            gap = max(1, int(RNG.exponential(params["gap_days"])))
            cur_date = cur_date + pd.Timedelta(days=gap)

    txns = pd.DataFrame(records)
    return txns


def inject_data_quality_issues(txns: pd.DataFrame) -> pd.DataFrame:
    """Real retail data is messy. Inject realistic issues for the cleaning step to handle."""
    txns = txns.copy()
    n = len(txns)
    # missing unit_price for ~0.5% of rows
    missing_idx = RNG.choice(n, size=int(n * 0.005), replace=False)
    txns.loc[missing_idx, "unit_price"] = np.nan
    # a few negative quantities (returns) and zero/negative revenue glitches
    return_idx = RNG.choice(n, size=int(n * 0.01), replace=False)
    txns.loc[return_idx, "quantity"] = -txns.loc[return_idx, "quantity"]
    txns.loc[return_idx, "revenue"] = -txns.loc[return_idx, "revenue"]
    # duplicate ~0.3% of rows (POS double-scan)
    dup_idx = RNG.choice(n, size=int(n * 0.003), replace=False)
    dups = txns.loc[dup_idx]
    txns = pd.concat([txns, dups], ignore_index=True)
    return txns


def main():
    print("Generating products...")
    products = generate_products()
    print("Generating customers...")
    customers = generate_customers()
    print("Generating transactions (this models real purchase behavior, may take a minute)...")
    txns = generate_transactions(customers, products)
    txns = inject_data_quality_issues(txns)

    products.to_csv(DATA_DIR / "products.csv", index=False)
    customers.to_csv(DATA_DIR / "customers.csv", index=False)
    txns.to_csv(DATA_DIR / "transactions.csv", index=False)

    print(f"Products: {len(products):,} rows")
    print(f"Customers: {len(customers):,} rows")
    print(f"Transactions: {len(txns):,} rows")
    print(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
