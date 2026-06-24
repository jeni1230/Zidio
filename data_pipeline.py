"""
RetailPulse - Data Ingestion & Cleaning (F01)
================================================
Loads raw sales/customer/inventory data, validates quality, and produces
clean, analysis-ready datasets. Mirrors what Great Expectations-style
checks would assert in production (see assertions inline).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw():
    txns = pd.read_csv(RAW_DIR / "transactions.csv", parse_dates=["date"])
    customers = pd.read_csv(RAW_DIR / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(RAW_DIR / "products.csv")
    return txns, customers, products


def data_quality_report(txns: pd.DataFrame) -> dict:
    report = {
        "n_rows_raw": int(len(txns)),
        "n_duplicates": int(txns.duplicated().sum()),
        "n_missing_unit_price": int(txns["unit_price"].isna().sum()),
        "n_negative_quantity_returns": int((txns["quantity"] < 0).sum()),
        "n_null_customer_id": int(txns["customer_id"].isna().sum()),
        "date_min": str(txns["date"].min().date()),
        "date_max": str(txns["date"].max().date()),
    }
    return report


def clean_transactions(txns: pd.DataFrame) -> pd.DataFrame:
    df = txns.copy()

    # 1. Drop exact duplicates (POS double-scan)
    df = df.drop_duplicates()

    # 2. Impute missing unit_price using product-level median (fallback to category median)
    price_by_product = df.groupby("product_id")["unit_price"].median()
    df["unit_price"] = df["unit_price"].fillna(df["product_id"].map(price_by_product))
    price_by_category = df.groupby("category")["unit_price"].median()
    df["unit_price"] = df["unit_price"].fillna(df["category"].map(price_by_category))

    # 3. Recompute revenue where it disagrees with qty*price (data entry errors), keep sign for returns
    df["revenue"] = np.sign(df["quantity"]) * (df["quantity"].abs() * df["unit_price"]).round(2)

    # 4. Separate returns from sales (negative quantity = return)
    df["is_return"] = df["quantity"] < 0

    # 5. Drop rows with missing critical keys
    before = len(df)
    df = df.dropna(subset=["customer_id", "product_id", "date"])
    dropped = before - len(df)

    # 6. Type enforcement
    df["date"] = pd.to_datetime(df["date"])
    df["quantity"] = df["quantity"].astype(int)

    print(f"Cleaning: dropped {dropped} rows with missing keys, "
          f"imputed prices, recomputed revenue, flagged {df['is_return'].sum()} returns")
    return df


def validate_clean(df: pd.DataFrame):
    """Lightweight Great-Expectations-style assertions; raises if violated."""
    assert df["unit_price"].isna().sum() == 0, "unit_price still has nulls after cleaning"
    assert df["customer_id"].isna().sum() == 0, "customer_id has nulls"
    assert df["date"].between("2020-01-01", "2030-01-01").all(), "date out of plausible range"
    assert (df.loc[~df["is_return"], "quantity"] > 0).all(), "non-return rows must have positive quantity"
    print("✅ All data validation checks passed")


def main():
    txns, customers, products = load_raw()

    report = data_quality_report(txns)
    print("Data Quality Report (raw):")
    print(json.dumps(report, indent=2))

    clean_txns = clean_transactions(txns)
    validate_clean(clean_txns)

    clean_txns.to_csv(PROCESSED_DIR / "transactions_clean.csv", index=False)
    customers.to_csv(PROCESSED_DIR / "customers_clean.csv", index=False)
    products.to_csv(PROCESSED_DIR / "products_clean.csv", index=False)

    with open(PROCESSED_DIR / "data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved cleaned datasets to {PROCESSED_DIR}")
    print(f"Final transaction count: {len(clean_txns):,}")


if __name__ == "__main__":
    main()
