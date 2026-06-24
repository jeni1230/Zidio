"""
RetailPulse - Full Pipeline Runner
=====================================
Runs the entire pipeline end-to-end, in dependency order:
  1. Data generation   (synthetic source data -> data/raw/)
  2. Data cleaning      (F01 -> data/processed/)
  3. Segmentation       (F02)
  4. Demand forecasting (F03)
  5. Churn prediction   (F04)
  6. Inventory optimization (F05)

After this completes, launch the dashboard (F06) with:
    streamlit run dashboard/app.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("Data Generation", "src/data_generation.py"),
    ("Data Ingestion & Cleaning (F01)", "src/data_pipeline.py"),
    ("Customer Segmentation (F02)", "src/segmentation.py"),
    ("Demand Forecasting (F03)", "src/forecasting.py"),
    ("Churn Prediction (F04)", "src/churn.py"),
    ("Inventory Optimization (F05)", "src/inventory.py"),
]


def main():
    print("=" * 70)
    print("RetailPulse — Full Pipeline Run")
    print("=" * 70)
    total_start = time.time()

    for name, script in STEPS:
        print(f"\n▶ {name}  ({script})")
        print("-" * 70)
        start = time.time()
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT))
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"\n❌ Step failed: {name} (exit code {result.returncode})")
            sys.exit(result.returncode)
        print(f"✅ {name} complete in {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print(f"Pipeline complete in {time.time() - total_start:.1f}s")
    print("Launch the dashboard with: streamlit run dashboard/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
