"""Prediction throughput of the XGBoost Python package with host (NumPy) input."""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="XGBoost JSON model path")
    parser.add_argument("input", help="CSV input path")
    parser.add_argument("output_dir", help="directory for the prediction files")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    booster = xgb.Booster()
    booster.load_model(args.model)
    booster.set_param({"device": "cuda"})
    print(f"model load {(time.perf_counter() - start) * 1000:.2f} ms")

    base = pd.read_csv(args.input)[booster.feature_names].to_numpy(np.float32)
    for factor in (1, 10, 20):
        data = np.ascontiguousarray(np.tile(base, (factor, 1)))
        booster.inplace_predict(data)
        times = []
        for _ in range(9):
            start = time.perf_counter()
            prediction = booster.inplace_predict(data)
            times.append((time.perf_counter() - start) * 1000)
        median = sorted(times)[len(times) // 2]
        print(f"rows {len(data):8d} | host {median:.2f} ms {len(data) / median / 1000:.2f} M/s")
        prediction.astype(np.float32).tofile(output_dir / f"xgboost_python_x{factor}.bin")


if __name__ == "__main__":
    main()
