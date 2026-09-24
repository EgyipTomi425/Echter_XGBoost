"""Prediction throughput of cuML's Forest Inference Library (FIL) with GPU (CuPy) input."""

import argparse
import time
import warnings
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import xgboost as xgb
from cuml.fil import ForestInference

LAYOUTS = ("depth_first", "breadth_first", "layered")
CHUNK_SIZES = (1, 2, 4, 8, 16, 32)


def median_ms(run, repetitions=9):
    run()
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        run()
        cp.cuda.Device().synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return sorted(times)[len(times) // 2]


def compare(prediction, reference_path):
    if not reference_path.exists():
        return ""
    reference = np.fromfile(reference_path, np.float32)
    ulp = np.abs(prediction.view(np.int32).astype(np.int64) - reference.view(np.int32).astype(np.int64))
    return f" | identical to {reference_path.name}: {(prediction == reference).mean() * 100:.1f}%, max {ulp.max()} ULP"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="XGBoost JSON model path")
    parser.add_argument("input", help="CSV input path")
    parser.add_argument("output_dir", help="directory for the prediction files")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def load(layout):
        return ForestInference.load(
            args.model, model_type="xgboost_json", precision="single", layout=layout, output_type="cupy")

    start = time.perf_counter()
    models = {LAYOUTS[0]: load(LAYOUTS[0])}
    print(f"model load {(time.perf_counter() - start) * 1000:.2f} ms")
    models.update({layout: load(layout) for layout in LAYOUTS[1:]})

    features = xgb.Booster(model_file=args.model).feature_names
    base = pd.read_csv(args.input)[features].to_numpy(np.float32)
    for factor in (1, 10, 20):
        data = cp.asarray(np.ascontiguousarray(np.tile(base, (factor, 1))))
        rows = len(data)
        default_ms = median_ms(lambda: models[LAYOUTS[0]].predict(data))
        best_ms, layout, chunk_size = min(
            (median_ms(lambda: model.predict(data, chunk_size=chunk_size), 5), layout, chunk_size)
            for layout, model in models.items()
            for chunk_size in CHUNK_SIZES)
        prediction = cp.asnumpy(models[layout].predict(data, chunk_size=chunk_size)).astype(np.float32).ravel()
        prediction.tofile(output_dir / f"cuml_fil_x{factor}.bin")
        print(f"rows {rows:8d} | default {default_ms:.2f} ms {rows / default_ms / 1000:.2f} M/s"
              f" | best {best_ms:.2f} ms {rows / best_ms / 1000:.2f} M/s ({layout}, chunk {chunk_size})"
              + compare(prediction, output_dir / f"echter_x{factor}.bin"))


if __name__ == "__main__":
    main()
