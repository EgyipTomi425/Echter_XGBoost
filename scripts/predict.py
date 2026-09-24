"""Reference predictions with the XGBoost Python package on the GPU.

Reads a CSV with pandas, predicts with an XGBoost JSON model, and writes the
key column, the optional target column, and the prediction, so the result can
be compared with echter_xgb_predict. The model's feature names select the
input columns.
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb


def milliseconds(start, end):
    return (end - start) * 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="XGBoost JSON model path")
    parser.add_argument("input", help="CSV input path")
    parser.add_argument(
        "--output",
        help="output CSV path (default: outputs/<input-stem>_xgboost_predictions.csv)",
    )
    parser.add_argument(
        "--key-column",
        help="identifier column written as 'id' (default: the first CSV column)",
    )
    parser.add_argument(
        "--target-column",
        help="optional column written as 'target'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or (
        f"outputs/{Path(args.input).stem}_xgboost_predictions.csv"
    )

    total_start = time.perf_counter()

    # ============================================================
    # CSV READ
    # ============================================================

    read_start = time.perf_counter()
    print("Loading CSV...")
    df = pd.read_csv(args.input)
    read_end = time.perf_counter()

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ============================================================
    # MODEL LOAD
    # ============================================================

    model_start = time.perf_counter()
    print("Loading XGBoost model...")
    model = xgb.Booster()
    model.load_model(args.model)
    model.set_param({"device": "cuda"})
    model_end = time.perf_counter()

    features = model.feature_names
    if not features:
        raise RuntimeError("the model has no feature names to select CSV columns")
    print(f"Model expects: {model.num_features()} features")

    missing = [column for column in features if column not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")

    key_column = args.key_column or df.columns[0]
    for column in (key_column, args.target_column):
        if column is not None and column not in df.columns:
            raise RuntimeError(f"Missing column: {column}")

    # ============================================================
    # DMatrix creation
    # ============================================================

    dmatrix_start = time.perf_counter()
    print("Creating DMatrix...")
    dmatrix = xgb.DMatrix(df[features], feature_names=features)
    dmatrix_end = time.perf_counter()

    # ============================================================
    # GPU PREDICTION
    # ============================================================

    predict_start = time.perf_counter()
    print("Running XGBoost prediction on GPU...")
    pred = model.predict(dmatrix)
    predict_end = time.perf_counter()

    # ============================================================
    # CSV WRITE
    # ============================================================

    out = pd.DataFrame({"id": df[key_column]})
    if args.target_column is not None:
        out["target"] = df[args.target_column]
    out["prediction"] = pred

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_start = time.perf_counter()
    out.to_csv(output_path, index=False)
    write_end = time.perf_counter()

    total_end = time.perf_counter()

    # ============================================================
    # RESULTS
    # ============================================================

    print()
    print("========================================")
    print("XGBoost GPU prediction")
    print("========================================")
    print(f"Model load ms:      {milliseconds(model_start, model_end):.3f}")
    print(f"CSV read ms:        {milliseconds(read_start, read_end):.3f}")
    print(f"DMatrix create ms:  {milliseconds(dmatrix_start, dmatrix_end):.3f}")
    print(f"Prediction ms:      {milliseconds(predict_start, predict_end):.3f}")
    print(f"CSV write ms:       {milliseconds(write_start, write_end):.3f}")
    print(f"Total ms:           {milliseconds(total_start, total_end):.3f}")
    print("----------------------------------------")
    print(f"Rows:               {len(df)}")
    print(f"Input columns:      {len(df.columns)}")
    print(f"Feature columns:    {len(features)}")
    print(f"Output:             {output_path}")
    print("========================================")


if __name__ == "__main__":
    main()
