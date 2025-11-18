from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LGBM = ROOT / "reports" / "models" / "lgbm"
RAW = ROOT / "data" / "raw"


def main(sym: str):
    rep = pd.read_csv(LGBM / sym / "calibration_report.csv")
    ts = pd.read_csv(LGBM / sym / "calibrated_eval_timeseries.csv", index_col=0, parse_dates=True)
    imp = (
        pd.read_csv(LGBM / sym / "feature_importance.csv", header=None, names=["feat", "gain"])
        .sort_values("gain", ascending=False)
        .head(20)
        if (LGBM / sym / "feature_importance.csv").exists()
        else None
    )
    print(rep)
    print(
        "Rows (eval):", len(ts), "  proba_cal μ/σ:", ts["proba_cal"].mean(), ts["proba_cal"].std()
    )
    if imp is not None:
        print("Top features:\n", imp.head(10))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    args = ap.parse_args()
    main(args.symbol)
