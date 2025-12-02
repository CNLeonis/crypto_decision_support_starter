from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "data_quality"
REPORTS.mkdir(parents=True, exist_ok=True)


def _spacing_checks(dt: pd.Series, freq_seconds: float) -> dict:
    diffs = dt.diff().dropna().dt.total_seconds()
    if diffs.empty:
        return {
            "median_step_sec": 0.0,
            "mode_step_sec": 0.0,
            "spacing_unique": 0,
            "spacing_bad_pct": 0.0,
        }
    median_step = float(diffs.median())
    mode_step = float(diffs.mode().iloc[0])
    unique_steps = int(diffs.nunique())
    tol = 0.2  # 20% tolerancji na rozjazd interwału
    bad = (diffs < freq_seconds * (1 - tol)) | (diffs > freq_seconds * (1 + tol))
    spacing_bad_pct = float(bad.mean()) if len(diffs) else 0.0
    return {
        "median_step_sec": median_step,
        "mode_step_sec": mode_step,
        "spacing_unique": unique_steps,
        "spacing_bad_pct": spacing_bad_pct,
    }


def _price_sanity(pdf: pd.DataFrame) -> dict:
    o, h, low, c = (pdf.get(k) for k in ["open", "high", "low", "close"])
    if any(x is None for x in (o, h, low, c)):
        return {}
    o = o.astype(float)
    h = h.astype(float)
    low = low.astype(float)
    c = c.astype(float)

    non_positive = int(((o <= 0) | (h <= 0) | (low <= 0) | (c <= 0)).sum())
    bad_bounds = int(((h < low) | (h < o) | (h < c) | (low > o) | (low > c)).sum())
    return {
        "price_non_positive": non_positive,
        "price_ohlc_inconsistent": bad_bounds,
    }


def analyze_file(path: Path, freq: str) -> dict:
    df = pl.read_parquet(path).sort("datetime")
    dt = pd.to_datetime(df["datetime"].to_pandas(), utc=True)

    dup = int(dt.duplicated().sum())
    full = pd.date_range(dt.min(), dt.max(), freq=freq, tz="UTC")
    missing = int(len(full.difference(pd.Index(dt))))
    zero_vol = int((df["volume"] == 0).sum())

    na_counts = {f"na_{col}": int(df[col].null_count()) for col in df.columns if col != "datetime"}
    spacing = _spacing_checks(pd.Series(dt), pd.Timedelta(freq).total_seconds())
    price_checks = _price_sanity(df.to_pandas())

    return {
        "file": path.name,
        "rows": int(df.height),
        "start": dt.min(),
        "end": dt.max(),
        "duplicates": dup,
        "missing_expected": missing,
        "zero_volume": zero_vol,
        **na_counts,
        **spacing,
        **price_checks,
    }


def save_reports(rows: list[dict], outdir: Path) -> None:
    pdf = pd.DataFrame(rows)
    csv_path = outdir / "qa_summary.csv"
    pdf.to_csv(csv_path, index=False)

    try:
        md_table = pdf.to_markdown(index=False)
    except Exception:
        # Fallback when tabulate is not installed
        md_table = pdf.to_string(index=False)
    md_lines = ["# Data QA summary", "", md_table]
    (outdir / "qa_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved QA report: {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="QA raw OHLCV data for gaps/duplicates/zero volume.")
    ap.add_argument("--raw-dir", type=Path, default=RAW, help="Folder z plikami *_1h.parquet")
    ap.add_argument(
        "--freq", type=str, default="1h", help="Oczekiwana częstotliwość (pandas offset)"
    )
    ap.add_argument("--outdir", type=Path, default=REPORTS, help="Folder na raporty (CSV/Markdown)")
    args = ap.parse_args()

    files = sorted(args.raw_dir.glob("*_*.parquet"))
    if not files:
        print(f"Brak plików w {args.raw_dir}")
        return

    rows: list[dict] = []
    for f in files:
        try:
            rows.append(analyze_file(f, freq=args.freq))
        except Exception as exc:
            rows.append({"file": f.name, "error": str(exc)})

    save_reports(rows, args.outdir)


if __name__ == "__main__":
    main()
