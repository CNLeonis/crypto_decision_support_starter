from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scala wszystkie oof_predictions.csv danego symbolu i łączy je z cenami,"
            " zapisując ujednolicony plik predictions_vs_price.csv."
        )
    )
    parser.add_argument("--symbol", required=True, help="Symbol, np. ADA_USDT.")
    parser.add_argument(
        "--oof-root",
        type=Path,
        default=Path("reports/models/lgbm"),
        help="Katalog z wynikami modeli (zawiera podkatalogi <symbol>/fold_*).",
    )
    parser.add_argument(
        "--price",
        type=Path,
        help="Opcjonalna ścieżka do cen (domyślnie data/raw/<symbol>_1h.parquet).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Ścieżka docelowa CSV (domyślnie reports/models/lgbm/<symbol>/predictions_vs_price.csv).",
    )
    parser.add_argument(
        "--drop-na-close",
        action="store_true",
        help="Usuń wiersze, które po joinie nie mają ceny (domyślnie zostają).",
    )
    parser.add_argument(
        "--min-date",
        help="Przytnij zakres od tej daty (np. 2021-01-01).",
    )
    parser.add_argument(
        "--max-date",
        help="Przytnij zakres do tej daty (np. 2024-12-31).",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Zapisz dodatkowo kopię w formacie Parquet.",
    )
    return parser.parse_args()


def find_oof_files(root: Path, symbol: str) -> list[Path]:
    sym_dir = root / symbol
    if not sym_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu {sym_dir}")
    files = sorted(sym_dir.glob("fold_*/oof_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"W katalogu {sym_dir} nie ma plików fold_*/oof_predictions.csv.")
    return files


def load_oof_files(files: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in files:
        fold_name = path.parent.name
        df = pd.read_csv(path)
        if "datetime" not in df.columns:
            raise ValueError(f"{path} nie zawiera kolumny 'datetime'.")
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["fold_id"] = fold_name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values("datetime")


def load_prices(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError(f"Plik {path} musi mieć kolumny datetime oraz close.")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").set_index("datetime")["close"]


def parse_date(val: str | None) -> pd.Timestamp | None:
    if not val:
        return None
    ts = pd.Timestamp(val)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def main() -> None:
    args = parse_args()
    symbol = args.symbol
    price_path = args.price or Path(f"data/raw/{symbol}_1h.parquet")
    out_path = (
        args.out
        if args.out is not None
        else Path("reports/models/lgbm") / symbol / "predictions_vs_price.csv"
    )

    oof_files = find_oof_files(args.oof_root, symbol)
    preds = load_oof_files(oof_files)

    prices = load_prices(price_path)
    merged = preds.join(prices.rename("close"), on="datetime", how="left")

    min_date = parse_date(args.min_date)
    max_date = parse_date(args.max_date)
    if min_date is not None:
        merged = merged.loc[merged["datetime"] >= min_date]
    if max_date is not None:
        merged = merged.loc[merged["datetime"] <= max_date]

    if args.drop_na_close:
        merged = merged.dropna(subset=["close"])

    merged = merged.rename(columns={"proba": "proba_up"})
    desired_cols = ["datetime", "close", "proba_up", "y_true", "pred", "fold_id"]
    if "proba_std" in merged.columns:
        desired_cols.insert(3, "proba_std")
    missing_cols = [col for col in desired_cols if col not in merged.columns]
    if missing_cols:
        raise ValueError(f"Brakuje kolumn w danych scalonych: {missing_cols}")
    merged = merged[desired_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    if args.parquet:
        merged.to_parquet(out_path.with_suffix(".parquet"), index=False)

    print(
        f"Zapisano {len(merged)} wierszy do {out_path}"
        + (" (oraz Parquet)" if args.parquet else "")
    )


if __name__ == "__main__":
    main()
