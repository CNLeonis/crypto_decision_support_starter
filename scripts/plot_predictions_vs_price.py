from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rysuje wykres (close vs P(up)) na podstawie predictions_vs_price.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Plik predictions_vs_price.csv (z kolumnami datetime, close, proba_up, y_true, pred, fold_id).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Ścieżka docelowego PNG. Domyślnie <csv_dir>/<symbol>_pred_vs_price.png.",
    )
    parser.add_argument(
        "--symbol", help="Symbol w tytule (np. ADA_USDT). Wykryty z CSV jeśli brak."
    )
    parser.add_argument("--min-date", help="Filtr: tylko dane >= tej dacie (np. 2021-01-01).")
    parser.add_argument("--max-date", help="Filtr: tylko dane <= tej dacie.")
    parser.add_argument("--dpi", type=int, default=150, help="Rozdzielczość pliku PNG.")
    parser.add_argument(
        "--show-folds",
        action="store_true",
        help="Oznacz kolorem tła zakresy poszczególnych foldów (jeśli kolumna fold_id istnieje).",
    )
    return parser.parse_args()


def infer_symbol(csv_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    parts = csv_path.parts
    for part in parts[::-1]:
        if "_" in part and part.upper().endswith("USDT"):
            return part
    return csv_path.stem


def parse_date(val: str | None) -> pd.Timestamp | None:
    if not val:
        return None
    ts = pd.Timestamp(val)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def load_data(path: Path, min_date: str | None, max_date: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert("UTC")
    df = df.sort_values("datetime")
    lo = parse_date(min_date)
    hi = parse_date(max_date)
    if lo is not None:
        df = df[df["datetime"] >= lo]
    if hi is not None:
        df = df[df["datetime"] <= hi]
    if df.empty:
        raise SystemExit("Brak danych po zastosowaniu filtrów.")
    return df


def plot(df: pd.DataFrame, symbol: str, out_path: Path, dpi: int, show_folds: bool) -> None:
    fig, ax_price = plt.subplots(figsize=(14, 5))
    ax_price.plot(df["datetime"], df["close"], color="tab:blue", linewidth=1.0, label="Close price")
    ax_price.set_ylabel("Close [USDT]")
    ax_price.set_title(f"{symbol} | Predykcja vs rzeczywistość")

    ax_prob = ax_price.twinx()
    ax_prob.plot(df["datetime"], df["proba_up"], color="tab:orange", linewidth=0.9, label="P(up)")
    ax_prob.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("Probability (up move)")

    correct = df["pred"].astype(int) == df["y_true"].astype(int)
    ax_prob.scatter(
        df.loc[correct, "datetime"],
        df.loc[correct, "proba_up"],
        color="tab:green",
        s=10,
        label="correct",
        alpha=0.75,
    )
    ax_prob.scatter(
        df.loc[~correct, "datetime"],
        df.loc[~correct, "proba_up"],
        color="tab:red",
        s=10,
        label="error",
        alpha=0.75,
    )

    fold_handles = []
    if show_folds and "fold_id" in df.columns:
        from matplotlib.patches import Patch

        fold_series = df["fold_id"].astype(str)
        cmap = plt.get_cmap("tab20")
        fold_ids = {name: idx for idx, name in enumerate(pd.unique(fold_series))}
        span_start = df["datetime"].iloc[0]
        current_fold = fold_series.iloc[0]
        for ts, fold in zip(df["datetime"], fold_series):
            if fold != current_fold:
                color = cmap(fold_ids[current_fold] % cmap.N)
                ax_price.axvspan(span_start, ts, alpha=0.08, color=color)
                span_start = ts
                current_fold = fold
        color = cmap(fold_ids[current_fold] % cmap.N)
        ax_price.axvspan(span_start, df["datetime"].iloc[-1], alpha=0.08, color=color)
        fold_handles = [
            Patch(facecolor=cmap(fold_ids[name] % cmap.N), alpha=0.3, label=name)
            for name in pd.unique(fold_series)
        ]

    acc = (correct.sum() / len(df)) * 100.0
    ax_price.text(
        0.01,
        0.02,
        f"samples: {len(df)} | accuracy: {acc:.2f}%",
        transform=ax_price.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    lines1, labels1 = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_prob.get_legend_handles_labels()
    legend_handles = lines1 + lines2 + fold_handles
    legend_labels = labels1 + labels2 + [h.get_label() for h in fold_handles]
    if legend_handles:
        ax_price.legend(legend_handles, legend_labels, loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.csv, args.min_date, args.max_date)
    symbol = infer_symbol(args.csv, args.symbol)
    out_path = args.out if args.out is not None else args.csv.parent / f"{symbol}_pred_vs_price.png"
    plot(df=df, symbol=symbol, out_path=out_path, dpi=args.dpi, show_folds=args.show_folds)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
