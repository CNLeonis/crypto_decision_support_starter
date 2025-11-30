from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.backtest.core import CostModel, compute_strategy_returns, metrics_from_returns
from src.strategies.confidence_long_short import position_confidence_long_short

ROOT = Path(__file__).resolve().parents[2]
REPORTS_MODELS = ROOT / "reports" / "models" / "lgbm"
OUT_DIR = ROOT / "reports" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_inference(symbol: str) -> pd.DataFrame | None:
    path = REPORTS_MODELS / symbol / "inference_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime")
    df.set_index("datetime", inplace=True)
    return df


def load_config(path: Path | None) -> tuple[dict, dict, list[str]]:
    if path is None:
        cfg_path = ROOT / "configs" / "strategy_confidence_ls.yaml"
    else:
        cfg_path = path
    if not cfg_path.exists():
        return (
            {
                "p_long_enter": 0.55,
                "p_long_exit": 0.50,
                "p_short_enter": 0.55,
                "p_short_exit": 0.50,
                "max_std": 0.10,
                "stop_loss_pct": 0.04,
                "min_size": 0.1,
                "max_size": 1.0,
            },
            {},
            [],
        )
    data = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    defaults = data.get("defaults", {})
    symbols_cfg = data.get("symbols", {})
    disabled = data.get("disabled", [])
    return defaults, symbols_cfg, disabled


def run_for_symbol(symbol: str, params: dict) -> dict | None:
    df = load_inference(symbol)
    if df is None or df.empty:
        return None

    pos = position_confidence_long_short(
        df,
        p_long_enter=float(params.get("p_long_enter", 0.55)),
        p_long_exit=float(params.get("p_long_exit", 0.50)),
        p_short_enter=float(params.get("p_short_enter", 0.55)),
        p_short_exit=float(params.get("p_short_exit", 0.50)),
        max_std=float(params.get("max_std", 0.10)),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.04)),
        min_size=float(params.get("min_size", 0.1)),
        max_size=float(params.get("max_size", 1.0)),
    )

    returns = compute_strategy_returns(df["close"], pos, CostModel())
    metrics = metrics_from_returns(returns)
    metrics = {k: float(v) for k, v in metrics.items()}
    metrics.update(
        {
            "symbol": symbol,
            "variant": "confidence_long_short_v2",
            **{k: params[k] for k in params},
            "samples": len(df),
        }
    )
    return metrics


def main(
    symbols: list[str] | None = None, *, overrides: dict, config_path: Path | None = None
) -> None:
    if symbols is None:
        symbols = sorted(p.name for p in REPORTS_MODELS.glob("*_USDT") if p.is_dir())

    defaults, per_symbol_cfg, disabled = load_config(config_path)
    defaults = {**defaults, **overrides}

    rows: list[dict] = []
    for sym in symbols:
        if sym in disabled:
            continue
        params = defaults.copy()
        params.update(per_symbol_cfg.get(sym, {}))
        res = run_for_symbol(sym, params)
        if res is None:
            rows.append(
                {"symbol": sym, "variant": "confidence_long_short_v2", "error": "no inference data"}
            )
            continue
        rows.append(res)

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "metrics_confidence_long_short.csv"
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=str, default="ALL")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default configs/strategy_confidence_ls.yaml)",
    )
    ap.add_argument("--p-long-enter", type=float, help="Override p_long_enter")
    ap.add_argument("--p-long-exit", type=float, help="Override p_long_exit")
    ap.add_argument("--p-short-enter", type=float, help="Override p_short_enter")
    ap.add_argument("--p-short-exit", type=float, help="Override p_short_exit")
    ap.add_argument("--max-std", type=float, help="Override max_std")
    ap.add_argument("--stop-loss", type=float, help="Override stop_loss_pct")
    ap.add_argument("--min-size", type=float, help="Override min_size")
    ap.add_argument("--max-size", type=float, help="Override max_size")
    args = ap.parse_args()
    if args.symbols == "ALL":
        syms = None
    else:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    overrides = {}
    for key, value in [
        ("p_long_enter", args.p_long_enter),
        ("p_long_exit", args.p_long_exit),
        ("p_short_enter", args.p_short_enter),
        ("p_short_exit", args.p_short_exit),
        ("max_std", args.max_std),
        ("stop_loss_pct", args.stop_loss),
        ("min_size", args.min_size),
        ("max_size", args.max_size),
    ]:
        if value is not None:
            overrides[key] = value
    main(syms, overrides=overrides, config_path=args.config)
