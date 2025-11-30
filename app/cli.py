from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> None:
    """Thin wrapper over subprocess.run with exit-code passthrough."""
    print(">", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def cmd_train(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "-m", "src.models.train_lgbm_altcoins"]
    if args.symbols and args.symbols.upper() != "ALL":
        cmd.extend(["--symbols", args.symbols])
    run_cmd(cmd)


def cmd_backtest(args: argparse.Namespace) -> None:
    if not args.skip_baseline:
        run_cmd([sys.executable, "-m", "src.backtest.run_altcoins"])

    if not args.skip_calibrated:
        run_cmd([sys.executable, "-m", "src.backtest.run_calibrated"])

    if not args.skip_confidence:
        cmd = [sys.executable, "-m", "src.backtest.run_confidence_long"]
        if args.symbols and args.symbols.upper() != "ALL":
            cmd.extend(["--symbols", args.symbols])
        run_cmd(cmd)

    if not args.skip_confidence_ls:
        cmd = [sys.executable, "-m", "src.backtest.run_confidence_long_short"]
        if args.symbols and args.symbols.upper() != "ALL":
            cmd.extend(["--symbols", args.symbols])
        run_cmd(cmd)


def cmd_tune(args: argparse.Namespace) -> None:
    scripts_dir = ROOT / "scripts"
    if args.mode == "long":
        cmd = [
            sys.executable,
            str(scripts_dir / "tune_confidence_long.py"),
            "--p-enter",
            args.p_enter,
            "--max-std",
            args.max_std,
        ]
    else:
        cmd = [
            sys.executable,
            str(scripts_dir / "tune_confidence_long_short.py"),
            "--p",
            args.p,
            "--max-std",
            args.max_std,
        ]
    if args.symbols and args.symbols.upper() != "ALL":
        cmd.extend(["--symbols", args.symbols])
    run_cmd(cmd)


def cmd_inference(args: argparse.Namespace) -> None:
    scripts_dir = ROOT / "scripts"
    sym = args.symbol.upper()

    # Predict OOS window
    cmd_predict = [
        sys.executable,
        str(scripts_dir / "predict_future_lgbm.py"),
        "--symbol",
        sym,
        "--test-size",
        str(args.test_size),
    ]
    run_cmd(cmd_predict)

    if args.skip_join:
        return

    # Join OOF and price (OOF path)
    cmd_join_oof = [
        sys.executable,
        str(scripts_dir / "build_predictions_vs_price.py"),
        "--symbol",
        sym,
    ]
    run_cmd(cmd_join_oof)

    if args.skip_plots:
        return

    oof_csv = ROOT / "reports" / "models" / "lgbm" / sym / "predictions_vs_price.csv"
    if oof_csv.exists():
        run_cmd(
            [
                sys.executable,
                str(scripts_dir / "plot_predictions_vs_price.py"),
                "--csv",
                str(oof_csv),
                "--symbol",
                sym,
            ]
        )

    infer_csv = ROOT / "reports" / "models" / "lgbm" / sym / "inference_predictions.csv"
    if infer_csv.exists():
        infer_out = infer_csv.with_name(f"{sym}_inference_pred_vs_price.png")
        run_cmd(
            [
                sys.executable,
                str(scripts_dir / "plot_predictions_vs_price.py"),
                "--csv",
                str(infer_csv),
                "--symbol",
                sym,
                "--out",
                str(infer_out),
            ]
        )


def cmd_live(_: argparse.Namespace) -> None:
    # Placeholder: live loop not yet implemented.
    print("Live inference loop not yet implemented. Use 'cli inference' for batch predictions.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m app.cli", description="Project CLI wrapper")
    sp = ap.add_subparsers(dest="command", required=True)

    p_train = sp.add_parser("train", help="Train LightGBM models")
    p_train.add_argument(
        "--symbols", type=str, default="ALL", help="Comma-separated symbols or ALL"
    )
    p_train.set_defaults(func=cmd_train)

    p_bt = sp.add_parser("backtest", help="Run backtests (baseline + calibrated + confidence)")
    p_bt.add_argument("--symbols", type=str, default="ALL", help="Comma-separated symbols or ALL")
    p_bt.add_argument("--skip-baseline", action="store_true", help="Skip baseline backtests")
    p_bt.add_argument("--skip-calibrated", action="store_true", help="Skip calibrated backtests")
    p_bt.add_argument("--skip-confidence", action="store_true", help="Skip confidence long-only")
    p_bt.add_argument(
        "--skip-confidence-ls", action="store_true", help="Skip confidence long/short backtests"
    )
    p_bt.set_defaults(func=cmd_backtest)

    p_tune = sp.add_parser("tune", help="Grid-search for confidence strategies")
    p_tune.add_argument("--symbols", type=str, default="ALL", help="Comma-separated symbols or ALL")
    p_tune.add_argument("--mode", choices=["long", "ls"], default="long", help="Strategy variant")
    p_tune.add_argument("--p-enter", dest="p_enter", default="0.52,0.54,0.55,0.57")
    p_tune.add_argument("--max-std", dest="max_std", default="0.05,0.08,0.10,0.12")
    p_tune.add_argument("--p", dest="p", default="0.52,0.54,0.55,0.56", help="For long/short")
    p_tune.set_defaults(func=cmd_tune)

    p_inf = sp.add_parser("inference", help="Run OOS inference and plotting")
    p_inf.add_argument("--symbol", required=True, help="Symbol, e.g., ADA_USDT")
    p_inf.add_argument("--test-size", type=int, default=500, help="OOS window size")
    p_inf.add_argument("--skip-join", action="store_true", help="Skip join with price/OOF")
    p_inf.add_argument("--skip-plots", action="store_true", help="Skip plotting")
    p_inf.set_defaults(func=cmd_inference)

    p_live = sp.add_parser("live", help="Live inference loop (placeholder)")
    p_live.set_defaults(func=cmd_live)

    return ap


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
