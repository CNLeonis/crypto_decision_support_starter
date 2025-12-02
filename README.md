# Crypto Decision Support (starter)

Starter repo for a crypto decision-support system (ML + backtests + Streamlit). Contains data loaders, LightGBM training, calibration, confidence-based strategies, and a thin CLI wrapper.

## Requirements
- Python 3.11
- git, pre-commit

## Quick start (uv)
```bash
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
pre-commit install
```

## Poetry (alternative)
```bash
pipx install poetry
poetry install
pre-commit install
```

## Data download (Binance via CCXT)
Edit `configs/data.yaml` (markets + timeframes) and run:
```bash
python -m src.data.download
```
Default timeframes: 3m, 5m, 15m, 1h, 4h, 12h, 1d. Files are saved to `data/raw/<SYMBOL>_<tf>.parquet`.
Note: multi-timeframe download loops per symbol and timeframe; respects CCXT rate limits.

## Runner (PowerShell)
- Full pipeline: `pwsh -File scripts/run-all.ps1 -Install` or `.\scripts\run-all.ps1 -Install`
- Switches: `-NoDownload`, `-SkipEDA`, `-SkipBacktest`, `-SkipTrain`, `-SkipInference`, `-Symbols "BTC_USDT,ETH_USDT"`
- After training, pipeline runs inference (`scripts/predict_future_lgbm.py`), joins/plots predictions, then runs confidence backtests (long + long/short).
Use `-SkipInference` if you only need backtests; logs go to `reports/logs/`.

### Tuning confidence-long
```bash
python scripts/tune_confidence_long.py --p-enter "0.54,0.55,0.56" --max-std "0.08,0.1,0.12"
python -m src.backtest.run_confidence_long
python -m src.backtest.run_confidence_long_short --symbols ADA_USDT --p-long-enter 0.55 --p-short-enter 0.55
```
Outputs land in `reports/backtest/metrics_confidence_long*.csv`. Streamlit uses these to show strategy metrics.
Grid results (`*_grid.csv`) and per-symbol best (`*_best.csv`) live in `reports/backtest/`.

## CLI (thin wrapper)
```bash
python -m app.cli train --symbols ADA_USDT,ETH_USDT
python -m app.cli backtest --symbols ALL
python -m app.cli tune --mode long --p-enter "0.54,0.56" --max-std "0.08,0.1"
python -m app.cli tune --mode ls --p "0.52,0.54,0.56" --max-std "0.08,0.1"
python -m app.cli inference --symbol ADA_USDT
python -m app.cli data-qa --freq 1h          # QA (gaps/dupes/volume/spacing/ohlc sanity)
# python -m app.cli live   # placeholder
```
Tip: use the venv binary explicitly on Windows, e.g. `.\.venv\Scripts\python.exe -m app.cli data-qa`.

## Project layout
- `configs/` – data/model/strategy configs.
- `data/` – raw Parquet data.
- `reports/` – EDA, backtests, model artifacts (`reports/models/lgbm`), logs, data QA.
- `src/` – code (data, features, models, strategies, backtest).
- `scripts/` – helper scripts (runner, tuning, plotting).
- `app/` – Streamlit + CLI.
- `tests/` – unit tests for core logic.

## Market assumptions (backtest/inference)
- Trading costs: taker + slippage in bps (default 7.5 + 2.0 per leg). No funding/borrow costs; assume 1x.
- Liquidity/impact: no impact model or volume limits; assume fill at close with specified bps cost.
- Position limits: strategies use min/max size from strategy configs; no portfolio-level exposure cap.
- Data scope: Binance OHLCV (default start 2021-01-01), multi-timeframe as per `configs/data.yaml`.
- Limitations: no impact model, no funding/borrow, no explicit liquidity checks; backtest assumes fills at close with costs.
For shorter timeframes (3m/5m/15m), consider higher slippage/taker costs and adjusted feature windows/embargo in walk-forward.
