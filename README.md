# Crypto Decision Support (starter)

Ambitny projekt: system wspomagania decyzji inwestycyjnych na rynku kryptowalut.
Ten starter zawiera szkielet repo, konfigurację formatowania i minimalny loader danych.

## Wymagania

- Python 3.11
- git, pre-commit

## Szybki start z **uv**

```bash
# w katalogu projektu
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
pre-commit install
```

## Alternatywnie: Poetry

```bash
pipx install poetry
poetry install
pre-commit install
```

## Pobieranie danych OHLCV (Binance, 1h)

Edytuj `configs/data.yaml` i uruchom:

```bash
python -m src.data.download
```

Pliki zostaną zapisane do `data/raw/` w formacie Parquet.

## Runner (PowerShell)

- Uruchom pełny pipeline z Windows/PowerShell: `pwsh -File scripts/run-all.ps1 -Install` lub `.\scripts\run-all.ps1 -Install`
- Przydatne przełączniki: `-NoDownload`, `-SkipEDA`, `-SkipBacktest`, `-SkipTrain`, `-SkipInference`, `-Symbols "BTC_USDT,ETH_USDT"` (podawaj symbol w formacie jak w nazwach plików z `data/raw/`)
- Domyślnie pipeline po treningu uruchamia też inferencję (`scripts/predict_future_lgbm.py`), scala prognozy (`scripts/build_predictions_vs_price.py`) i generuje wykresy (`scripts/plot_predictions_vs_price.py`) - `-SkipInference` pomija ten blok, jeżeli potrzebujesz tylko backtestów.
- Po wygenerowaniu predykcji pipeline liczy dodatkową strategię long-only z filtrem pewności (`src/backtest/run_confidence_long.py`), a wyniki zapisuje w `reports/backtest/metrics_confidence_long.csv`. Parametry strategii można ustawić w `configs/strategy_confidence.yaml` lub nadpisać z CLI (np. `python -m src.backtest.run_confidence_long --p-enter 0.54`).

### Strojenie strategii confidence-long

Przydatne polecenia:

```bash
# szybkie grid-search progów
python scripts/tune_confidence_long.py --p-enter "0.54,0.55,0.56" --max-std "0.08,0.1,0.12"

# przeliczenie strategii po edycji configs/strategy_confidence.yaml
python -m src.backtest.run_confidence_long

# long/short (V2) z konfigiem configs/strategy_confidence_ls.yaml
python -m src.backtest.run_confidence_long_short --symbols ADA_USDT --p-long-enter 0.55 --p-short-enter 0.55
```

Wyniki trafią odpowiednio do `reports/backtest/metrics_confidence_long_grid.csv` oraz `reports/backtest/metrics_confidence_long.csv`. Streamlit (sekcja “Confidence-long strategy metrics”) korzysta z tych danych, by pokazać skuteczność strategii dla wybranego symbolu.
- Skrypt zakłada wirtualne środowisko w `.venv` (wykrywa automatycznie) i zapisuje logi do `reports/logs/`.

## CLI (cienka nakładka)

Zamiast wielu skryptów możesz użyć prostego CLI:

```bash
python -m app.cli train --symbols ADA_USDT,ETH_USDT   # trening
python -m app.cli backtest --symbols ALL              # backtest (baseline + calibrated + confidence)
python -m app.cli tune --mode long --p-enter "0.54,0.56" --max-std "0.08,0.1"
python -m app.cli tune --mode ls --p "0.52,0.54,0.56" --max-std "0.08,0.1"
python -m app.cli inference --symbol ADA_USDT         # inferencja + join + wykresy
# python -m app.cli live   # placeholder na pętlę live
```

## Struktura katalogów

- `configs/` – YAML z danymi, modelami, strategiami (confidence long/ls).
- `data/` – dane surowe (Parquet).
- `reports/` – artefakty (EDA, backtest, modele LightGBM w `reports/models/lgbm`, logi).
- `src/` – logika (dane, feature’y, modele, strategie, backtest).
- `scripts/` – skrypty pomocnicze (runner, tuning, wykresy) – opcjonalnie wołane przez CLI.
- `app/` – Streamlit + CLI.
- `tests/` – (do uzupełnienia) testy jednostkowe core.
