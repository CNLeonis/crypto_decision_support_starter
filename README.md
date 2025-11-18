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
- Uruchom pełny pipeline z Windows/PowerShell: `pwsh -File scripts/run-all.ps1 -Install`
- Przydatne przełączniki: `-NoDownload`, `-SkipEDA`, `-SkipBacktest`, `-SkipTrain`, `-Symbols "BTC,ETH"`
- Skrypt zakłada wirtualne środowisko w `.venv` (wykrywa automatycznie) i zapisuje logi do `reports/logs/`.
