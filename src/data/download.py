from __future__ import annotations

import time
import os
from typing import Optional, List, Tuple
from datetime import datetime, timezone
import yaml
import polars as pl
import ccxt  # type: ignore


def parse_exchange_and_symbol(spec: str) -> Tuple[str, str]:
    # format: "BINANCE:BTC/USDT"
    if ":" not in spec:
        raise ValueError(f"Invalid market spec: {spec}")
    ex, sym = spec.split(":", 1)
    return ex.lower(), sym


def to_millis(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def from_iso(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def load_config(path: str = "configs/data.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def fetch_ohlcv_all(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    end_ms: Optional[int],
    limit: int,
) -> List[List]:
    all_rows: List[List] = []
    since = since_ms
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break
        all_rows.extend(candles)
        # advance
        last_ts = candles[-1][0]
        # Break if we've reached end_ms
        if end_ms is not None and last_ts >= end_ms:
            break
        since = last_ts + 1
        # rate limit friendly
        time.sleep(exchange.rateLimit / 1000.0)
        # stop if no progress
        if len(candles) < limit:
            break
    return all_rows


def main() -> None:
    cfg = load_config()
    markets = cfg.get("markets", [])
    timeframe = cfg.get("timeframe", "1h")
    start_iso = cfg.get("start", "2021-01-01")
    end_iso = cfg.get("end", None)
    limit = int(cfg.get("limit_per_call", 1000))

    start_dt = from_iso(start_iso) or datetime(2021, 1, 1, tzinfo=timezone.utc)
    end_dt = from_iso(end_iso) if end_iso else None
    since_ms = to_millis(start_dt)
    end_ms = to_millis(end_dt) if end_dt else None

    for market in markets:
        ex_name, symbol = parse_exchange_and_symbol(market)
        if ex_name != "binance":
            raise NotImplementedError("Na start obsługujemy tylko BINANCE.")
        exchange = ccxt.binance({"enableRateLimit": True})
        print(f"-> Pobieram {symbol} {timeframe} od {start_dt.isoformat()} ...")

        rows = fetch_ohlcv_all(exchange, symbol, timeframe, since_ms, end_ms, limit)
        if not rows:
            print(f"Brak danych dla {symbol}.")
            continue

        df = pl.DataFrame(
            rows,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
        ).with_columns(
            pl.from_epoch(pl.col("timestamp") // 1000, time_unit="s").alias("datetime")
        ).select(["datetime", "open", "high", "low", "close", "volume"])

        out_path = f"data/raw/{symbol.replace('/','_')}_{timeframe}.parquet"
        ensure_dir(out_path)
        df.write_parquet(out_path)
        print(f"Zapisano: {out_path} ({df.height} wierszy)")


if __name__ == "__main__":
    main()
