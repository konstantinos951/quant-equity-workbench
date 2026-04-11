from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import time
from typing import Any

import pandas as pd

from services.fmp_client import CacheRecord, DiskCacheStore


class YFinanceClientError(RuntimeError):
    pass


class YFinanceClient:
    def __init__(self, cache_dir: Path, timeout: int = 20) -> None:
        self.timeout = timeout
        self.cache = DiskCacheStore(cache_dir)

    def get_daily_history(
        self,
        symbol: str,
        history_years: int,
        ttl_seconds: float,
        force_refresh: bool = False,
        allow_stale_on_error: bool = True,
    ) -> CacheRecord:
        cache_key = f"yfinance_daily:{symbol.upper()}:{history_years}"
        if not force_refresh:
            cached = self.cache.get(cache_key, ttl_seconds=ttl_seconds)
            if cached is not None:
                return cached

        stale = self.cache.get_any_age(cache_key)
        try:
            payload = self._fetch_payload(symbol=symbol, history_years=history_years)
        except Exception as exc:
            if stale is not None and allow_stale_on_error:
                return stale
            raise YFinanceClientError(str(exc)) from exc

        self.cache.set(cache_key, payload)
        return CacheRecord(payload=payload, fetched_at=time.time(), age_seconds=0.0, source="network")

    def _fetch_payload(self, symbol: str, history_years: int) -> list[dict[str, Any]]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise YFinanceClientError(
                "Λείπει το package yfinance. Τρέξε ξανά `pip install -r requirements.txt` για να ενεργοποιηθεί το Yahoo fallback."
            ) from exc

        end = pd.Timestamp.utcnow().tz_localize(None)
        start = end - timedelta(days=int(history_years * 370))
        ticker = yf.Ticker(symbol)
        frame = ticker.history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=True,
            timeout=self.timeout,
            raise_errors=False,
        )
        if frame is None or frame.empty:
            raise YFinanceClientError(f"Το Yahoo fallback δεν επέστρεψε usable daily history για το {symbol}.")

        working = frame.copy()
        working.index = pd.to_datetime(working.index, errors="coerce").tz_localize(None)
        working = working[working.index.notna()].sort_index()
        if working.empty or "Close" not in working.columns:
            raise YFinanceClientError(f"Το Yahoo fallback έστειλε κενό ή ελλιπές dataset για το {symbol}.")

        rows: list[dict[str, Any]] = []
        for index, row in working.iterrows():
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "date": pd.Timestamp(index).strftime("%Y-%m-%d"),
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low": row.get("Low"),
                    "close": row.get("Close"),
                    "volume": row.get("Volume"),
                }
            )
        return rows
