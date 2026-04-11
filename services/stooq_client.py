from __future__ import annotations

import csv
import io
from pathlib import Path
import time

import requests

from services.fmp_client import CacheRecord, DiskCacheStore


class StooqClientError(RuntimeError):
    pass


class StooqClient:
    base_url = "https://stooq.com/q/d/l/"

    def __init__(self, cache_dir: Path, timeout: int = 20) -> None:
        self.timeout = timeout
        self.cache = DiskCacheStore(cache_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Quant Equity Workbench research@local.dev",
                "Accept": "text/csv,text/plain,*/*",
            }
        )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        normalized = symbol.strip().lower()
        if "." not in normalized:
            normalized = f"{normalized}.us"
        return normalized

    def get_daily_history(
        self,
        symbol: str,
        ttl_seconds: float,
        force_refresh: bool = False,
        allow_stale_on_error: bool = True,
    ) -> CacheRecord:
        normalized_symbol = self._normalize_symbol(symbol)
        cache_key = f"stooq_daily:{normalized_symbol}"

        if not force_refresh:
            cached = self.cache.get(cache_key, ttl_seconds=ttl_seconds)
            if cached is not None:
                return cached

        stale = self.cache.get_any_age(cache_key)

        try:
            response = self.session.get(
                self.base_url,
                params={"s": normalized_symbol, "i": "d"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            if stale is not None and allow_stale_on_error:
                return stale
            raise StooqClientError(f"Αποτυχία επικοινωνίας με το Stooq: {exc}") from exc

        if not response.ok:
            if stale is not None and allow_stale_on_error:
                return stale
            raise StooqClientError(f"Το Stooq επέστρεψε HTTP {response.status_code} για το symbol {symbol}.")

        payload = self._parse_csv_payload(response.text, symbol=symbol)
        if not payload:
            if stale is not None and allow_stale_on_error:
                return stale
            raise StooqClientError(f"Το Stooq δεν είχε usable daily history για το {symbol}.")

        self.cache.set(cache_key, payload)
        return CacheRecord(payload=payload, fetched_at=time.time(), age_seconds=0.0, source="network")

    @staticmethod
    def _parse_csv_payload(raw_csv: str, symbol: str) -> list[dict[str, str]]:
        text = (raw_csv or "").strip()
        if not text or "No data" in text:
            return []

        reader = csv.DictReader(io.StringIO(text))
        rows: list[dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            date_value = str(row.get("Date") or "").strip()
            close_value = str(row.get("Close") or "").strip()
            if not date_value or close_value in {"", "0", "N/D"}:
                continue
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "date": date_value,
                    "open": str(row.get("Open") or "").strip(),
                    "high": str(row.get("High") or "").strip(),
                    "low": str(row.get("Low") or "").strip(),
                    "close": close_value,
                    "volume": str(row.get("Volume") or "").strip(),
                }
            )
        return rows
