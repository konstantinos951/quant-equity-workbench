from __future__ import annotations

from io import StringIO
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests

from services.fmp_client import CacheRecord, DiskCacheStore


class TreasuryClientError(RuntimeError):
    pass


class TreasuryClient:
    base_url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView"

    def __init__(self, cache_dir: Path, timeout: int = 25) -> None:
        self.timeout = timeout
        self.cache = DiskCacheStore(cache_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Quant Equity Workbench research@local.dev",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

    def get_latest_us_10y_rate(
        self,
        ttl_seconds: float,
        force_refresh: bool = False,
        allow_stale_on_error: bool = True,
    ) -> CacheRecord:
        cache_key = "treasury_us_10y_latest"
        if not force_refresh:
            cached = self.cache.get(cache_key, ttl_seconds=ttl_seconds)
            if cached is not None:
                return cached

        stale = self.cache.get_any_age(cache_key)

        month_candidates = [
            pd.Timestamp.utcnow().strftime("%Y%m"),
            (pd.Timestamp.utcnow() - pd.DateOffset(months=1)).strftime("%Y%m"),
        ]

        last_error = None
        for month in month_candidates:
            try:
                payload = self._fetch_month(month)
            except TreasuryClientError as exc:
                last_error = exc
                continue

            self.cache.set(cache_key, payload)
            return CacheRecord(payload=payload, fetched_at=time.time(), age_seconds=0.0, source="network")

        if stale is not None and allow_stale_on_error:
            return stale

        raise TreasuryClientError(str(last_error or "Δεν ήταν δυνατό να γίνει fetch το Treasury yield table."))

    def _fetch_month(self, month: str) -> dict[str, Any]:
        try:
            response = self.session.get(
                self.base_url,
                params={
                    "type": "daily_treasury_yield_curve",
                    "field_tdr_date_value_month": month,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise TreasuryClientError(f"Αποτυχία επικοινωνίας με το U.S. Treasury: {exc}") from exc

        if not response.ok:
            raise TreasuryClientError(f"Το U.S. Treasury επέστρεψε HTTP {response.status_code}.")

        try:
            tables = pd.read_html(StringIO(response.text))
        except ValueError as exc:
            raise TreasuryClientError("Δεν βρέθηκε αναγνώσιμο yield table στο Treasury page.") from exc

        for table in tables:
            normalized = {str(column).strip().lower(): column for column in table.columns}
            date_column = next((normalized[key] for key in normalized if "date" in key), None)
            ten_year_column = next(
                (
                    normalized[key]
                    for key in normalized
                    if "10 yr" in key or "10-year" in key or "10 year" in key
                ),
                None,
            )
            if date_column is None or ten_year_column is None:
                continue

            frame = table[[date_column, ten_year_column]].copy()
            frame.columns = ["record_date", "rate"]
            frame["record_date"] = pd.to_datetime(frame["record_date"], errors="coerce")
            frame["rate"] = pd.to_numeric(frame["rate"], errors="coerce")
            frame = frame.dropna(subset=["record_date", "rate"]).sort_values("record_date")
            if frame.empty:
                continue

            latest = frame.iloc[-1]
            return {
                "rate": float(latest["rate"]) / 100.0,
                "record_date": latest["record_date"].date().isoformat(),
                "source_url": response.url,
            }

        raise TreasuryClientError("Το Treasury page δεν περιείχε χρησιμοποιήσιμο 10-year rate table.")
