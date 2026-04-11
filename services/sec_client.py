from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import requests

from services.fmp_client import CacheRecord, DiskCacheStore


class SECClientError(RuntimeError):
    pass


class SECClient:
    ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
    companyfacts_url_template = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(self, user_agent: str, cache_dir: Path, timeout: int = 25) -> None:
        self.timeout = timeout
        self.user_agent = user_agent.strip() or "Quant Equity Workbench research@local.dev"
        self.cache = DiskCacheStore(cache_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json, text/plain, */*",
            }
        )

    def request_json(
        self,
        cache_key: str,
        url: str,
        ttl_seconds: float,
        force_refresh: bool = False,
        allow_stale_on_error: bool = True,
    ) -> CacheRecord:
        if not force_refresh:
            cached = self.cache.get(cache_key, ttl_seconds=ttl_seconds)
            if cached is not None:
                return cached

        stale = self.cache.get_any_age(cache_key)

        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            if stale is not None and allow_stale_on_error:
                return stale
            raise SECClientError(f"Αποτυχία επικοινωνίας με το SEC: {exc}") from exc

        if response.status_code == 429:
            if stale is not None and allow_stale_on_error:
                return stale
            raise SECClientError("Το SEC επέστρεψε 429 rate limit. Δοκίμασε λίγο αργότερα ή δούλεψε με cached data.")

        if response.status_code in (401, 403):
            if stale is not None and allow_stale_on_error:
                return stale
            raise SECClientError(
                "Το SEC απέρριψε το request. Συνήθως αυτό σημαίνει ότι χρειάζεται σωστό User-Agent με στοιχείο επικοινωνίας."
            )

        if not response.ok:
            if stale is not None and allow_stale_on_error:
                return stale
            raise SECClientError(f"Το SEC επέστρεψε HTTP {response.status_code}.")

        try:
            payload = response.json()
        except ValueError as exc:
            if stale is not None and allow_stale_on_error:
                return stale
            raise SECClientError("Το SEC επέστρεψε μη έγκυρο JSON.") from exc

        self.cache.set(cache_key, payload)
        return CacheRecord(payload=payload, fetched_at=time.time(), age_seconds=0.0, source="network")

    def get_company_tickers(self, ttl_seconds: float, force_refresh: bool = False) -> CacheRecord:
        return self.request_json(
            cache_key="company_tickers",
            url=self.ticker_map_url,
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
        )

    def get_companyfacts(self, cik: str, ttl_seconds: float, force_refresh: bool = False) -> CacheRecord:
        normalized = str(cik).zfill(10)
        return self.request_json(
            cache_key=f"companyfacts:{normalized}",
            url=self.companyfacts_url_template.format(cik=normalized),
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
        )
