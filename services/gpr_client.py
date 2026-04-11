from __future__ import annotations

import io
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from services.fmp_client import CacheRecord, DiskCacheStore


class GPRClientError(RuntimeError):
    pass


class GPRClient:
    page_url = "https://www.matteoiacoviello.com/gpr_country.htm"

    def __init__(self, cache_dir: Path, timeout: int = 25) -> None:
        self.timeout = timeout
        self.cache = DiskCacheStore(cache_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Quant Equity Workbench research@local.dev",
                "Accept": "text/html,application/vnd.ms-excel,application/octet-stream,*/*",
            }
        )

    def get_recent_daily_index(self, ttl_seconds: float, force_refresh: bool = False) -> CacheRecord:
        cache_key = "gpr_recent_daily"
        if not force_refresh:
            cached = self.cache.get(cache_key, ttl_seconds=ttl_seconds)
            if cached is not None:
                return cached

        stale = self.cache.get_any_age(cache_key)
        try:
            payload = self._fetch_payload()
        except Exception as exc:
            if stale is not None:
                return stale
            raise GPRClientError(str(exc)) from exc

        self.cache.set(cache_key, payload)
        return CacheRecord(payload=payload, fetched_at=0.0, age_seconds=0.0, source="network")

    def _fetch_payload(self) -> dict[str, Any]:
        try:
            page_response = self.session.get(self.page_url, timeout=self.timeout)
            page_response.raise_for_status()
        except requests.RequestException as exc:
            raise GPRClientError(f"Αποτυχία λήψης της σελίδας GPR index: {exc}") from exc

        match = re.search(r'href="([^"]*data_gpr_daily_recent_[0-9]{8}\.xls)"', page_response.text, re.IGNORECASE)
        if not match:
            raise GPRClientError("Δεν βρέθηκε link για το daily GPR dataset.")

        data_url = urljoin(self.page_url, match.group(1))
        try:
            data_response = self.session.get(data_url, timeout=self.timeout)
            data_response.raise_for_status()
        except requests.RequestException as exc:
            raise GPRClientError(f"Αποτυχία λήψης του daily GPR dataset: {exc}") from exc

        try:
            frame = pd.read_excel(io.BytesIO(data_response.content))
        except Exception as exc:
            raise GPRClientError("Δεν μπόρεσα να διαβάσω το daily GPR Excel dataset.") from exc

        normalized = self._normalize_frame(frame)
        if normalized.empty:
            raise GPRClientError("Το daily GPR dataset δεν είχε αναγνωρίσιμα δεδομένα.")

        return {
            "source_page": self.page_url,
            "data_url": data_url,
            "rows": normalized.reset_index().rename(columns={"index": "date"}).to_dict(orient="records"),
        }

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        working.columns = [str(column).strip() for column in working.columns]
        columns_lower = {str(column).strip().lower(): column for column in working.columns}

        date_column = None
        for candidate in ("date", "day", "observation_date"):
            if candidate in columns_lower:
                date_column = columns_lower[candidate]
                break

        if date_column is None and {"year", "month", "day"}.issubset(columns_lower):
            year_col = columns_lower["year"]
            month_col = columns_lower["month"]
            day_col = columns_lower["day"]
            working["Date"] = pd.to_datetime(
                {
                    "year": pd.to_numeric(working[year_col], errors="coerce"),
                    "month": pd.to_numeric(working[month_col], errors="coerce"),
                    "day": pd.to_numeric(working[day_col], errors="coerce"),
                },
                errors="coerce",
            )
        elif date_column is not None:
            working["Date"] = pd.to_datetime(working[date_column], errors="coerce")
        else:
            return pd.DataFrame()

        candidates = {}
        for column in working.columns:
            key = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
            candidates[key] = column

        def pick_column(possible_keys: tuple[str, ...]) -> str | None:
            for key, original in candidates.items():
                if any(possible in key for possible in possible_keys):
                    return original
            return None

        gpr_col = pick_column(("gpr",))
        threat_col = pick_column(("threat", "threats"))
        act_col = pick_column(("act", "acts"))

        if gpr_col is None:
            numeric_columns = [
                column for column in working.columns
                if column != "Date" and pd.to_numeric(working[column], errors="coerce").notna().sum() > len(working) * 0.6
            ]
            if numeric_columns:
                gpr_col = numeric_columns[0]

        if gpr_col is None:
            return pd.DataFrame()

        result = pd.DataFrame(index=pd.to_datetime(working["Date"], errors="coerce"))
        result["GPR"] = pd.to_numeric(working[gpr_col], errors="coerce")
        if threat_col is not None:
            result["GPR_Threat"] = pd.to_numeric(working[threat_col], errors="coerce")
        if act_col is not None:
            result["GPR_Act"] = pd.to_numeric(working[act_col], errors="coerce")
        result = result[result.index.notna()].sort_index().dropna(how="all")
        return result
