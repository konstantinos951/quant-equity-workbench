from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import requests


class FMPError(RuntimeError):
    pass


@dataclass(slots=True)
class CacheRecord:
    payload: Any
    fetched_at: float
    age_seconds: float
    source: str

    @property
    def stale(self) -> bool:
        return self.source == "stale-cache"


class DiskCacheStore:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        digest = sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str, ttl_seconds: float | None = None) -> CacheRecord | None:
        record = self.get_any_age(key)
        if record is None:
            return None
        if ttl_seconds is not None and record.age_seconds > ttl_seconds:
            return None
        record.source = "cache"
        return record

    def get_any_age(self, key: str) -> CacheRecord | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        fetched_at = float(raw.get("fetched_at", 0.0))
        if fetched_at <= 0:
            return None

        age_seconds = max(time.time() - fetched_at, 0.0)
        return CacheRecord(
            payload=raw.get("payload"),
            fetched_at=fetched_at,
            age_seconds=age_seconds,
            source="stale-cache",
        )

    def set(self, key: str, payload: Any) -> None:
        path = self._path_for_key(key)
        raw = {"fetched_at": time.time(), "payload": payload}
        path.write_text(json.dumps(raw), encoding="utf-8")


class FMPClient:
    base_url = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: str, cache_dir: Path, timeout: int = 25) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.cache = DiskCacheStore(cache_dir)

    def _cache_key(self, path: str, params: dict[str, Any]) -> str:
        canonical = {"path": path.lstrip("/"), "params": params}
        return json.dumps(canonical, sort_keys=True, default=str)

    def has_fresh_cache(self, path: str, ttl_seconds: float, **params: Any) -> bool:
        key = self._cache_key(path, self._normalized_params(params))
        return self.cache.get(key, ttl_seconds=ttl_seconds) is not None

    def peek_cache(self, path: str, ttl_seconds: float, **params: Any) -> CacheRecord | None:
        key = self._cache_key(path, self._normalized_params(params))
        return self.cache.get(key, ttl_seconds=ttl_seconds)

    def request_json(
        self,
        path: str,
        ttl_seconds: float,
        force_refresh: bool = False,
        allow_stale_on_error: bool = True,
        **params: Any,
    ) -> CacheRecord:
        if not self.api_key:
            raise FMPError("Λείπει το FMP_API_KEY. Βάλ' το στο αρχείο .env πριν τρέξεις την ανάλυση.")

        normalized_params = self._normalized_params(params)
        cache_key = self._cache_key(path, normalized_params)

        if not force_refresh:
            cached = self.cache.get(cache_key, ttl_seconds=ttl_seconds)
            if cached is not None:
                return cached

        stale = self.cache.get_any_age(cache_key)

        try:
            response = self.session.get(
                f"{self.base_url}/{path.lstrip('/')}",
                params={**normalized_params, "apikey": self.api_key},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            if stale is not None and allow_stale_on_error:
                return stale
            raise FMPError(f"Αποτυχία επικοινωνίας με το FMP: {exc}") from exc

        if response.status_code == 429:
            if stale is not None and allow_stale_on_error:
                return stale
            raise FMPError("Το FMP επέστρεψε 429 Too Many Requests. Περίμενε λίγο ή δούλεψε με cached μετοχές.")

        if response.status_code in (401, 402, 403):
            if stale is not None and allow_stale_on_error:
                return stale
            raise FMPError(
                f"Το FMP επέστρεψε HTTP {response.status_code} για το endpoint {path}. "
                "Αυτό συνήθως σημαίνει είτε plan restriction είτε symbol/provider coverage issue στο free plan."
            )

        if not response.ok:
            if stale is not None and allow_stale_on_error:
                return stale
            raise FMPError(f"Το FMP επέστρεψε HTTP {response.status_code} για το endpoint {path}.")

        try:
            payload = response.json()
        except ValueError as exc:
            if stale is not None and allow_stale_on_error:
                return stale
            raise FMPError(f"Το FMP έστειλε μη έγκυρο JSON για το endpoint {path}.") from exc

        if isinstance(payload, dict):
            message = str(
                payload.get("Error Message")
                or payload.get("error")
                or payload.get("message")
                or payload.get("Information")
                or ""
            ).strip()
            if message:
                if stale is not None and allow_stale_on_error:
                    return stale
                raise FMPError(message)

        self.cache.set(cache_key, payload)
        return CacheRecord(
            payload=payload,
            fetched_at=time.time(),
            age_seconds=0.0,
            source="network",
        )

    @staticmethod
    def _normalized_params(params: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in params.items() if value not in (None, "")}
