from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st

from services.fmp_client import CacheRecord, FMPClient, FMPError
from services.gpr_client import GPRClient, GPRClientError
from services.models import StockDataset
from services.sec_client import SECClient, SECClientError
from services.stooq_client import StooqClient, StooqClientError
from services.yfinance_client import YFinanceClient, YFinanceClientError

load_dotenv()


class DataRetrievalError(RuntimeError):
    pass


PROFILE_TTL = 3 * 24 * 60 * 60
SEARCH_TTL = 30 * 24 * 60 * 60
QUOTE_TTL = 15 * 60
PRICE_TTL = 12 * 60 * 60
STOOQ_PRICE_TTL = 12 * 60 * 60
SEC_MAP_TTL = 30 * 24 * 60 * 60
SEC_FACTS_TTL = 3 * 24 * 60 * 60
GPR_TTL = 3 * 24 * 60 * 60
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
US_EXCHANGES = {
    "NASDAQ",
    "NYSE",
    "AMEX",
    "NYSE American",
    "NASDAQ Global Select",
    "NASDAQ Global Market",
}
BASE_CONTEXT_SYMBOLS = {
    "SPY": "US market",
    "QQQ": "growth leadership",
    "IWM": "breadth / small caps",
    "GLD": "safe haven / gold",
    "TLT": "duration / rates stress",
    "USO": "energy / geopolitical proxy",
    "UUP": "dollar pressure",
}
SECTOR_ETF_MAP = {
    "technology": "XLK",
    "financial": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "consumer discretionary": "XLY",
    "consumer staples": "XLP",
    "industrial": "XLI",
    "communication": "XLC",
    "utility": "XLU",
    "real estate": "XLRE",
    "material": "XLB",
}
SEC_INCOME_CONCEPTS = {
    "Total Revenue": [
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
    ],
    "Net Income": [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "ProfitLoss"),
    ],
    "Operating Income": [
        ("us-gaap", "OperatingIncomeLoss"),
    ],
    "EBITDA": [
        ("us-gaap", "EarningsBeforeInterestTaxesDepreciationAndAmortization"),
    ],
    "Weighted Average Shares": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("dei", "EntityCommonStockSharesOutstandingAxis"),
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ],
}
SEC_CASHFLOW_CONCEPTS = {
    "Operating Cash Flow": [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("us-gaap", "NetCashProvidedByUsedInContinuingOperations"),
    ],
    "Capital Expenditure": [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "CapitalExpendituresIncurredButNotYetPaid"),
    ],
    "Free Cash Flow": [
        ("us-gaap", "FreeCashFlow"),
    ],
}
SEC_BALANCE_CONCEPTS = {
    "Cash And Cash Equivalents": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    ],
    "Current Debt": [
        ("us-gaap", "LongTermDebtCurrent"),
        ("us-gaap", "ShortTermBorrowings"),
        ("us-gaap", "ShortTermDebt"),
    ],
    "Long Term Debt": [
        ("us-gaap", "LongTermDebtNoncurrent"),
        ("us-gaap", "LongTermDebt"),
        ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
    ],
    "Total Debt": [
        ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
        ("us-gaap", "DebtInstrumentCarryingAmount"),
    ],
    "Stockholders Equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
}


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, "", "None", "-", "null"):
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _secret_or_env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        secret_value = st.secrets.get(name, default)
    except Exception:
        secret_value = default
    return str(secret_value).strip()


@lru_cache(maxsize=1)
def _fmp_client() -> FMPClient:
    api_key = _secret_or_env("FMP_API_KEY")
    cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "fmp"
    return FMPClient(api_key=api_key, cache_dir=cache_dir)


@lru_cache(maxsize=1)
def _sec_client() -> SECClient:
    user_agent = _secret_or_env("SEC_USER_AGENT")
    cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "sec"
    return SECClient(user_agent=user_agent, cache_dir=cache_dir)


@lru_cache(maxsize=1)
def _gpr_client() -> GPRClient:
    cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "gpr"
    return GPRClient(cache_dir=cache_dir)


@lru_cache(maxsize=1)
def _stooq_client() -> StooqClient:
    cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "stooq"
    return StooqClient(cache_dir=cache_dir)


@lru_cache(maxsize=1)
def _yfinance_client() -> YFinanceClient:
    cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "yfinance"
    return YFinanceClient(cache_dir=cache_dir)


def _normalize_symbol(query: str) -> str:
    candidate = query.strip().upper()
    candidate = candidate.replace(" ", "")
    for separator in (":", "/"):
        if separator in candidate:
            left, right = candidate.rsplit(separator, 1)
            if right in {"US", "NYSE", "NASDAQ", "AMEX"} and left:
                candidate = left
    if "." in candidate:
        left, right = candidate.rsplit(".", 1)
        if right in {"US", "NYSE", "NASDAQ", "AMEX", "NYSEARCA", "NYSEAMERICAN"} and left:
            candidate = left
    return candidate


def _is_probable_ticker(query: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,11}", _normalize_symbol(query)))


def _record_payload(record: CacheRecord) -> Any:
    return record.payload


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if "historical" in payload and isinstance(payload["historical"], list):
            return [item for item in payload["historical"] if isinstance(item, dict)]
        if "facts" in payload:
            return [payload]
        return [payload]
    return []


def _first_record(payload: Any) -> dict[str, Any]:
    records = _records_from_payload(payload)
    return records[0] if records else {}


def _search_result_rank(query: str, item: dict[str, Any]) -> tuple[int, int, int, str]:
    symbol = str(item.get("symbol") or item.get("ticker") or "").upper()
    name = str(item.get("name") or item.get("companyName") or "")
    exchange = str(item.get("exchangeShortName") or item.get("exchange") or "")
    lower_query = query.strip().lower()
    exact_name = 0 if name.lower() == lower_query else 1
    exact_symbol = 0 if symbol.lower() == lower_query else 1
    exchange_penalty = 0 if exchange in US_EXCHANGES else 1
    return (exchange_penalty, exact_symbol, exact_name, symbol)


def _history_params(symbol: str, history_years: int) -> dict[str, Any]:
    end_date = pd.Timestamp.utcnow().normalize()
    start_date = end_date - pd.Timedelta(days=history_years * 370)
    return {
        "symbol": symbol,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
    }


def _fresh_fmp_cache(path: str, ttl_seconds: float, **params: Any) -> bool:
    return _fmp_client().has_fresh_cache(path, ttl_seconds=ttl_seconds, **params)


def _fresh_sec_cache(cache_key: str, ttl_seconds: float) -> bool:
    return _sec_client().cache.get(cache_key, ttl_seconds=ttl_seconds) is not None


def _fresh_gpr_cache(ttl_seconds: float) -> bool:
    return _gpr_client().cache.get("gpr_recent_daily", ttl_seconds=ttl_seconds) is not None


def _fresh_stooq_cache(symbol: str, ttl_seconds: float) -> bool:
    normalized = _stooq_client()._normalize_symbol(symbol)
    return _stooq_client().cache.get(f"stooq_daily:{normalized}", ttl_seconds=ttl_seconds) is not None


def _fresh_yfinance_cache(symbol: str, history_years: int, ttl_seconds: float) -> bool:
    cache_key = f"yfinance_daily:{symbol.upper()}:{history_years}"
    return _yfinance_client().cache.get(cache_key, ttl_seconds=ttl_seconds) is not None


def _fetch_fmp_record(path: str, ttl_seconds: float, force_refresh: bool, **params: Any) -> CacheRecord:
    try:
        return _fmp_client().request_json(
            path=path,
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
            **params,
        )
    except FMPError as exc:
        raise DataRetrievalError(str(exc)) from exc


def _fetch_sec_ticker_map(force_refresh: bool) -> CacheRecord:
    try:
        return _sec_client().get_company_tickers(ttl_seconds=SEC_MAP_TTL, force_refresh=force_refresh)
    except SECClientError as exc:
        raise DataRetrievalError(str(exc)) from exc


def _fetch_sec_companyfacts(cik: str, force_refresh: bool) -> CacheRecord:
    try:
        return _sec_client().get_companyfacts(cik=cik, ttl_seconds=SEC_FACTS_TTL, force_refresh=force_refresh)
    except SECClientError as exc:
        raise DataRetrievalError(str(exc)) from exc


def _fetch_gpr_daily(force_refresh: bool) -> CacheRecord:
    try:
        return _gpr_client().get_recent_daily_index(ttl_seconds=GPR_TTL, force_refresh=force_refresh)
    except GPRClientError as exc:
        raise DataRetrievalError(str(exc)) from exc


def _fetch_stooq_history(symbol: str, force_refresh: bool) -> CacheRecord:
    try:
        return _stooq_client().get_daily_history(
            symbol=symbol,
            ttl_seconds=STOOQ_PRICE_TTL,
            force_refresh=force_refresh,
        )
    except StooqClientError as exc:
        raise DataRetrievalError(str(exc)) from exc


def _fetch_yfinance_history(symbol: str, history_years: int, force_refresh: bool) -> CacheRecord:
    try:
        return _yfinance_client().get_daily_history(
            symbol=symbol,
            history_years=history_years,
            ttl_seconds=PRICE_TTL,
            force_refresh=force_refresh,
        )
    except YFinanceClientError as exc:
        raise DataRetrievalError(str(exc)) from exc


def _planned_requests(
    symbol: str,
    history_years: int,
    use_live_quote: bool,
    sector: str | None = None,
) -> list[dict[str, Any]]:
    requests_to_make = [
        {"provider": "FMP", "label": "Profile", "path": "profile", "ttl": PROFILE_TTL, "params": {"symbol": symbol}},
        {
            "provider": "FMP",
            "label": "Primary price history",
            "path": "historical-price-eod/light",
            "ttl": PRICE_TTL,
            "params": _history_params(symbol, history_years),
        },
        {"provider": "SEC", "label": "SEC ticker map", "cache_key": "company_tickers", "ttl": SEC_MAP_TTL},
        {"provider": "SEC", "label": "SEC companyfacts", "cache_key": f"companyfacts:{symbol}", "ttl": SEC_FACTS_TTL},
        {"provider": "GPR", "label": "Daily geopolitical risk index", "cache_key": "gpr_recent_daily", "ttl": GPR_TTL},
    ]

    for context_symbol, label in BASE_CONTEXT_SYMBOLS.items():
        requests_to_make.append(
            {
                "provider": "FMP",
                "label": f"Context: {label}",
                "path": "historical-price-eod/light",
                "ttl": PRICE_TTL,
                "params": _history_params(context_symbol, history_years),
            }
        )

    sector_etf = _sector_etf(sector or "")
    if sector_etf and sector_etf not in BASE_CONTEXT_SYMBOLS:
        requests_to_make.append(
            {
                "provider": "FMP",
                "label": f"Context: sector ETF ({sector_etf})",
                "path": "historical-price-eod/light",
                "ttl": PRICE_TTL,
                "params": _history_params(sector_etf, history_years),
            }
        )

    if use_live_quote:
        requests_to_make.append(
            {"provider": "FMP", "label": "Live quote", "path": "quote-short", "ttl": QUOTE_TTL, "params": {"symbol": symbol}}
        )
    return requests_to_make


def estimate_analysis_budget(
    query: str,
    history_years: int,
    use_live_quote: bool,
    force_refresh: bool = False,
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        return {
            "label": "0",
            "exact": True,
            "estimated_calls": 0,
            "range": (0, 0),
            "symbol_hint": None,
            "breakdown": [],
            "note": "Γράψε ticker ή όνομα εταιρείας για να υπολογιστεί το request budget.",
        }

    breakdown: list[dict[str, Any]] = []
    fixed_calls = 0
    symbol_hint = _normalize_symbol(query) if _is_probable_ticker(query) else None
    profile_record = None

    if symbol_hint is None:
        search_cached = (not force_refresh) and _fresh_fmp_cache("search-name", SEARCH_TTL, query=query)
        search_will_call = force_refresh or not search_cached
        fixed_calls += int(search_will_call)
        breakdown.append(
            {
                "label": "FMP name lookup",
                "will_call": search_will_call,
                "detail": "Αν βάλεις ticker αντί για όνομα, γλιτώνεις αυτό το request.",
            }
        )
        search_record = _fmp_client().peek_cache("search-name", SEARCH_TTL, query=query)
        if search_record is not None:
            matches = _records_from_payload(_record_payload(search_record))
            if matches:
                matches = sorted(matches, key=lambda item: _search_result_rank(query, item))
                best = matches[0]
                symbol_hint = str(best.get("symbol") or best.get("ticker") or "").upper()
                profile_record = _fmp_client().peek_cache("profile", PROFILE_TTL, symbol=symbol_hint)
    else:
        profile_record = _fmp_client().peek_cache("profile", PROFILE_TTL, symbol=symbol_hint)

    sector_hint = None
    if profile_record is not None:
        sector_hint = str(_first_record(_record_payload(profile_record)).get("sector") or "")

    if symbol_hint:
        cached_cik = _resolve_cik_from_cached_map(symbol_hint)
        estimated_calls = fixed_calls
        for item in _planned_requests(symbol_hint, history_years, use_live_quote, sector=sector_hint):
            if item["provider"] == "FMP":
                cached = (not force_refresh) and _fresh_fmp_cache(item["path"], item["ttl"], **item["params"])
            else:
                if item["provider"] == "SEC":
                    sec_cache_key = item["cache_key"]
                    if sec_cache_key.startswith("companyfacts:") and cached_cik:
                        sec_cache_key = f"companyfacts:{cached_cik}"
                    cached = (not force_refresh) and _fresh_sec_cache(sec_cache_key, item["ttl"])
                else:
                    cached = (not force_refresh) and _fresh_gpr_cache(item["ttl"])
            will_call = force_refresh or not cached
            estimated_calls += int(will_call)
            breakdown.append(
                {
                    "label": item["label"],
                    "will_call": will_call,
                    "detail": "Θα γίνει νέο request." if will_call else "Υπάρχει fresh local cache για αυτό το source.",
                }
            )
        return {
            "label": str(estimated_calls),
            "exact": True,
            "estimated_calls": estimated_calls,
            "range": (estimated_calls, estimated_calls),
            "symbol_hint": symbol_hint,
            "breakdown": breakdown,
            "note": "Ο αριθμός περιλαμβάνει FMP free requests και SEC fetches. Σε δύσκολα symbols μπορεί να γίνει και fallback history request από Yahoo ή Stooq.",
        }

    upper = fixed_calls + 11 + (1 if use_live_quote else 0)
    return {
        "label": f"{fixed_calls}-{upper}",
        "exact": False,
        "estimated_calls": None,
        "range": (fixed_calls, upper),
        "symbol_hint": None,
        "breakdown": breakdown,
        "note": "Για ακριβέστερο estimate βάλε ticker. Το free-mode χρησιμοποιεί FMP + SEC και, όταν χρειάζεται, fallback price history από Yahoo ή Stooq.",
    }


def resolve_symbol(query: str, force_refresh: bool = False) -> tuple[str, CacheRecord | None]:
    candidate = query.strip()
    if not candidate:
        raise DataRetrievalError("Βάλε ticker ή όνομα εταιρείας.")

    if _is_probable_ticker(candidate):
        return _normalize_symbol(candidate), None

    search_record = _fetch_fmp_record(
        path="search-name",
        ttl_seconds=SEARCH_TTL,
        force_refresh=force_refresh,
        query=candidate,
    )
    matches = _records_from_payload(_record_payload(search_record))
    if not matches:
        raise DataRetrievalError("Δεν βρέθηκε εταιρεία με αυτό το όνομα στο FMP.")

    matches = sorted(matches, key=lambda item: _search_result_rank(candidate, item))
    symbol = str(matches[0].get("symbol") or matches[0].get("ticker") or "").upper()
    if not symbol:
        raise DataRetrievalError("Το FMP επέστρεψε lookup χωρίς έγκυρο ticker.")
    return symbol, search_record


def _fetch_price_history(
    symbol: str,
    history_years: int,
    force_refresh: bool,
    allow_provider_fallbacks: bool = True,
) -> tuple[pd.DataFrame, CacheRecord, str]:
    history_record: CacheRecord | None = None
    provider_used = "FMP"
    provider_errors: list[str] = []
    for path in ("historical-price-eod/light", "historical-price-eod/full"):
        try:
            history_record = _fetch_fmp_record(
                path=path,
                ttl_seconds=PRICE_TTL,
                force_refresh=force_refresh,
                **_history_params(symbol, history_years),
            )
            break
        except DataRetrievalError as exc:
            provider_errors.append(f"FMP {path}: {exc}")

    if history_record is None and allow_provider_fallbacks:
        try:
            history_record = _fetch_yfinance_history(
                symbol=symbol,
                history_years=history_years,
                force_refresh=force_refresh,
            )
            provider_used = "Yahoo"
        except DataRetrievalError as exc:
            provider_errors.append(f"Yahoo: {exc}")

    if history_record is None and allow_provider_fallbacks:
        try:
            history_record = _fetch_stooq_history(symbol=symbol, force_refresh=force_refresh)
            provider_used = "Stooq"
        except DataRetrievalError as exc:
            provider_errors.append(f"Stooq: {exc}")

    if history_record is None:
        raise DataRetrievalError(
            f"Δεν βρέθηκε usable daily history για το {symbol}. "
            f"Providers tried: {' | '.join(provider_errors)}"
        )

    records = _records_from_payload(_record_payload(history_record))
    if not records:
        raise DataRetrievalError(f"Δεν βρέθηκε αρκετό ιστορικό τιμών για το {symbol}.")

    frame = pd.DataFrame(records)
    if "date" not in frame.columns:
        raise DataRetrievalError(f"Το FMP history endpoint δεν επέστρεψε date field για το {symbol}.")

    close_column = "close" if "close" in frame.columns else "adjClose" if "adjClose" in frame.columns else "price"
    frame["Date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame[close_column], errors="coerce")
    for source, target in (("open", "Open"), ("high", "High"), ("low", "Low"), ("volume", "Volume")):
        if source in frame.columns:
            frame[target] = pd.to_numeric(frame[source], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")

    if len(frame) < 120:
        raise DataRetrievalError(f"Ο provider επέστρεψε πολύ λίγο history για το {symbol}.")

    preferred_columns = [column for column in ("Open", "High", "Low", "Close", "Volume") if column in frame.columns]
    return frame[preferred_columns], history_record, provider_used


def _fetch_optional_live_quote(symbol: str, force_refresh: bool) -> tuple[float | None, CacheRecord | None, str | None]:
    try:
        record = _fetch_fmp_record(
            path="quote-short",
            ttl_seconds=QUOTE_TTL,
            force_refresh=force_refresh,
            symbol=symbol,
        )
    except DataRetrievalError as exc:
        return None, None, str(exc)

    quote = _first_record(_record_payload(record))
    return _to_float(quote.get("price")), record, None


def _gpr_frame_from_record(record: CacheRecord) -> pd.DataFrame:
    payload = _first_record(_record_payload(record))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    if "date" not in frame.columns:
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in frame.columns:
        if column == "date":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    return frame


def _normalize_ticker_map(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if all(str(key).isdigit() for key in payload):
            return [item for item in payload.values() if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _resolve_cik(symbol: str, force_refresh: bool) -> tuple[str | None, CacheRecord]:
    ticker_record = _fetch_sec_ticker_map(force_refresh=force_refresh)
    entries = _normalize_ticker_map(_record_payload(ticker_record))
    lookup = {str(item.get("ticker", "")).upper(): str(item.get("cik_str", "")).zfill(10) for item in entries}
    return lookup.get(symbol.upper()), ticker_record


def _resolve_sec_entry(symbol: str, force_refresh: bool) -> tuple[dict[str, Any] | None, CacheRecord]:
    ticker_record = _fetch_sec_ticker_map(force_refresh=force_refresh)
    entries = _normalize_ticker_map(_record_payload(ticker_record))
    for item in entries:
        if str(item.get("ticker", "")).upper() == symbol.upper():
            return item, ticker_record
    return None, ticker_record


def _resolve_cik_from_cached_map(symbol: str) -> str | None:
    cached = _sec_client().cache.get("company_tickers", ttl_seconds=SEC_MAP_TTL)
    if cached is None:
        return None
    entries = _normalize_ticker_map(_record_payload(cached))
    lookup = {str(item.get("ticker", "")).upper(): str(item.get("cik_str", "")).zfill(10) for item in entries}
    return lookup.get(symbol.upper())


def _select_annual_fact_series(
    companyfacts: dict[str, Any],
    concepts: list[tuple[str, str]],
    unit_candidates: tuple[str, ...],
    instant: bool = False,
) -> pd.Series:
    facts = companyfacts.get("facts", {})
    for taxonomy, concept in concepts:
        concept_data = facts.get(taxonomy, {}).get(concept, {})
        units = concept_data.get("units", {})
        for unit in unit_candidates:
            items = units.get(unit, [])
            if not items:
                continue

            rows = []
            for item in items:
                form = str(item.get("form") or "")
                if form not in ANNUAL_FORMS:
                    continue
                end = pd.to_datetime(item.get("end"), errors="coerce")
                if pd.isna(end):
                    continue
                value = _to_float(item.get("val"))
                if value is None:
                    continue
                filed = pd.to_datetime(item.get("filed"), errors="coerce")
                if instant:
                    rows.append((end, filed, value))
                    continue

                start = pd.to_datetime(item.get("start"), errors="coerce")
                fp = str(item.get("fp") or "")
                if pd.isna(start):
                    continue
                duration = (end - start).days
                if not (300 <= duration <= 380 or fp == "FY"):
                    continue
                rows.append((end, filed, value))

            if not rows:
                continue

            frame = pd.DataFrame(rows, columns=["end", "filed", "value"]).sort_values(["end", "filed"])
            series = frame.drop_duplicates(subset=["end"], keep="last").set_index("end")["value"].sort_index()
            if not series.empty:
                return series.astype(float)

    return pd.Series(dtype=float)


def _series_dict_to_frame(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    rows: dict[str, dict[pd.Timestamp, float]] = {}
    for label, series in series_map.items():
        if series.empty:
            continue
        for index, value in series.items():
            date = pd.to_datetime(index, errors="coerce")
            if pd.isna(date):
                continue
            rows.setdefault(label, {})[date] = float(value)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).T
    frame.columns = pd.to_datetime(frame.columns, errors="coerce")
    frame = frame.loc[:, frame.columns.notna()].sort_index(axis=1)
    return frame.astype(float)


def _build_sec_statement_frames(companyfacts: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    income_series = {
        label: _select_annual_fact_series(
            companyfacts,
            concepts=concepts,
            unit_candidates=("USD", "USDm", "USD/shares", "shares"),
            instant=(label == "Weighted Average Shares"),
        )
        for label, concepts in SEC_INCOME_CONCEPTS.items()
    }
    cashflow_series = {
        label: _select_annual_fact_series(
            companyfacts,
            concepts=concepts,
            unit_candidates=("USD", "USDm"),
            instant=False,
        )
        for label, concepts in SEC_CASHFLOW_CONCEPTS.items()
    }
    balance_series = {
        label: _select_annual_fact_series(
            companyfacts,
            concepts=concepts,
            unit_candidates=("USD", "USDm"),
            instant=True,
        )
        for label, concepts in SEC_BALANCE_CONCEPTS.items()
    }
    return (
        _series_dict_to_frame(income_series),
        _series_dict_to_frame(balance_series),
        _series_dict_to_frame(cashflow_series),
    )


def _extract_latest_value(frame: pd.DataFrame, labels: tuple[str, ...]) -> float:
    for label in labels:
        if label not in frame.index:
            continue
        series = pd.to_numeric(frame.loc[label], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[-1])
    return 0.0


def _series_growth(series: pd.Series) -> float | None:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if len(cleaned) < 2:
        return None

    cagr = None
    positive = cleaned[cleaned > 0]
    if len(positive) >= 2:
        start = float(positive.iloc[0])
        end = float(positive.iloc[-1])
        periods = len(positive) - 1
        if start > 0 and end > 0 and periods > 0:
            cagr = (end / start) ** (1 / periods) - 1

    recent_growth = cleaned.replace(0, np.nan).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if cagr is not None and not recent_growth.empty:
        return float(np.clip(0.45 * cagr + 0.55 * recent_growth.tail(3).median(), -0.35, 0.45))
    if cagr is not None:
        return float(np.clip(cagr, -0.35, 0.45))
    if not recent_growth.empty:
        return float(np.clip(recent_growth.tail(3).median(), -0.35, 0.45))
    return None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0 or not np.isfinite(denominator):
        return None
    value = numerator / denominator
    if not np.isfinite(value):
        return None
    return float(value)


def _sector_etf(sector: str) -> str | None:
    sector_lower = sector.lower()
    for key, etf in SECTOR_ETF_MAP.items():
        if key in sector_lower:
            return etf
    return None


def _fetch_context_histories(
    sector: str,
    history_years: int,
    force_refresh: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, CacheRecord], str | None]:
    context_histories: dict[str, pd.DataFrame] = {}
    context_records: dict[str, CacheRecord] = {}
    context_symbols = list(BASE_CONTEXT_SYMBOLS)
    sector_etf = _sector_etf(sector)
    if sector_etf and sector_etf not in context_symbols:
        context_symbols.append(sector_etf)

    for context_symbol in context_symbols:
        try:
            history, record, _provider = _fetch_price_history(
                context_symbol,
                history_years=history_years,
                force_refresh=force_refresh,
                allow_provider_fallbacks=False,
            )
            context_histories[context_symbol] = history
            context_records[context_symbol] = record
        except DataRetrievalError:
            continue

    return context_histories, context_records, sector_etf


def _fetch_macro_context(force_refresh: bool) -> tuple[dict[str, pd.Series], CacheRecord | None]:
    try:
        gpr_record = _fetch_gpr_daily(force_refresh=force_refresh)
    except DataRetrievalError:
        return {}, None

    frame = _gpr_frame_from_record(gpr_record)
    if frame.empty or "GPR" not in frame.columns:
        return {}, gpr_record

    macro_series = {
        "GPR": frame["GPR"].dropna(),
    }
    if "GPR_Threat" in frame.columns:
        macro_series["GPR_Threat"] = frame["GPR_Threat"].dropna()
    if "GPR_Act" in frame.columns:
        macro_series["GPR_Act"] = frame["GPR_Act"].dropna()
    return macro_series, gpr_record


def _cache_message(label: str, record: CacheRecord | None) -> str | None:
    if record is None or record.source == "network":
        return None
    if record.source == "cache":
        return f"{label}: local cache"
    if record.source == "stale-cache":
        return f"{label}: stale cache fallback"
    return f"{label}: {record.source}"


def _coverage_profile(profile: dict[str, Any], cik: str | None, annual_cashflow: pd.DataFrame, price_provider: str) -> tuple[str, str]:
    score = 0
    if price_provider in {"FMP", "Yahoo"}:
        score += 2
    elif price_provider == "Stooq":
        score += 1
    if profile:
        score += 1
    if cik:
        score += 1
    if not annual_cashflow.empty:
        score += 2

    if score >= 5:
        return "High Coverage", "Το ticker έχει αρκετά καλή κάλυψη για full valuation + forecast."
    if score >= 3:
        return "Medium Coverage", "Το ticker έχει usable κάλυψη, αλλά κάποια πεδία μπορεί να έρχονται με fallback ή λιγότερη πληρότητα."
    return "Fragile Coverage", "Το ticker στηρίζεται σε πιο αδύναμο free-data stack, άρα το αποτέλεσμα είναι περισσότερο exploratory."


def _build_proxy_price_history(
    current_price: float,
    beta: float,
    market_cap: float,
    context_histories: dict[str, pd.DataFrame],
    sector_etf: str | None,
) -> tuple[pd.DataFrame, str]:
    proxy_symbol = None
    for candidate in (sector_etf, "IWM", "QQQ", "SPY"):
        if candidate and candidate in context_histories:
            proxy_symbol = candidate
            break

    if proxy_symbol is None:
        raise DataRetrievalError("Δεν βρέθηκε ούτε direct ούτε proxy history για να τρέξει forecast mode.")

    frame = context_histories[proxy_symbol].copy()
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close) < 120:
        raise DataRetrievalError("Το διαθέσιμο proxy history ήταν πολύ μικρό για fallback forecasting.")

    proxy_returns = np.log(close).diff().dropna()
    centered = proxy_returns - float(proxy_returns.mean())

    small_cap_multiplier = 1.0
    if 0 < market_cap < 2_000_000_000:
        small_cap_multiplier = 1.35
    elif 0 < market_cap < 10_000_000_000:
        small_cap_multiplier = 1.15

    beta_scale = np.clip(beta if np.isfinite(beta) else 1.15, 0.85, 1.85)
    scaled_returns = float(proxy_returns.mean()) + centered * beta_scale * small_cap_multiplier
    scaled_returns = scaled_returns.clip(lower=-0.22, upper=0.22)

    synthetic_close = np.exp(np.cumsum(scaled_returns.to_numpy(dtype=float)))
    synthetic_close = current_price * synthetic_close / synthetic_close[-1]
    synthetic_frame = pd.DataFrame(index=scaled_returns.index)
    synthetic_frame["Close"] = synthetic_close
    synthetic_frame["Open"] = synthetic_frame["Close"].shift(1).fillna(synthetic_frame["Close"])
    synthetic_frame["High"] = synthetic_frame[["Open", "Close"]].max(axis=1) * 1.008
    synthetic_frame["Low"] = synthetic_frame[["Open", "Close"]].min(axis=1) * 0.992
    volume_source = pd.to_numeric(frame.get("Volume"), errors="coerce") if "Volume" in frame.columns else None
    if volume_source is not None and not volume_source.dropna().empty:
        synthetic_frame["Volume"] = volume_source.reindex(synthetic_frame.index).ffill().bfill()

    return synthetic_frame, proxy_symbol


def fetch_stock_dataset(
    query: str,
    history_years: int = 5,
    force_refresh: bool = False,
    use_live_quote: bool = False,
) -> StockDataset:
    symbol, search_record = resolve_symbol(query, force_refresh=force_refresh)

    profile_warning = None
    try:
        profile_record = _fetch_fmp_record(
            path="profile",
            ttl_seconds=PROFILE_TTL,
            force_refresh=force_refresh,
            symbol=symbol,
        )
    except DataRetrievalError as exc:
        profile_record = None
        profile_warning = str(exc)
    profile = _first_record(_record_payload(profile_record)) if profile_record is not None else {}

    history_warning = None
    price_history = pd.DataFrame()
    history_record = None
    price_provider = "N/A"
    try:
        price_history, history_record, price_provider = _fetch_price_history(
            symbol,
            history_years=history_years,
            force_refresh=force_refresh,
            allow_provider_fallbacks=True,
        )
    except DataRetrievalError as exc:
        history_warning = str(exc)

    context_histories, context_records, sector_etf = _fetch_context_histories(
        sector=str(profile.get("sector") or ""),
        history_years=history_years,
        force_refresh=force_refresh,
    )
    macro_context_series, gpr_record = _fetch_macro_context(force_refresh=force_refresh)

    sec_entry, sec_map_record = _resolve_sec_entry(symbol, force_refresh=force_refresh)
    cik = str(sec_entry.get("cik_str", "")).zfill(10) if sec_entry and sec_entry.get("cik_str") else None
    sec_facts_record = None
    annual_income_stmt = pd.DataFrame()
    annual_balance_sheet = pd.DataFrame()
    annual_cashflow = pd.DataFrame()

    if cik:
        sec_facts_record = _fetch_sec_companyfacts(cik=cik, force_refresh=force_refresh)
        annual_income_stmt, annual_balance_sheet, annual_cashflow = _build_sec_statement_frames(
            _first_record(_record_payload(sec_facts_record))
        )

    live_quote_record = None
    live_quote = None
    live_quote_warning = None
    if use_live_quote:
        live_quote, live_quote_record, live_quote_warning = _fetch_optional_live_quote(
            symbol,
            force_refresh=force_refresh,
        )

    profile_price = _to_float(profile.get("price")) or _to_float(profile.get("stockPrice"))
    current_price_candidate = live_quote if live_quote is not None and live_quote > 0 else profile_price
    latest_close = None
    if not price_history.empty and "Close" in price_history.columns:
        close_series = pd.to_numeric(price_history["Close"], errors="coerce").dropna()
        if not close_series.empty:
            latest_close = float(close_series.iloc[-1])
    current_price = latest_close if latest_close is not None else current_price_candidate

    revenue_series = (
        pd.to_numeric(annual_income_stmt.loc["Total Revenue"], errors="coerce").dropna()
        if "Total Revenue" in annual_income_stmt.index
        else pd.Series(dtype=float)
    )
    net_income_series = (
        pd.to_numeric(annual_income_stmt.loc["Net Income"], errors="coerce").dropna()
        if "Net Income" in annual_income_stmt.index
        else pd.Series(dtype=float)
    )
    ebitda_series = (
        pd.to_numeric(annual_income_stmt.loc["EBITDA"], errors="coerce").dropna()
        if "EBITDA" in annual_income_stmt.index
        else pd.Series(dtype=float)
    )
    equity_series = (
        pd.to_numeric(annual_balance_sheet.loc["Stockholders Equity"], errors="coerce").dropna()
        if "Stockholders Equity" in annual_balance_sheet.index
        else pd.Series(dtype=float)
    )

    market_cap = _to_float(profile.get("mktCap")) or _to_float(profile.get("marketCap")) or 0.0
    shares_outstanding = (
        _to_float(profile.get("sharesOutstanding"))
        or _extract_latest_value(annual_income_stmt, ("Weighted Average Shares",))
        or 0.0
    )
    beta = _to_float(profile.get("beta")) or 1.0
    cash_and_equivalents = _extract_latest_value(annual_balance_sheet, ("Cash And Cash Equivalents",))
    total_debt = _extract_latest_value(annual_balance_sheet, ("Total Debt",))
    if total_debt <= 0:
        total_debt = (
            _extract_latest_value(annual_balance_sheet, ("Current Debt",))
            + _extract_latest_value(annual_balance_sheet, ("Long Term Debt",))
        )

    if current_price is None and market_cap > 0 and shares_outstanding > 0:
        current_price = market_cap / max(shares_outstanding, 1e-6)

    if current_price is None or current_price <= 0:
        fallback_quote, fallback_quote_record, fallback_quote_warning = _fetch_optional_live_quote(
            symbol,
            force_refresh=force_refresh,
        )
        if live_quote_record is None:
            live_quote_record = fallback_quote_record
        if live_quote_warning is None:
            live_quote_warning = fallback_quote_warning
        if fallback_quote is not None and fallback_quote > 0:
            current_price = fallback_quote

    if current_price is None or current_price <= 0:
        raise DataRetrievalError(
            f"Δεν βρέθηκε usable current price για το {symbol}. "
            "Χωρίς price anchor δεν γίνεται ούτε valuation ούτε forecast."
        )

    if shares_outstanding <= 0:
        shares_outstanding = market_cap / max(current_price, 1e-6)
    if market_cap <= 0 and shares_outstanding > 0:
        market_cap = current_price * shares_outstanding

    history_mode = "direct"
    history_proxy_symbol = None
    if price_history.empty:
        try:
            price_history, history_proxy_symbol = _build_proxy_price_history(
                current_price=float(current_price),
                beta=float(beta),
                market_cap=float(market_cap),
                context_histories=context_histories,
                sector_etf=sector_etf,
            )
            price_provider = "Proxy"
            history_mode = "proxy"
        except DataRetrievalError:
            price_history = pd.DataFrame(
                [{"Close": float(current_price)}],
                index=[pd.Timestamp.utcnow().normalize()],
            )
            history_mode = "minimal"

    latest_revenue = float(revenue_series.iloc[-1]) if not revenue_series.empty else 0.0
    latest_net_income = float(net_income_series.iloc[-1]) if not net_income_series.empty else 0.0
    latest_ebitda = float(ebitda_series.iloc[-1]) if not ebitda_series.empty else 0.0
    latest_equity = float(equity_series.iloc[-1]) if not equity_series.empty else 0.0
    enterprise_value = market_cap + total_debt - cash_and_equivalents

    revenue_growth = _series_growth(revenue_series)
    earnings_growth = _series_growth(net_income_series)
    trailing_pe = _safe_ratio(market_cap, latest_net_income) if latest_net_income > 0 else None
    price_to_book = _safe_ratio(market_cap, latest_equity) if latest_equity > 0 else None
    enterprise_to_ebitda = _safe_ratio(enterprise_value, latest_ebitda) if latest_ebitda > 0 else None
    return_on_equity = _safe_ratio(latest_net_income, latest_equity) if latest_equity > 0 else None
    profit_margin = _safe_ratio(latest_net_income, latest_revenue) if latest_revenue > 0 else None
    peg_ratio = _safe_ratio(trailing_pe or 0.0, (earnings_growth or 0.0) * 100.0) if (trailing_pe and earnings_growth and earnings_growth > 0) else None

    cache_messages = []
    for label, record in (
        ("search", search_record),
        ("profile", profile_record),
        ("stock prices", history_record),
        ("sec ticker map", sec_map_record),
        ("sec companyfacts", sec_facts_record),
        ("daily gpr", gpr_record),
        ("live quote", live_quote_record),
    ):
        message = _cache_message(label, record)
        if message is not None:
            cache_messages.append(message)

    for context_symbol, record in context_records.items():
        message = _cache_message(f"context {context_symbol}", record)
        if message is not None:
            cache_messages.append(message)

    if live_quote_warning:
        cache_messages.append(f"live quote unavailable: {live_quote_warning}")
    if profile_warning:
        cache_messages.append(f"profile unavailable: {profile_warning}")
    if history_warning:
        cache_messages.append(f"direct history unavailable: {history_warning}")
    if history_proxy_symbol:
        cache_messages.append(f"proxy history mode: {history_proxy_symbol}")
    if history_mode == "minimal":
        cache_messages.append("minimal price mode: valuation/snapshot only, no robust forecast history")

    actual_network_calls = sum(
        1
        for record in [search_record, profile_record, history_record, sec_map_record, sec_facts_record, gpr_record, live_quote_record, *context_records.values()]
        if record is not None and record.source == "network"
    )
    coverage_label, coverage_note = _coverage_profile(
        profile=profile,
        cik=cik,
        annual_cashflow=annual_cashflow,
        price_provider=price_provider,
    )
    if history_mode == "proxy":
        coverage_label = "Fragile Coverage"
        coverage_note = (
            f"Δεν βρέθηκε direct daily history και χρησιμοποιήθηκε proxy history από {history_proxy_symbol}. "
            "Το forecast mode παραμένει usable αλλά είναι πιο exploratory."
        )
    elif history_mode == "minimal":
        coverage_label = "Fragile Coverage"
        coverage_note = (
            "Δεν βρέθηκε ούτε direct ούτε proxy history. Το app συνεχίζει με current-price anchor "
            "και fundamentals όπου υπάρχουν, αλλά το forecast mode θα είναι ουσιαστικά μη διαθέσιμο."
        )

    return StockDataset(
        symbol=symbol,
        company_name=str(profile.get("companyName") or profile.get("name") or (sec_entry or {}).get("title") or symbol),
        currency=str(profile.get("currency") or "USD"),
        sector=str(profile.get("sector") or "N/A"),
        industry=str(profile.get("industry") or "N/A"),
        description=str(profile.get("description") or ""),
        current_price=float(current_price),
        market_cap=float(market_cap),
        shares_outstanding=float(shares_outstanding),
        beta=float(beta),
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        profit_margin=profit_margin,
        trailing_pe=trailing_pe,
        peg_ratio=peg_ratio,
        price_to_book=price_to_book,
        enterprise_to_ebitda=enterprise_to_ebitda,
        return_on_equity=return_on_equity,
        cash_and_equivalents=float(cash_and_equivalents),
        total_debt=float(total_debt),
        risk_free_rate=0.042,
        price_history=price_history,
        context_price_history=context_histories,
        context_macro_series=macro_context_series,
        annual_cashflow=annual_cashflow,
        annual_balance_sheet=annual_balance_sheet,
        annual_income_stmt=annual_income_stmt,
        raw_info={
            "data_source": "FMP free + SEC companyfacts",
            "provider_code": "fmp-sec-free",
            "actual_network_calls": actual_network_calls,
            "price_basis": (
                "live quote from FMP"
                if live_quote is not None
                else f"latest close from {price_provider} history"
            ),
            "price_provider": price_provider,
            "history_mode": history_mode,
            "history_proxy_symbol": history_proxy_symbol,
            "cache_messages": cache_messages,
            "live_quote_used": bool(live_quote is not None),
            "profile_available": bool(profile),
            "sec_cik": cik,
            "sec_fundamentals_available": bool(cik and not annual_cashflow.empty),
            "context_symbols": list(context_histories),
            "macro_context_symbols": list(macro_context_series),
            "sector_etf": sector_etf,
            "coverage_label": coverage_label,
            "coverage_note": coverage_note,
        },
    )
