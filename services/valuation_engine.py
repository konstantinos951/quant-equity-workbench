from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.models import StockDataset, ValuationConfig


class ValuationError(RuntimeError):
    pass


_OPERATING_CASHFLOW_FIELDS = (
    "Operating Cash Flow",
    "Cash Flow From Continuing Operating Activities",
)
_CAPEX_FIELDS = (
    "Capital Expenditure",
    "Capital Expenditures",
)
_DNA_FIELDS = ("Depreciation And Amortization",)
_REVENUE_FIELDS = (
    "Total Revenue",
    "Operating Revenue",
)
_NET_INCOME_FIELDS = ("Net Income",)
_OPERATING_INCOME_FIELDS = ("Operating Income",)
_BOOK_VALUE_FIELDS = ("Stockholders Equity",)
_CURRENT_ASSETS_FIELDS = ("Current Assets",)
_CURRENT_LIABILITIES_FIELDS = ("Current Liabilities",)
_CURRENT_DEBT_FIELDS = ("Current Debt",)
_CASH_FIELDS = ("Cash And Cash Equivalents",)


def _extract_series(frame: pd.DataFrame, labels: tuple[str, ...]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    for label in labels:
        if label not in frame.index:
            continue

        series = pd.to_numeric(frame.loc[label], errors="coerce").dropna()
        if series.empty:
            continue

        series.index = pd.to_datetime(series.index, errors="coerce")
        series = series[series.index.notna()].sort_index()
        if not series.empty:
            return series.astype(float)

    return pd.Series(dtype=float)


def _compute_fcf_series(dataset: StockDataset) -> pd.Series:
    operating = _extract_series(dataset.annual_cashflow, _OPERATING_CASHFLOW_FIELDS)
    capex = _extract_series(dataset.annual_cashflow, _CAPEX_FIELDS)
    if operating.empty or capex.empty:
        return pd.Series(dtype=float)

    aligned = pd.concat([operating.rename("operating"), capex.rename("capex")], axis=1, join="inner")
    if aligned.empty:
        return pd.Series(dtype=float)

    aligned["fcf"] = aligned["operating"] - aligned["capex"].abs()
    return aligned["fcf"].astype(float)


def _pert_sample(
    rng: np.random.Generator,
    minimum: float,
    mode: float,
    maximum: float,
    size: int,
    gamma: float = 4.0,
) -> np.ndarray:
    if maximum <= minimum:
        return np.full(size, float(mode))

    mode = float(np.clip(mode, minimum, maximum))
    alpha = 1 + gamma * (mode - minimum) / (maximum - minimum)
    beta = 1 + gamma * (maximum - mode) / (maximum - minimum)
    samples = rng.beta(alpha, beta, size=size)
    return minimum + samples * (maximum - minimum)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _series_cagr(series: pd.Series) -> float | None:
    cleaned = series[series > 0].dropna()
    if len(cleaned) < 2:
        return None

    start = float(cleaned.iloc[0])
    end = float(cleaned.iloc[-1])
    periods = len(cleaned) - 1
    if start <= 0 or end <= 0 or periods <= 0:
        return None
    return (end / start) ** (1 / periods) - 1


def _weighted_average(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.05
    weight_sum = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / weight_sum


def _estimate_growth_rate(dataset: StockDataset, fcf_series: pd.Series, revenue_series: pd.Series) -> float:
    values: list[float] = []
    weights: list[float] = []

    fcf_cagr = _series_cagr(fcf_series)
    if fcf_cagr is not None:
        values.append(fcf_cagr)
        weights.append(0.45)

    fcf_growth = (
        fcf_series.replace(0, np.nan).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    )
    if not fcf_growth.empty:
        values.append(float(fcf_growth.tail(3).median()))
        weights.append(0.20)

    revenue_cagr = _series_cagr(revenue_series)
    if revenue_cagr is not None:
        values.append(revenue_cagr)
        weights.append(0.15)

    if dataset.revenue_growth is not None:
        values.append(dataset.revenue_growth)
        weights.append(0.15)

    if dataset.earnings_growth is not None:
        values.append(dataset.earnings_growth)
        weights.append(0.05)

    base_growth = _weighted_average(values, weights)
    growth_evidence = max(
        [
            value
            for value in (
                fcf_cagr,
                revenue_cagr,
                dataset.revenue_growth,
                dataset.earnings_growth,
            )
            if value is not None
        ],
        default=base_growth,
    )
    margin = dataset.profit_margin or 0.0

    if growth_evidence >= 0.30 and margin >= 0.05:
        upper_cap = 0.30
    elif growth_evidence >= 0.20:
        upper_cap = 0.26 if margin >= 0 else 0.24
    elif growth_evidence >= 0.12:
        upper_cap = 0.22
    else:
        upper_cap = 0.18

    if margin >= 0.15:
        upper_cap += 0.02

    return _clamp(base_growth, -0.05, min(upper_cap, 0.32))


def _effective_projection_years(dataset: StockDataset, base_years: int, growth_rate: float) -> int:
    effective_years = int(base_years)
    sector = dataset.sector.lower()
    margin = dataset.profit_margin or 0.0

    if growth_rate >= 0.25 and margin >= 0.12 and dataset.market_cap >= 50_000_000_000:
        effective_years += 3
    elif growth_rate >= 0.22 and margin >= 0.04 and dataset.market_cap >= 5_000_000_000:
        effective_years += 2
    elif growth_rate >= 0.14 and sector in {"technology", "communication services"}:
        effective_years += 1

    return max(base_years, min(effective_years, 12))


def _normalize_starting_fcf(fcf_series: pd.Series) -> tuple[float, str]:
    if fcf_series.empty:
        raise ValuationError("Δεν βρέθηκαν cash flow στοιχεία αρκετά για DCF valuation.")

    latest = float(fcf_series.iloc[-1])
    if latest > 0:
        return latest, "Χρησιμοποιήθηκε το τελευταίο reported Free Cash Flow."

    positive_values = fcf_series[fcf_series > 0]
    if len(positive_values) >= 2:
        normalized = float(positive_values.tail(3).median())
        return normalized, "Το τελευταίο FCF ήταν αρνητικό, οπότε έγινε normalization από τα τελευταία θετικά έτη."

    raise ValuationError(
        "Η εταιρεία δεν έχει αρκετά σταθερά θετικό Free Cash Flow ώστε να στηριχθεί αξιόπιστο DCF."
    )


def _estimate_discount_rate(dataset: StockDataset, config: ValuationConfig) -> dict[str, float]:
    adjusted_beta = 0.65 * dataset.beta + 0.35 * 1.0
    cost_of_equity = dataset.risk_free_rate + max(adjusted_beta, 0.7) * config.equity_risk_premium
    cost_of_debt = max(dataset.risk_free_rate + config.debt_spread, 0.035)

    equity_value = max(dataset.market_cap, dataset.current_price * max(dataset.shares_outstanding, 1.0))
    debt_value = max(dataset.total_debt, 0.0)
    capital = equity_value + debt_value

    if capital <= 0:
        wacc = cost_of_equity
    else:
        wacc = (
            (equity_value / capital) * cost_of_equity
            + (debt_value / capital) * cost_of_debt * (1 - config.tax_rate)
        )

    return {
        "risk_free_rate": dataset.risk_free_rate,
        "beta_used": float(adjusted_beta),
        "cost_of_equity": _clamp(cost_of_equity, 0.06, 0.22),
        "cost_of_debt": _clamp(cost_of_debt, 0.03, 0.14),
        "wacc": _clamp(wacc, 0.06, 0.18),
    }


def _discounted_cash_flow(
    starting_fcf: float,
    growth_path: np.ndarray,
    discount_rate: float,
    terminal_growth: float,
    net_cash: float,
    shares_outstanding: float,
) -> dict[str, Any]:
    years = np.arange(1, len(growth_path) + 1)
    fcf_path = starting_fcf * np.cumprod(1 + growth_path)
    discount_factors = (1 + discount_rate) ** years
    pv_of_cashflows = float(np.sum(fcf_path / discount_factors))

    spread = max(discount_rate - terminal_growth, 0.005)
    terminal_fcf = float(fcf_path[-1] * (1 + terminal_growth))
    terminal_value = terminal_fcf / spread
    pv_terminal_value = terminal_value / ((1 + discount_rate) ** len(growth_path))

    enterprise_value = pv_of_cashflows + pv_terminal_value
    equity_value = enterprise_value + net_cash
    intrinsic_value = equity_value / max(shares_outstanding, 1.0)

    return {
        "fcf_path": fcf_path,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value": intrinsic_value,
        "terminal_value": terminal_value,
    }


def _valuation_verdict(
    percentiles: dict[str, float],
    current_price: float,
    probability_undervalued: float,
) -> tuple[str, str]:
    median_intrinsic = percentiles["p50"]
    p35 = percentiles.get("p35", percentiles.get("p25", median_intrinsic))
    p65 = percentiles.get("p65", percentiles.get("p75", median_intrinsic))
    p05 = percentiles.get("p05", p35)
    p95 = percentiles.get("p95", p65)
    spread_ratio = (p95 - p05) / max(abs(median_intrinsic), 1e-6)

    if current_price < p35 and probability_undervalued >= 0.65:
        return (
            "Undervalued",
            "Η τρέχουσα τιμή κάθεται κάτω από τη χαμηλότερη ζώνη του fair-value range και η κατανομή δείχνει ουσιαστική πιθανότητα undervaluation.",
        )
    if current_price > p65 and probability_undervalued <= 0.35:
        return (
            "Overvalued",
            "Η τρέχουσα τιμή βρίσκεται πάνω από την ανώτερη ζώνη του fair-value range και η αγορά φαίνεται να ενσωματώνει πιο αισιόδοξες παραδοχές από το base valuation.",
        )
    if spread_ratio >= 1.10:
        return (
            "Fair / Uncertain",
            "Το fair-value range είναι πολύ φαρδύ σε σχέση με τη median εκτίμηση, άρα η σωστή στάση είναι περισσότερο ταπεινότητα παρά απόλυτο label.",
        )
    return (
        "Fairly Priced",
        "Η τρέχουσα τιμή φαίνεται να κάθεται μέσα στη βασική ζώνη του estimated fair-value range.",
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0 or not np.isfinite(denominator):
        return None
    value = numerator / denominator
    if not np.isfinite(value):
        return None
    return float(value)


def _scenario_intrinsic_value(
    starting_fcf: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float,
    net_cash: float,
    shares_outstanding: float,
    projection_years: int,
) -> float:
    growth_path = np.linspace(growth_rate, terminal_growth, projection_years)
    growth_path = np.clip(growth_path, -0.25, 0.40)
    scenario = _discounted_cash_flow(
        starting_fcf=starting_fcf,
        growth_path=growth_path,
        discount_rate=discount_rate,
        terminal_growth=min(terminal_growth, discount_rate - 0.005),
        net_cash=net_cash,
        shares_outstanding=shares_outstanding,
    )
    return float(scenario["intrinsic_value"])


def _reverse_dcf_market_implied_growth(
    starting_fcf: float,
    discount_rate: float,
    terminal_growth: float,
    net_cash: float,
    shares_outstanding: float,
    current_price: float,
    projection_years: int,
) -> dict[str, Any]:
    def intrinsic_for_growth(initial_growth: float) -> float:
        return _scenario_intrinsic_value(
            starting_fcf=starting_fcf,
            growth_rate=initial_growth,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            net_cash=net_cash,
            shares_outstanding=shares_outstanding,
            projection_years=projection_years,
        )

    lower_growth = -0.20
    upper_growth = 1.50
    lower_value = intrinsic_for_growth(lower_growth)
    upper_value = intrinsic_for_growth(upper_growth)

    upper_bound_hit = False
    if current_price <= lower_value:
        required_growth = lower_growth
    elif current_price >= upper_value:
        required_growth = upper_growth
        upper_bound_hit = True
    else:
        low = lower_growth
        high = upper_growth
        for _ in range(80):
            mid = (low + high) / 2
            intrinsic = intrinsic_for_growth(mid)
            if intrinsic < current_price:
                low = mid
            else:
                high = mid
        required_growth = (low + high) / 2

    if upper_bound_hit or required_growth >= 0.45:
        label = "Very demanding"
        note = "Η αγορά τιμολογεί πολύ επιθετικό growth path σε σχέση με τα σημερινά reported fundamentals."
    elif required_growth >= 0.25:
        label = "Demanding"
        note = "Η αγορά ζητά υψηλό πολυετές growth για να στηριχθεί η τρέχουσα τιμή."
    elif required_growth >= 0.12:
        label = "Plausible"
        note = "Η τρέχουσα τιμή μπορεί να στηριχθεί, αλλά απαιτεί ακόμα ουσιαστικό growth execution."
    else:
        label = "Modest"
        note = "Η αγορά δεν απαιτεί ακραίο growth path για να στηριχθεί η τρέχουσα τιμή."

    return {
        "required_initial_growth": float(required_growth),
        "projection_years": int(projection_years),
        "upper_bound_hit": upper_bound_hit,
        "label": label,
        "note": note,
    }


def _build_dcf_scenario_table(
    starting_fcf: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float,
    net_cash: float,
    shares_outstanding: float,
    projection_years: int,
    current_price: float,
) -> list[dict[str, Any]]:
    scenario_specs = [
        ("Bear", 0.92, growth_rate - 0.04, discount_rate + 0.015, terminal_growth - 0.004),
        ("Base", 1.00, growth_rate, discount_rate, terminal_growth),
        ("Bull", 1.08, growth_rate + 0.04, discount_rate - 0.012, terminal_growth + 0.004),
    ]
    scenarios: list[dict[str, Any]] = []

    for label, fcf_multiplier, growth, discount, terminal in scenario_specs:
        intrinsic = _scenario_intrinsic_value(
            starting_fcf=starting_fcf * fcf_multiplier,
            growth_rate=_clamp(growth, -0.10, 0.36),
            discount_rate=_clamp(discount, 0.055, 0.22),
            terminal_growth=_clamp(terminal, 0.005, 0.035),
            net_cash=net_cash,
            shares_outstanding=shares_outstanding,
            projection_years=projection_years,
        )
        scenarios.append(
            {
                "scenario": label,
                "intrinsic_value": intrinsic,
                "margin_vs_market": intrinsic / current_price - 1,
            }
        )

    return scenarios


def _build_dcf_sensitivity_matrix(
    starting_fcf: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float,
    net_cash: float,
    shares_outstanding: float,
    projection_years: int,
) -> pd.DataFrame:
    growth_shifts = np.array([-0.04, -0.02, 0.00, 0.02, 0.04])
    discount_shifts = np.array([-0.02, -0.01, 0.00, 0.01, 0.02])

    matrix = pd.DataFrame(
        index=[f"WACC {discount_rate + shift:.1%}" for shift in discount_shifts],
        columns=[f"Growth {growth_rate + shift:.1%}" for shift in growth_shifts],
        dtype=float,
    )

    for discount_shift in discount_shifts:
        adjusted_discount = _clamp(discount_rate + discount_shift, 0.055, 0.22)
        for growth_shift in growth_shifts:
            adjusted_growth = _clamp(growth_rate + growth_shift, -0.10, 0.36)
            adjusted_terminal = _clamp(terminal_growth + growth_shift * 0.20, 0.005, 0.035)
            intrinsic = _scenario_intrinsic_value(
                starting_fcf=starting_fcf,
                growth_rate=adjusted_growth,
                discount_rate=adjusted_discount,
                terminal_growth=adjusted_terminal,
                net_cash=net_cash,
                shares_outstanding=shares_outstanding,
                projection_years=projection_years,
            )
            matrix.loc[f"WACC {adjusted_discount:.1%}", f"Growth {adjusted_growth:.1%}"] = intrinsic

    return matrix


def _equity_per_share_series(dataset: StockDataset) -> pd.Series:
    book_series = _extract_series(dataset.annual_balance_sheet, _BOOK_VALUE_FIELDS)
    if book_series.empty or dataset.shares_outstanding <= 0:
        return pd.Series(dtype=float)
    return (book_series / dataset.shares_outstanding).astype(float)


def _eps_series(dataset: StockDataset) -> pd.Series:
    earnings_series = _extract_series(dataset.annual_income_stmt, _NET_INCOME_FIELDS)
    if earnings_series.empty or dataset.shares_outstanding <= 0:
        return pd.Series(dtype=float)
    return (earnings_series / dataset.shares_outstanding).astype(float)


def _roe_series(dataset: StockDataset) -> pd.Series:
    earnings = _extract_series(dataset.annual_income_stmt, _NET_INCOME_FIELDS)
    equity = _extract_series(dataset.annual_balance_sheet, _BOOK_VALUE_FIELDS)
    if earnings.empty or equity.empty:
        return pd.Series(dtype=float)

    aligned = pd.concat([earnings.rename("ni"), equity.rename("equity")], axis=1, join="inner").sort_index()
    if aligned.empty:
        return pd.Series(dtype=float)

    aligned["base_equity"] = aligned["equity"].shift(1).fillna(aligned["equity"])
    roe = aligned["ni"] / aligned["base_equity"].replace(0, np.nan)
    return roe.replace([np.inf, -np.inf], np.nan).dropna().clip(-0.30, 0.45)


def _estimate_retention_rate(dataset: StockDataset) -> float:
    earnings = _extract_series(dataset.annual_income_stmt, _NET_INCOME_FIELDS)
    equity = _extract_series(dataset.annual_balance_sheet, _BOOK_VALUE_FIELDS)
    aligned = pd.concat([earnings.rename("ni"), equity.rename("equity")], axis=1, join="inner").sort_index()
    if len(aligned) < 2:
        return 0.55

    retention = (aligned["equity"].diff() / aligned["ni"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    filtered = retention[(retention > -0.10) & (retention < 1.10)]
    if filtered.empty:
        return 0.55
    return _clamp(float(filtered.tail(4).median()), 0.10, 0.90)


def _normalize_starting_roe(roe_series: pd.Series) -> tuple[float, str]:
    if roe_series.empty:
        raise ValuationError("Δεν υπάρχουν αρκετά earnings/book value στοιχεία για residual income valuation.")

    normalized = float(roe_series.tail(3).median())
    if not np.isfinite(normalized):
        raise ValuationError("Το ROE history δεν ήταν αρκετά καθαρό για residual income valuation.")

    note = "Το starting ROE προέκυψε από median των τελευταίων reported annual ROEs."
    return _clamp(normalized, -0.15, 0.30), note


def _rim_suitability(dataset: StockDataset, starting_roe: float) -> tuple[bool, str | None]:
    industry = dataset.industry.lower()
    price_to_book = dataset.price_to_book or 0.0
    asset_light_software = any(token in industry for token in ("software", "internet", "platform", "application"))

    if asset_light_software and price_to_book >= 8:
        return (
            False,
            "Το residual income model παραλείφθηκε γιατί το ticker είναι asset-light / software με πολύ υψηλό P/B, "
            "οπότε το book value δεν είναι καλό valuation anchor.",
        )

    if starting_roe <= -0.05:
        return (
            False,
            "Το residual income model παραλείφθηκε γιατί το starting ROE είναι πολύ αρνητικό και θα έβγαζε ασταθές fair value.",
        )

    return True, None


def _residual_income_value(
    starting_book_value: float,
    starting_roe: float,
    cost_of_equity: float,
    retention_rate: float,
    terminal_roe: float,
    terminal_growth: float,
    projection_years: int,
) -> dict[str, Any]:
    book_value = starting_book_value
    roe_path = np.linspace(starting_roe, terminal_roe, projection_years)
    residual_income_path = []
    earnings_path = []

    for roe in roe_path:
        earnings = book_value * roe
        residual_income = book_value * (roe - cost_of_equity)
        residual_income_path.append(residual_income)
        earnings_path.append(earnings)
        book_value = book_value + earnings * retention_rate

    residual_income_array = np.asarray(residual_income_path, dtype=float)
    years = np.arange(1, projection_years + 1)
    pv_residual_income = float(np.sum(residual_income_array / ((1 + cost_of_equity) ** years)))

    terminal_growth = min(terminal_growth, cost_of_equity - 0.005)
    terminal_residual_income = float(book_value * (terminal_roe - cost_of_equity))
    terminal_value = terminal_residual_income * (1 + terminal_growth) / max(cost_of_equity - terminal_growth, 0.005)
    pv_terminal_value = terminal_value / ((1 + cost_of_equity) ** projection_years)
    intrinsic_value = starting_book_value + pv_residual_income + pv_terminal_value

    return {
        "intrinsic_value": intrinsic_value,
        "roe_path": roe_path,
        "earnings_path": np.asarray(earnings_path, dtype=float),
        "terminal_value": terminal_value,
        "ending_book_value": book_value,
    }


def _rim_intrinsic_value(
    starting_book_value: float,
    starting_roe: float,
    cost_of_equity: float,
    retention_rate: float,
    terminal_roe: float,
    terminal_growth: float,
    projection_years: int,
) -> float:
    return float(
        _residual_income_value(
            starting_book_value=starting_book_value,
            starting_roe=starting_roe,
            cost_of_equity=cost_of_equity,
            retention_rate=retention_rate,
            terminal_roe=terminal_roe,
            terminal_growth=terminal_growth,
            projection_years=projection_years,
        )["intrinsic_value"]
    )


def _build_rim_scenario_table(
    starting_book_value: float,
    starting_roe: float,
    cost_of_equity: float,
    retention_rate: float,
    terminal_roe: float,
    terminal_growth: float,
    projection_years: int,
    current_price: float,
) -> list[dict[str, Any]]:
    scenario_specs = [
        ("Bear", 0.97, starting_roe - 0.03, cost_of_equity + 0.015, retention_rate - 0.08, terminal_roe - 0.02),
        ("Base", 1.00, starting_roe, cost_of_equity, retention_rate, terminal_roe),
        ("Bull", 1.03, starting_roe + 0.03, cost_of_equity - 0.010, retention_rate + 0.05, terminal_roe + 0.02),
    ]
    scenarios: list[dict[str, Any]] = []

    for label, book_multiplier, roe, cost_equity, retention, terminal in scenario_specs:
        intrinsic = _rim_intrinsic_value(
            starting_book_value=starting_book_value * book_multiplier,
            starting_roe=_clamp(roe, -0.12, 0.35),
            cost_of_equity=_clamp(cost_equity, 0.06, 0.22),
            retention_rate=_clamp(retention, 0.05, 0.95),
            terminal_roe=_clamp(terminal, cost_equity - 0.02, cost_equity + 0.04),
            terminal_growth=_clamp(terminal_growth, 0.0, 0.025),
            projection_years=projection_years,
        )
        scenarios.append(
            {
                "scenario": label,
                "intrinsic_value": intrinsic,
                "margin_vs_market": intrinsic / current_price - 1,
            }
        )

    return scenarios


def _build_rim_sensitivity_matrix(
    starting_book_value: float,
    starting_roe: float,
    cost_of_equity: float,
    retention_rate: float,
    terminal_roe: float,
    terminal_growth: float,
    projection_years: int,
) -> pd.DataFrame:
    roe_shifts = np.array([-0.04, -0.02, 0.00, 0.02, 0.04])
    cost_shifts = np.array([-0.02, -0.01, 0.00, 0.01, 0.02])
    matrix = pd.DataFrame(
        index=[f"Cost of equity {cost_of_equity + shift:.1%}" for shift in cost_shifts],
        columns=[f"Starting ROE {starting_roe + shift:.1%}" for shift in roe_shifts],
        dtype=float,
    )

    for cost_shift in cost_shifts:
        adjusted_cost = _clamp(cost_of_equity + cost_shift, 0.06, 0.22)
        for roe_shift in roe_shifts:
            adjusted_roe = _clamp(starting_roe + roe_shift, -0.12, 0.35)
            adjusted_terminal_roe = _clamp(terminal_roe + roe_shift * 0.35, adjusted_cost - 0.02, adjusted_cost + 0.04)
            intrinsic = _rim_intrinsic_value(
                starting_book_value=starting_book_value,
                starting_roe=adjusted_roe,
                cost_of_equity=adjusted_cost,
                retention_rate=retention_rate,
                terminal_roe=adjusted_terminal_roe,
                terminal_growth=terminal_growth,
                projection_years=projection_years,
            )
            matrix.loc[
                f"Cost of equity {adjusted_cost:.1%}",
                f"Starting ROE {adjusted_roe:.1%}",
            ] = intrinsic

    return matrix


def _base_ratios(dataset: StockDataset) -> dict[str, float | None]:
    return {
        "trailing_pe": dataset.trailing_pe,
        "peg_ratio": dataset.peg_ratio,
        "price_to_book": dataset.price_to_book,
        "enterprise_to_ebitda": dataset.enterprise_to_ebitda,
        "return_on_equity": dataset.return_on_equity,
        "profit_margin": dataset.profit_margin,
    }


def _operating_margin_series(dataset: StockDataset) -> pd.Series:
    revenue_series = _extract_series(dataset.annual_income_stmt, _REVENUE_FIELDS)
    operating_income_series = _extract_series(dataset.annual_income_stmt, _OPERATING_INCOME_FIELDS)
    if revenue_series.empty or operating_income_series.empty:
        return pd.Series(dtype=float)

    aligned = pd.concat(
        [revenue_series.rename("revenue"), operating_income_series.rename("operating_income")],
        axis=1,
        join="inner",
    ).sort_index()
    if aligned.empty:
        return pd.Series(dtype=float)

    margin = aligned["operating_income"] / aligned["revenue"].replace(0, np.nan)
    return margin.replace([np.inf, -np.inf], np.nan).dropna().clip(-0.25, 0.45)


def _dna_series(dataset: StockDataset) -> pd.Series:
    dna = _extract_series(dataset.annual_cashflow, _DNA_FIELDS)
    if not dna.empty:
        return dna.abs().astype(float)

    ebitda_series = _extract_series(dataset.annual_income_stmt, ("EBITDA",))
    operating_income_series = _extract_series(dataset.annual_income_stmt, _OPERATING_INCOME_FIELDS)
    if ebitda_series.empty or operating_income_series.empty:
        return pd.Series(dtype=float)

    aligned = pd.concat(
        [ebitda_series.rename("ebitda"), operating_income_series.rename("operating_income")],
        axis=1,
        join="inner",
    ).sort_index()
    if aligned.empty:
        return pd.Series(dtype=float)

    derived = (aligned["ebitda"] - aligned["operating_income"]).clip(lower=0.0)
    return derived.astype(float)


def _non_cash_working_capital_series(dataset: StockDataset) -> pd.Series:
    current_assets = _extract_series(dataset.annual_balance_sheet, _CURRENT_ASSETS_FIELDS)
    current_liabilities = _extract_series(dataset.annual_balance_sheet, _CURRENT_LIABILITIES_FIELDS)
    if current_assets.empty or current_liabilities.empty:
        return pd.Series(dtype=float)

    cash_series = _extract_series(dataset.annual_balance_sheet, _CASH_FIELDS)
    current_debt_series = _extract_series(dataset.annual_balance_sheet, _CURRENT_DEBT_FIELDS)

    aligned = pd.concat(
        [
            current_assets.rename("current_assets"),
            current_liabilities.rename("current_liabilities"),
            cash_series.rename("cash"),
            current_debt_series.rename("current_debt"),
        ],
        axis=1,
        join="inner",
    ).sort_index()
    if aligned.empty:
        aligned = pd.concat(
            [
                current_assets.rename("current_assets"),
                current_liabilities.rename("current_liabilities"),
            ],
            axis=1,
            join="inner",
        ).sort_index()
        if aligned.empty:
            return pd.Series(dtype=float)
        aligned["cash"] = 0.0
        aligned["current_debt"] = 0.0
    else:
        aligned["cash"] = aligned["cash"].fillna(0.0)
        aligned["current_debt"] = aligned["current_debt"].fillna(0.0)

    non_cash_wc = (aligned["current_assets"] - aligned["cash"]) - (
        aligned["current_liabilities"] - aligned["current_debt"]
    )
    return non_cash_wc.astype(float)


def _historical_fcff_frame(dataset: StockDataset, tax_rate: float) -> pd.DataFrame:
    revenue_series = _extract_series(dataset.annual_income_stmt, _REVENUE_FIELDS)
    operating_income_series = _extract_series(dataset.annual_income_stmt, _OPERATING_INCOME_FIELDS)
    if revenue_series.empty or operating_income_series.empty:
        return pd.DataFrame()

    aligned = pd.concat(
        [
            revenue_series.rename("revenue"),
            operating_income_series.rename("operating_income"),
            _dna_series(dataset).rename("dna"),
            _extract_series(dataset.annual_cashflow, _CAPEX_FIELDS).abs().rename("capex"),
            _non_cash_working_capital_series(dataset).rename("nwc"),
        ],
        axis=1,
        join="outer",
    ).sort_index()

    aligned = aligned[aligned["revenue"].notna() & aligned["operating_income"].notna()].copy()
    if aligned.empty:
        return pd.DataFrame()

    aligned["dna"] = aligned["dna"].fillna(0.0)
    aligned["capex"] = aligned["capex"].fillna(0.0)
    aligned["nwc"] = aligned["nwc"].ffill().bfill()
    aligned["delta_nwc"] = aligned["nwc"].diff().fillna(0.0)
    aligned["nopat"] = aligned["operating_income"] * (1 - tax_rate)
    aligned["reinvestment"] = aligned["capex"] - aligned["dna"] + aligned["delta_nwc"]
    aligned["fcff"] = aligned["nopat"] - aligned["reinvestment"]
    aligned["operating_margin"] = (
        aligned["operating_income"] / aligned["revenue"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    aligned["sales_delta"] = aligned["revenue"].diff()
    return aligned.dropna(subset=["revenue", "operating_income"]).copy()


def _capex_to_sales(dataset: StockDataset) -> float | None:
    revenue_series = _extract_series(dataset.annual_income_stmt, _REVENUE_FIELDS)
    capex_series = _extract_series(dataset.annual_cashflow, _CAPEX_FIELDS).abs()
    aligned = pd.concat([revenue_series.rename("revenue"), capex_series.rename("capex")], axis=1, join="inner")
    if aligned.empty:
        return None
    ratio = aligned["capex"] / aligned["revenue"].replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return None
    return float(ratio.tail(3).median())


def _classify_company(dataset: StockDataset, tax_rate: float) -> dict[str, Any]:
    revenue_series = _extract_series(dataset.annual_income_stmt, _REVENUE_FIELDS)
    operating_margin_series = _operating_margin_series(dataset)
    fcff_frame = _historical_fcff_frame(dataset, tax_rate)
    latest_fcff = float(fcff_frame["fcff"].iloc[-1]) if not fcff_frame.empty else None
    latest_net_income = float(
        _extract_series(dataset.annual_income_stmt, _NET_INCOME_FIELDS).iloc[-1]
    ) if not _extract_series(dataset.annual_income_stmt, _NET_INCOME_FIELDS).empty else None

    sector_lower = dataset.sector.lower()
    industry_lower = dataset.industry.lower()
    cyclical_tokens = ("semiconductor", "steel", "shipping", "oil", "gas", "chemical", "mining", "airline", "auto")
    financial_tokens = ("financial", "bank", "insurance", "capital markets", "asset management")
    asset_light_tokens = ("software", "internet", "platform", "application", "infrastructure")

    revenue_cagr = _series_cagr(revenue_series)
    revenue_growth = dataset.revenue_growth if dataset.revenue_growth is not None else revenue_cagr
    capex_to_sales = _capex_to_sales(dataset)

    is_financial = any(token in sector_lower or token in industry_lower for token in financial_tokens)
    is_reit = "reit" in industry_lower
    cyclical = any(token in industry_lower for token in cyclical_tokens)
    asset_light = bool(
        any(token in industry_lower for token in asset_light_tokens)
        and (capex_to_sales is None or capex_to_sales < 0.05)
    )
    high_growth = bool(revenue_growth is not None and revenue_growth > 0.18)
    has_positive_fcf = latest_fcff is not None and latest_fcff > 0
    has_positive_earnings = latest_net_income is not None and latest_net_income > 0
    data_poor = len(revenue_series.dropna()) < 3 or len(operating_margin_series.dropna()) < 3
    book_relevant = bool(is_financial or (dataset.price_to_book is not None and dataset.price_to_book < 4 and not asset_light))

    reasons: list[str] = []
    if is_financial:
        reasons.append("Financial sector detected, άρα το equity/book-based valuation είναι πιο κατάλληλο.")
    if is_reit:
        reasons.append("REIT-like company detected, άρα το γενικό FCFF engine δεν είναι το ιδανικό πρώτο εργαλείο.")
    if cyclical:
        reasons.append("Cyclical industry detected, άρα χρειάζεται πιο normalized margin handling.")
    if asset_light:
        reasons.append("Asset-light profile detected, άρα το book value είναι πιο αδύναμο anchor.")
    if high_growth:
        reasons.append("High-growth regime detected.")
    if data_poor:
        reasons.append("Limited fundamental history detected.")

    return {
        "is_financial": is_financial,
        "is_reit": is_reit,
        "cyclical": cyclical,
        "asset_light": asset_light,
        "high_growth": high_growth,
        "has_positive_fcf": has_positive_fcf,
        "has_positive_earnings": has_positive_earnings,
        "book_relevant": book_relevant,
        "data_poor": data_poor,
        "revenue_cagr": revenue_cagr,
        "fcff_frame": fcff_frame,
        "reasons": reasons,
    }


def _choose_valuation_framework(classification: dict[str, Any]) -> tuple[str, str]:
    if classification["is_reit"]:
        return "hard_to_value", "Το app δεν έχει ακόμα ειδικό REIT/AFFO engine, άρα το intrinsic valuation θα ήταν παραπλανητικό."
    if classification["is_financial"]:
        return "residual_income", "Η εταιρεία φαίνεται financial, άρα το residual-income rail είναι πιο κατάλληλο."
    if classification["data_poor"] and not classification["high_growth"]:
        return "hard_to_value", "Τα διαθέσιμα annual fundamentals είναι λίγα για robust intrinsic valuation."
    if classification["cyclical"]:
        return "normalized_fcff", "Η εταιρεία φαίνεται cyclical, άρα χρησιμοποιούμε normalized FCFF framework."
    if classification["high_growth"] and (not classification["has_positive_fcf"] or classification["asset_light"]):
        return "growth_fcff", "Η εταιρεία φαίνεται high-growth / asset-light, άρα χρησιμοποιούμε growth-scenario FCFF."
    return "fcff", "Η εταιρεία φαίνεται non-financial με αρκετά usable fundamentals, άρα default rail είναι FCFF + WACC."


def _sector_margin_anchor(dataset: StockDataset) -> float:
    sector_lower = dataset.sector.lower()
    industry_lower = dataset.industry.lower()
    if "software" in industry_lower or "internet" in industry_lower or "platform" in industry_lower:
        return 0.28
    if "semiconductor" in industry_lower:
        return 0.22
    if "industrial" in sector_lower:
        return 0.14
    if "consumer staples" in sector_lower:
        return 0.16
    if "consumer discretionary" in sector_lower:
        return 0.14
    if "health" in sector_lower:
        return 0.18
    if "energy" in sector_lower or "material" in sector_lower:
        return 0.16
    if "technology" in sector_lower or "communication" in sector_lower:
        return 0.22
    return 0.15


def _estimate_sales_to_capital(dataset: StockDataset, classification: dict[str, Any], fcff_frame: pd.DataFrame) -> float:
    if not fcff_frame.empty:
        valid = fcff_frame[(fcff_frame["sales_delta"] > 0) & (fcff_frame["reinvestment"] > 0)].copy()
        if not valid.empty:
            series = valid["sales_delta"] / valid["reinvestment"]
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            series = series[(series > 0.2) & (series < 8.0)]
            if not series.empty:
                historical_estimate = float(_clamp(float(series.tail(4).median()), 0.6, 5.0))
                if classification["asset_light"] and classification["high_growth"]:
                    return max(historical_estimate, 2.2)
                if classification["asset_light"]:
                    return max(historical_estimate, 1.8)
                if classification["cyclical"]:
                    return float(_clamp(historical_estimate, 0.8, 2.4))
                return historical_estimate

    if classification["asset_light"] and classification["high_growth"]:
        return 2.6
    if classification["asset_light"]:
        return 2.0
    if classification["cyclical"]:
        return 1.2
    if dataset.sector.lower().startswith("technology"):
        return 1.8
    return 1.5


def _margin_profile(
    dataset: StockDataset,
    classification: dict[str, Any],
    mode: str,
) -> tuple[float, float, pd.Series]:
    margin_series = _operating_margin_series(dataset)
    latest_margin = float(margin_series.iloc[-1]) if not margin_series.empty else float(dataset.profit_margin or 0.08)
    normalized_margin = float(margin_series.tail(5).median()) if not margin_series.empty else latest_margin
    sector_anchor = _sector_margin_anchor(dataset)

    if mode == "growth_fcff":
        starting_margin = _clamp(latest_margin, -0.10, 0.32)
        target_margin = _clamp(max(normalized_margin, sector_anchor, starting_margin + 0.03), 0.08, 0.36)
    elif mode == "normalized_fcff":
        starting_margin = _clamp(normalized_margin, -0.05, 0.28)
        target_margin = _clamp(0.7 * normalized_margin + 0.3 * sector_anchor, 0.05, 0.26)
    else:
        starting_margin = _clamp(latest_margin, -0.08, 0.28)
        target_margin = _clamp(max(starting_margin, 0.7 * normalized_margin + 0.3 * sector_anchor), 0.04, 0.28)

    if classification["asset_light"] and classification["high_growth"]:
        target_margin = _clamp(max(target_margin, 0.18), 0.06, 0.34)

    return starting_margin, target_margin, margin_series


def _base_revenue_growth(dataset: StockDataset, classification: dict[str, Any]) -> float:
    revenue_series = _extract_series(dataset.annual_income_stmt, _REVENUE_FIELDS)
    cagr = _series_cagr(revenue_series)
    recent = (
        revenue_series.replace(0, np.nan).pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(3).median()
        if not revenue_series.empty
        else np.nan
    )
    candidates = [value for value in (cagr, dataset.revenue_growth, None if pd.isna(recent) else float(recent)) if value is not None]
    base = float(np.median(candidates)) if candidates else 0.05

    if classification["high_growth"]:
        return _clamp(max(base, 0.12), 0.04, 0.35)
    if classification["cyclical"]:
        return _clamp(min(base, 0.10), -0.03, 0.12)
    return _clamp(base, -0.03, 0.18)


def _fcff_intrinsic_value_from_drivers(
    starting_revenue: float,
    starting_margin: float,
    target_margin: float,
    revenue_growth: float,
    discount_rate: float,
    terminal_growth: float,
    sales_to_capital: float,
    net_cash: float,
    shares_outstanding: float,
    projection_years: int,
    tax_rate: float,
) -> dict[str, Any]:
    fade = np.linspace(0.0, 1.0, projection_years)
    growth_path = revenue_growth + (terminal_growth - revenue_growth) * fade
    growth_path = np.clip(growth_path, -0.10, 0.40)
    margin_path = starting_margin + (target_margin - starting_margin) * np.power(fade, 0.85)
    margin_path = np.clip(margin_path, -0.15, 0.40)

    revenues = []
    fcff_values = []
    previous_revenue = starting_revenue
    for growth, margin in zip(growth_path, margin_path):
        revenue = previous_revenue * (1 + growth)
        delta_revenue = revenue - previous_revenue
        reinvestment = delta_revenue / max(sales_to_capital, 0.35)
        ebit = revenue * margin
        fcff = ebit * (1 - tax_rate) - reinvestment
        revenues.append(revenue)
        fcff_values.append(fcff)
        previous_revenue = revenue

    revenues_array = np.asarray(revenues, dtype=float)
    fcff_array = np.asarray(fcff_values, dtype=float)
    years = np.arange(1, projection_years + 1)
    discount_factors = (1 + discount_rate) ** years
    pv_fcf = float(np.sum(fcff_array / discount_factors))

    terminal_revenue = revenues_array[-1] * (1 + terminal_growth)
    terminal_reinvestment = (terminal_revenue - revenues_array[-1]) / max(sales_to_capital, 0.35)
    terminal_fcff = terminal_revenue * target_margin * (1 - tax_rate) - terminal_reinvestment
    spread = max(discount_rate - terminal_growth, 0.005)
    terminal_value = terminal_fcff / spread
    pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)
    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value + net_cash
    intrinsic_value = equity_value / max(shares_outstanding, 1.0)

    return {
        "revenues": revenues_array,
        "fcff_path": fcff_array,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value": intrinsic_value,
        "terminal_value": terminal_value,
    }


def _compute_fcff_report(
    dataset: StockDataset,
    config: ValuationConfig,
    discount_rates: dict[str, float],
    classification: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    revenue_series = _extract_series(dataset.annual_income_stmt, _REVENUE_FIELDS)
    if revenue_series.empty:
        raise ValuationError("Δεν υπάρχουν αρκετά revenue στοιχεία για FCFF valuation.")

    starting_revenue = float(revenue_series.iloc[-1])
    if starting_revenue <= 0:
        raise ValuationError("Το latest reported revenue δεν είναι usable για FCFF valuation.")

    tax_rate = config.tax_rate
    fcff_frame = classification["fcff_frame"]
    starting_margin, target_margin, margin_series = _margin_profile(dataset, classification, mode)
    revenue_growth = _base_revenue_growth(dataset, classification)
    if mode == "growth_fcff":
        revenue_growth = _clamp(max(revenue_growth, 0.10), 0.06, 0.35)
    elif mode == "normalized_fcff":
        revenue_growth = _clamp(min(revenue_growth, 0.10), -0.03, 0.12)

    effective_projection_years = _effective_projection_years(dataset, config.projection_years, revenue_growth)
    if mode == "growth_fcff":
        effective_projection_years = min(max(effective_projection_years, config.projection_years + 1), 12)

    terminal_growth = _clamp(min(0.0325, max(0.0125, dataset.risk_free_rate * 0.68)), 0.0125, 0.0325)
    sales_to_capital = _estimate_sales_to_capital(dataset, classification, fcff_frame)
    net_cash = dataset.cash_and_equivalents - dataset.total_debt

    deterministic = _fcff_intrinsic_value_from_drivers(
        starting_revenue=starting_revenue,
        starting_margin=starting_margin,
        target_margin=target_margin,
        revenue_growth=revenue_growth,
        discount_rate=discount_rates["wacc"],
        terminal_growth=terminal_growth,
        sales_to_capital=sales_to_capital,
        net_cash=net_cash,
        shares_outstanding=dataset.shares_outstanding,
        projection_years=effective_projection_years,
        tax_rate=tax_rate,
    )

    rng = np.random.default_rng(21 if mode == "fcff" else 29 if mode == "growth_fcff" else 33)
    growth_low = _clamp(revenue_growth - (0.05 if mode != "growth_fcff" else 0.08), -0.05, 0.24)
    growth_high = _clamp(revenue_growth + (0.05 if mode != "growth_fcff" else 0.10), 0.04, 0.42)
    target_margin_low = _clamp(target_margin - 0.04, -0.05, 0.28)
    target_margin_high = _clamp(target_margin + 0.05, 0.02, 0.38)
    starting_margin_low = _clamp(starting_margin - 0.03, -0.15, 0.25)
    starting_margin_high = _clamp(starting_margin + 0.03, -0.05, 0.30)
    sales_to_capital_low = _clamp(sales_to_capital * 0.75, 0.5, 4.5)
    sales_to_capital_high = _clamp(sales_to_capital * 1.25, 0.8, 6.0)
    discount_low = _clamp(discount_rates["wacc"] - 0.018, 0.055, 0.18)
    discount_high = _clamp(discount_rates["wacc"] + 0.022, 0.07, 0.23)
    terminal_low = _clamp(terminal_growth - 0.007, 0.0075, 0.03)
    terminal_high = _clamp(terminal_growth + 0.007, 0.012, 0.035)

    simulated_growth = _pert_sample(rng, growth_low, revenue_growth, growth_high, config.simulations)
    simulated_target_margin = _pert_sample(rng, target_margin_low, target_margin, target_margin_high, config.simulations)
    simulated_starting_margin = _pert_sample(rng, starting_margin_low, starting_margin, starting_margin_high, config.simulations)
    simulated_sales_to_capital = _pert_sample(rng, sales_to_capital_low, sales_to_capital, sales_to_capital_high, config.simulations)
    simulated_discount = _pert_sample(rng, discount_low, discount_rates["wacc"], discount_high, config.simulations)
    simulated_terminal = _pert_sample(rng, terminal_low, terminal_growth, terminal_high, config.simulations)
    simulated_terminal = np.minimum(simulated_terminal, simulated_discount - 0.005)

    growth_excess = simulated_growth - revenue_growth
    simulated_sales_to_capital = np.clip(
        simulated_sales_to_capital / (1.0 + np.maximum(growth_excess, 0.0) * 1.25),
        0.45,
        6.0,
    )
    simulated_target_margin = np.clip(
        simulated_target_margin - np.maximum(growth_excess, 0.0) * 0.04,
        -0.08,
        0.40,
    )

    intrinsic_distribution = np.empty(config.simulations, dtype=float)
    for index in range(config.simulations):
        intrinsic_distribution[index] = _fcff_intrinsic_value_from_drivers(
            starting_revenue=starting_revenue,
            starting_margin=float(simulated_starting_margin[index]),
            target_margin=float(simulated_target_margin[index]),
            revenue_growth=float(simulated_growth[index]),
            discount_rate=float(simulated_discount[index]),
            terminal_growth=float(simulated_terminal[index]),
            sales_to_capital=float(simulated_sales_to_capital[index]),
            net_cash=net_cash,
            shares_outstanding=dataset.shares_outstanding,
            projection_years=effective_projection_years,
            tax_rate=tax_rate,
        )["intrinsic_value"]

    percentiles = {
        "p05": float(np.percentile(intrinsic_distribution, 5)),
        "p25": float(np.percentile(intrinsic_distribution, 25)),
        "p35": float(np.percentile(intrinsic_distribution, 35)),
        "p50": float(np.percentile(intrinsic_distribution, 50)),
        "p65": float(np.percentile(intrinsic_distribution, 65)),
        "p75": float(np.percentile(intrinsic_distribution, 75)),
        "p95": float(np.percentile(intrinsic_distribution, 95)),
    }
    probability_undervalued = float(np.mean(intrinsic_distribution > dataset.current_price))
    verdict, verdict_reason = _valuation_verdict(percentiles, dataset.current_price, probability_undervalued)
    market_implied = _reverse_dcf_market_implied_growth(
        starting_fcf=max(float(deterministic["fcff_path"][0]), 1e-6),
        discount_rate=discount_rates["wacc"],
        terminal_growth=terminal_growth,
        net_cash=net_cash,
        shares_outstanding=dataset.shares_outstanding,
        current_price=dataset.current_price,
        projection_years=max(effective_projection_years, 7),
    )

    mode_name = {
        "fcff": "Monte Carlo FCFF DCF",
        "growth_fcff": "Monte Carlo Growth Scenario FCFF",
        "normalized_fcff": "Monte Carlo Normalized FCFF",
    }[mode]

    return {
        "method_family": mode,
        "method_used": mode_name,
        "current_price": dataset.current_price,
        "deterministic_intrinsic_value": float(deterministic["intrinsic_value"]),
        "intrinsic_value_distribution": intrinsic_distribution,
        "percentiles": percentiles,
        "margin_of_safety": percentiles["p50"] / dataset.current_price - 1,
        "probability_undervalued": probability_undervalued,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "assumptions": {
            "starting_revenue": starting_revenue,
            "starting_margin": starting_margin,
            "target_margin": target_margin,
            "revenue_growth": revenue_growth,
            "sales_to_capital": sales_to_capital,
            "terminal_growth": terminal_growth,
            "risk_free_rate": discount_rates["risk_free_rate"],
            "beta_used": discount_rates["beta_used"],
            "cost_of_equity": discount_rates["cost_of_equity"],
            "cost_of_debt": discount_rates["cost_of_debt"],
            "wacc": discount_rates["wacc"],
            "net_cash": net_cash,
            "projection_years": config.projection_years,
            "effective_projection_years": effective_projection_years,
        },
        "ratios": _base_ratios(dataset),
        "market_implied": market_implied,
        "scenario_table": _build_dcf_scenario_table(
            starting_fcf=max(float(deterministic["fcff_path"][0]), 1e-6),
            growth_rate=revenue_growth,
            discount_rate=discount_rates["wacc"],
            terminal_growth=terminal_growth,
            net_cash=net_cash,
            shares_outstanding=dataset.shares_outstanding,
            projection_years=effective_projection_years,
            current_price=dataset.current_price,
        ),
        "scenario_title": "Bear / Base / Bull valuation scenarios",
        "sensitivity_matrix": _build_dcf_sensitivity_matrix(
            starting_fcf=max(float(deterministic["fcff_path"][0]), 1e-6),
            growth_rate=revenue_growth,
            discount_rate=discount_rates["wacc"],
            terminal_growth=terminal_growth,
            net_cash=net_cash,
            shares_outstanding=dataset.shares_outstanding,
            projection_years=effective_projection_years,
        ),
        "sensitivity_title": "Valuation sensitivity to growth and WACC",
        "uncertainty": {
            "intrinsic_range_90": percentiles["p95"] - percentiles["p05"],
            "range_vs_median": (percentiles["p95"] - percentiles["p05"]) / max(percentiles["p50"], 1e-6),
        },
        "normalization_note": (
            "Το FCFF valuation βασίστηκε σε revenue -> margin -> reinvestment forecasting με sales-to-capital discipline."
            if mode != "normalized_fcff"
            else "Το FCFF valuation χρησιμοποιεί πιο normalized margins λόγω cyclical profile."
        ),
        "fundamental_series": {
            "Revenue": revenue_series,
            "Observed FCFF": fcff_frame["fcff"] if not fcff_frame.empty else pd.Series(dtype=float),
        },
        "input_rows": [
            ("Method", mode_name),
            ("Starting revenue", starting_revenue),
            ("Starting EBIT margin", starting_margin),
            ("Target EBIT margin", target_margin),
            ("Revenue growth", revenue_growth),
            ("Sales to capital", sales_to_capital),
            ("Terminal growth", terminal_growth),
            ("Risk-free rate", discount_rates["risk_free_rate"]),
            ("Beta used", discount_rates["beta_used"]),
            ("Cost of equity", discount_rates["cost_of_equity"]),
            ("Cost of debt", discount_rates["cost_of_debt"]),
            ("WACC", discount_rates["wacc"]),
            ("Net cash / debt", net_cash),
            ("User-selected DCF years", config.projection_years),
            ("Effective DCF years", effective_projection_years),
            ("Market-implied stage-1 growth", market_implied["required_initial_growth"]),
            ("90% intrinsic range", percentiles["p95"] - percentiles["p05"]),
        ],
    }


def _compute_dcf_report(
    dataset: StockDataset,
    config: ValuationConfig,
    discount_rates: dict[str, float],
) -> dict[str, Any]:
    revenue_series = _extract_series(dataset.annual_income_stmt, _REVENUE_FIELDS)
    fcf_series = _compute_fcf_series(dataset)
    starting_fcf, normalization_note = _normalize_starting_fcf(fcf_series)
    growth_rate = _estimate_growth_rate(dataset, fcf_series, revenue_series)
    effective_projection_years = _effective_projection_years(dataset, config.projection_years, growth_rate)
    terminal_growth = _clamp(min(0.03, max(0.01, dataset.risk_free_rate * 0.65)), 0.01, 0.03)
    net_cash = dataset.cash_and_equivalents - dataset.total_debt

    base_growth_path = np.linspace(growth_rate, terminal_growth, effective_projection_years)
    deterministic_dcf = _discounted_cash_flow(
        starting_fcf=starting_fcf,
        growth_path=base_growth_path,
        discount_rate=discount_rates["wacc"],
        terminal_growth=terminal_growth,
        net_cash=net_cash,
        shares_outstanding=dataset.shares_outstanding,
    )

    rng = np.random.default_rng(42)
    growth_low = _clamp(growth_rate - 0.08, -0.10, 0.24)
    growth_high = _clamp(max(growth_rate + 0.08, growth_rate + 0.02), 0.05, 0.40)
    discount_low = _clamp(discount_rates["wacc"] - 0.02, 0.055, 0.18)
    discount_high = _clamp(discount_rates["wacc"] + 0.025, 0.07, 0.22)
    terminal_low = _clamp(terminal_growth - 0.008, 0.005, 0.03)
    terminal_high = _clamp(terminal_growth + 0.008, 0.01, 0.035)
    starting_fcf_low = starting_fcf * 0.85
    starting_fcf_high = starting_fcf * 1.15

    simulated_starting_fcf = _pert_sample(
        rng,
        minimum=starting_fcf_low,
        mode=starting_fcf,
        maximum=starting_fcf_high,
        size=config.simulations,
    )
    simulated_growth = _pert_sample(
        rng,
        minimum=growth_low,
        mode=growth_rate,
        maximum=growth_high,
        size=config.simulations,
    )
    simulated_discount = _pert_sample(
        rng,
        minimum=discount_low,
        mode=discount_rates["wacc"],
        maximum=discount_high,
        size=config.simulations,
    )
    simulated_terminal = _pert_sample(
        rng,
        minimum=terminal_low,
        mode=terminal_growth,
        maximum=terminal_high,
        size=config.simulations,
    )
    simulated_terminal = np.minimum(simulated_terminal, simulated_discount - 0.005)

    fade = np.linspace(0.0, 1.0, effective_projection_years)
    growth_paths = simulated_growth[:, None] + (simulated_terminal - simulated_growth)[:, None] * fade
    growth_paths = np.clip(growth_paths, -0.25, 0.45)

    fcf_paths = simulated_starting_fcf[:, None] * np.cumprod(1 + growth_paths, axis=1)
    years = np.arange(1, effective_projection_years + 1)
    discount_factors = (1 + simulated_discount[:, None]) ** years
    pv_fcf = np.sum(fcf_paths / discount_factors, axis=1)
    terminal_fcf = fcf_paths[:, -1] * (1 + simulated_terminal)
    terminal_value = terminal_fcf / np.maximum(simulated_discount - simulated_terminal, 0.005)
    pv_terminal = terminal_value / ((1 + simulated_discount) ** effective_projection_years)
    intrinsic_distribution = (pv_fcf + pv_terminal + net_cash) / dataset.shares_outstanding

    percentiles = {
        "p05": float(np.percentile(intrinsic_distribution, 5)),
        "p25": float(np.percentile(intrinsic_distribution, 25)),
        "p35": float(np.percentile(intrinsic_distribution, 35)),
        "p50": float(np.percentile(intrinsic_distribution, 50)),
        "p65": float(np.percentile(intrinsic_distribution, 65)),
        "p75": float(np.percentile(intrinsic_distribution, 75)),
        "p95": float(np.percentile(intrinsic_distribution, 95)),
    }
    probability_undervalued = float(np.mean(intrinsic_distribution > dataset.current_price))
    verdict, verdict_reason = _valuation_verdict(percentiles, dataset.current_price, probability_undervalued)
    scenario_table = _build_dcf_scenario_table(
        starting_fcf=starting_fcf,
        growth_rate=growth_rate,
        discount_rate=discount_rates["wacc"],
        terminal_growth=terminal_growth,
        net_cash=net_cash,
        shares_outstanding=dataset.shares_outstanding,
        projection_years=effective_projection_years,
        current_price=dataset.current_price,
    )
    sensitivity_matrix = _build_dcf_sensitivity_matrix(
        starting_fcf=starting_fcf,
        growth_rate=growth_rate,
        discount_rate=discount_rates["wacc"],
        terminal_growth=terminal_growth,
        net_cash=net_cash,
        shares_outstanding=dataset.shares_outstanding,
        projection_years=effective_projection_years,
    )
    uncertainty_band = percentiles["p95"] - percentiles["p05"]
    market_implied = _reverse_dcf_market_implied_growth(
        starting_fcf=starting_fcf,
        discount_rate=discount_rates["wacc"],
        terminal_growth=terminal_growth,
        net_cash=net_cash,
        shares_outstanding=dataset.shares_outstanding,
        current_price=dataset.current_price,
        projection_years=max(effective_projection_years, 7),
    )

    return {
        "method_family": "dcf",
        "method_used": "Monte Carlo FCF DCF",
        "current_price": dataset.current_price,
        "deterministic_intrinsic_value": float(deterministic_dcf["intrinsic_value"]),
        "intrinsic_value_distribution": intrinsic_distribution,
        "percentiles": percentiles,
        "margin_of_safety": percentiles["p50"] / dataset.current_price - 1,
        "probability_undervalued": probability_undervalued,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "assumptions": {
            "starting_fcf": starting_fcf,
            "growth_rate": growth_rate,
            "terminal_growth": terminal_growth,
            "risk_free_rate": discount_rates["risk_free_rate"],
            "beta_used": discount_rates["beta_used"],
            "cost_of_equity": discount_rates["cost_of_equity"],
            "cost_of_debt": discount_rates["cost_of_debt"],
            "wacc": discount_rates["wacc"],
            "net_cash": net_cash,
            "projection_years": config.projection_years,
            "effective_projection_years": effective_projection_years,
        },
        "ratios": _base_ratios(dataset),
        "market_implied": market_implied,
        "scenario_table": scenario_table,
        "scenario_title": "Bear / Base / Bull valuation scenarios",
        "sensitivity_matrix": sensitivity_matrix,
        "sensitivity_title": "DCF sensitivity to growth and WACC",
        "uncertainty": {
            "intrinsic_range_90": uncertainty_band,
            "range_vs_median": uncertainty_band / max(percentiles["p50"], 1e-6),
        },
        "normalization_note": normalization_note,
        "fundamental_series": {
            "Free Cash Flow": fcf_series,
            "Revenue": revenue_series,
        },
        "input_rows": [
            ("Method", "Monte Carlo FCF DCF"),
            ("Starting FCF", starting_fcf),
            ("Growth rate", growth_rate),
            ("Terminal growth", terminal_growth),
            ("Risk-free rate", discount_rates["risk_free_rate"]),
            ("Beta used", discount_rates["beta_used"]),
            ("Cost of equity", discount_rates["cost_of_equity"]),
            ("Cost of debt", discount_rates["cost_of_debt"]),
            ("WACC", discount_rates["wacc"]),
            ("Net cash / debt", net_cash),
            ("User-selected DCF years", config.projection_years),
            ("Effective DCF years", effective_projection_years),
            ("Market-implied stage-1 growth", market_implied["required_initial_growth"]),
            ("90% intrinsic range", uncertainty_band),
        ],
    }


def _compute_residual_income_report(
    dataset: StockDataset,
    config: ValuationConfig,
    discount_rates: dict[str, float],
) -> dict[str, Any]:
    book_value_series = _equity_per_share_series(dataset)
    eps_series = _eps_series(dataset)
    roe_series = _roe_series(dataset)

    if book_value_series.empty or eps_series.empty:
        raise ValuationError("Δεν υπάρχουν αρκετά book value / earnings στοιχεία για residual income valuation.")

    starting_book_value = float(book_value_series.iloc[-1])
    if starting_book_value <= 0:
        raise ValuationError("Το latest book value per share δεν είναι θετικό, άρα το residual income model δεν είναι ασφαλές.")

    starting_roe, normalization_note = _normalize_starting_roe(roe_series)
    rim_suitable, rim_note = _rim_suitability(dataset, starting_roe)
    if not rim_suitable:
        raise ValuationError(rim_note or "Το residual income model δεν είναι κατάλληλο για αυτό το ticker.")
    retention_rate = _estimate_retention_rate(dataset)
    terminal_growth = _clamp(min(0.02, max(0.0, dataset.risk_free_rate * 0.40)), 0.0, 0.02)
    terminal_roe = _clamp(
        discount_rates["cost_of_equity"] + 0.25 * (starting_roe - discount_rates["cost_of_equity"]),
        discount_rates["cost_of_equity"] - 0.015,
        discount_rates["cost_of_equity"] + 0.03,
    )

    deterministic = _residual_income_value(
        starting_book_value=starting_book_value,
        starting_roe=starting_roe,
        cost_of_equity=discount_rates["cost_of_equity"],
        retention_rate=retention_rate,
        terminal_roe=terminal_roe,
        terminal_growth=terminal_growth,
        projection_years=config.projection_years,
    )

    rng = np.random.default_rng(17)
    simulated_book = _pert_sample(
        rng,
        minimum=starting_book_value * 0.92,
        mode=starting_book_value,
        maximum=starting_book_value * 1.08,
        size=config.simulations,
    )
    simulated_roe = _pert_sample(
        rng,
        minimum=_clamp(starting_roe - 0.04, -0.12, 0.25),
        mode=starting_roe,
        maximum=_clamp(starting_roe + 0.04, -0.05, 0.32),
        size=config.simulations,
    )
    simulated_cost = _pert_sample(
        rng,
        minimum=_clamp(discount_rates["cost_of_equity"] - 0.015, 0.06, 0.20),
        mode=discount_rates["cost_of_equity"],
        maximum=_clamp(discount_rates["cost_of_equity"] + 0.02, 0.07, 0.22),
        size=config.simulations,
    )
    simulated_retention = _pert_sample(
        rng,
        minimum=_clamp(retention_rate - 0.15, 0.05, 0.80),
        mode=retention_rate,
        maximum=_clamp(retention_rate + 0.15, 0.15, 0.95),
        size=config.simulations,
    )
    simulated_terminal_roe = _pert_sample(
        rng,
        minimum=_clamp(terminal_roe - 0.02, discount_rates["cost_of_equity"] - 0.02, 0.25),
        mode=terminal_roe,
        maximum=_clamp(terminal_roe + 0.02, discount_rates["cost_of_equity"] - 0.01, 0.30),
        size=config.simulations,
    )

    intrinsic_distribution = np.empty(config.simulations, dtype=float)
    for index in range(config.simulations):
        intrinsic_distribution[index] = _rim_intrinsic_value(
            starting_book_value=float(simulated_book[index]),
            starting_roe=float(simulated_roe[index]),
            cost_of_equity=float(simulated_cost[index]),
            retention_rate=float(simulated_retention[index]),
            terminal_roe=float(_clamp(simulated_terminal_roe[index], simulated_cost[index] - 0.02, simulated_cost[index] + 0.04)),
            terminal_growth=terminal_growth,
            projection_years=config.projection_years,
        )

    percentiles = {
        "p05": float(np.percentile(intrinsic_distribution, 5)),
        "p25": float(np.percentile(intrinsic_distribution, 25)),
        "p35": float(np.percentile(intrinsic_distribution, 35)),
        "p50": float(np.percentile(intrinsic_distribution, 50)),
        "p65": float(np.percentile(intrinsic_distribution, 65)),
        "p75": float(np.percentile(intrinsic_distribution, 75)),
        "p95": float(np.percentile(intrinsic_distribution, 95)),
    }
    probability_undervalued = float(np.mean(intrinsic_distribution > dataset.current_price))
    verdict, verdict_reason = _valuation_verdict(percentiles, dataset.current_price, probability_undervalued)
    uncertainty_band = percentiles["p95"] - percentiles["p05"]

    return {
        "method_family": "residual_income",
        "method_used": "Monte Carlo Residual Income",
        "current_price": dataset.current_price,
        "deterministic_intrinsic_value": float(deterministic["intrinsic_value"]),
        "intrinsic_value_distribution": intrinsic_distribution,
        "percentiles": percentiles,
        "margin_of_safety": percentiles["p50"] / dataset.current_price - 1,
        "probability_undervalued": probability_undervalued,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "assumptions": {
            "starting_book_value": starting_book_value,
            "starting_roe": starting_roe,
            "terminal_roe": terminal_roe,
            "retention_rate": retention_rate,
            "terminal_growth": terminal_growth,
            "risk_free_rate": discount_rates["risk_free_rate"],
            "beta_used": discount_rates["beta_used"],
            "cost_of_equity": discount_rates["cost_of_equity"],
            "projection_years": config.projection_years,
        },
        "ratios": _base_ratios(dataset),
        "scenario_table": _build_rim_scenario_table(
            starting_book_value=starting_book_value,
            starting_roe=starting_roe,
            cost_of_equity=discount_rates["cost_of_equity"],
            retention_rate=retention_rate,
            terminal_roe=terminal_roe,
            terminal_growth=terminal_growth,
            projection_years=config.projection_years,
            current_price=dataset.current_price,
        ),
        "scenario_title": "Bear / Base / Bull residual-income scenarios",
        "sensitivity_matrix": _build_rim_sensitivity_matrix(
            starting_book_value=starting_book_value,
            starting_roe=starting_roe,
            cost_of_equity=discount_rates["cost_of_equity"],
            retention_rate=retention_rate,
            terminal_roe=terminal_roe,
            terminal_growth=terminal_growth,
            projection_years=config.projection_years,
        ),
        "sensitivity_title": "Residual income sensitivity to ROE and cost of equity",
        "uncertainty": {
            "intrinsic_range_90": uncertainty_band,
            "range_vs_median": uncertainty_band / max(percentiles["p50"], 1e-6),
        },
        "normalization_note": normalization_note,
        "fundamental_series": {
            "Net Income Per Share": eps_series,
            "Book Value Per Share": book_value_series,
        },
        "roe_series": roe_series,
        "input_rows": [
            ("Method", "Monte Carlo Residual Income"),
            ("Book value / share", starting_book_value),
            ("Starting ROE", starting_roe),
            ("Terminal ROE", terminal_roe),
            ("Retention rate", retention_rate),
            ("Terminal growth", terminal_growth),
            ("Risk-free rate", discount_rates["risk_free_rate"]),
            ("Beta used", discount_rates["beta_used"]),
            ("Cost of equity", discount_rates["cost_of_equity"]),
            ("90% intrinsic range", uncertainty_band),
        ],
    }


def _cross_check_summary(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    gap_vs_primary = secondary["percentiles"]["p50"] / max(primary["percentiles"]["p50"], 1e-6) - 1
    return {
        "method_used": secondary["method_used"],
        "median_intrinsic": secondary["percentiles"]["p50"],
        "margin_vs_market": secondary["margin_of_safety"],
        "gap_vs_primary": gap_vs_primary,
    }


def _valuation_confidence(
    dataset: StockDataset,
    primary: dict[str, Any],
    secondary: dict[str, Any] | None,
    classification: dict[str, Any],
) -> dict[str, Any]:
    score = 58
    reasons: list[str] = []
    method_family = primary["method_family"]
    uncertainty_ratio = float(primary["uncertainty"]["range_vs_median"])

    if method_family in {"fcff", "growth_fcff", "normalized_fcff", "dcf"}:
        fcff_series = primary["fundamental_series"].get("Observed FCFF")
        if fcff_series is None or not isinstance(fcff_series, pd.Series):
            fcff_series = primary["fundamental_series"].get("Free Cash Flow", pd.Series(dtype=float))
        fcf_years = len(fcff_series.dropna())
        if fcf_years >= 4:
            score += 12
            reasons.append("Υπάρχουν αρκετά annual operating fundamentals για το primary FCFF rail.")
        elif fcf_years <= 1:
            score -= 12
            reasons.append("Το FCFF anchor στηρίζεται σε πολύ περιορισμένο observed history.")

        if classification["high_growth"]:
            score -= 4
            reasons.append("Η εταιρεία είναι high-growth, άρα μεγαλύτερο μέρος της αξίας κάθεται σε future execution assumptions.")
        if classification["cyclical"]:
            score -= 8
            reasons.append("Το cyclical profile αυξάνει το model risk γύρω από margins και normalized earnings power.")
    else:
        roe_years = len(primary.get("roe_series", pd.Series(dtype=float)).dropna())
        if roe_years >= 3:
            score += 14
            reasons.append("Υπάρχουν αρκετά annual ROE observations για residual-income anchor.")
        if primary["assumptions"]["starting_book_value"] > 0:
            score += 10
            reasons.append("Το book value per share είναι θετικό, κάτι που βοηθά το residual income model.")

    if dataset.raw_info.get("sec_fundamentals_available"):
        score += 10
        reasons.append("Υπάρχει SEC fundamentals coverage για το συγκεκριμένο ticker.")
    else:
        score -= 15
        reasons.append("Λείπει πλήρες SEC fundamentals layer, άρα το intrinsic valuation πατά σε φτωχότερο anchor.")

    if classification["data_poor"]:
        score -= 18
        reasons.append("Το available annual history είναι περιορισμένο, άρα η valuation αξιοπιστία πέφτει αισθητά.")

    if dataset.raw_info.get("history_mode") == "proxy":
        score -= 8
        reasons.append("Το ticker έτρεξε με proxy history για το forecast context, κάτι που μειώνει τη συνολική robustness εικόνα.")

    if uncertainty_ratio <= 0.55:
        score += 6
        reasons.append("Το intrinsic range είναι σχετικά συγκρατημένο σε σχέση με τη median valuation εκτίμηση.")
    elif uncertainty_ratio >= 1.10:
        score -= 12
        reasons.append("Το intrinsic range είναι πολύ πλατύ, άρα η valuation έξοδος πρέπει να διαβαστεί σαν zone και όχι σαν σημειακή τιμή.")

    if secondary is not None:
        score += 8
        gap = abs(secondary["percentiles"]["p50"] / max(primary["percentiles"]["p50"], 1e-6) - 1)
        if gap <= 0.15:
            score += 12
            reasons.append("Το cross-check valuation model συμφωνεί σχετικά κοντά με το primary anchor.")
        elif gap <= 0.30:
            score += 5
            reasons.append("Το cross-check model είναι χρήσιμο αλλά δείχνει material dispersion έναντι του primary.")
        else:
            score -= 6
            reasons.append("Το cross-check model διαφωνεί αρκετά, άρα το fair-value range θέλει μεγαλύτερη ταπεινότητα.")

    score = int(max(0, min(score, 95)))
    if score >= 80:
        label = "High"
    elif score >= 55:
        label = "Medium"
    else:
        label = "Low"

    return {
        "score": score,
        "label": label,
        "reasons": reasons,
    }


def _data_quality_summary(dataset: StockDataset, classification: dict[str, Any]) -> dict[str, Any]:
    score = 48
    reasons: list[str] = []

    revenue_years = len(_extract_series(dataset.annual_income_stmt, _REVENUE_FIELDS).dropna())
    margin_years = len(_operating_margin_series(dataset).dropna())
    direct_history = dataset.raw_info.get("history_mode") == "direct"
    proxy_history = dataset.raw_info.get("history_mode") == "proxy"
    minimal_history = dataset.raw_info.get("history_mode") == "minimal"

    if dataset.raw_info.get("profile_available"):
        score += 8
        reasons.append("Υπάρχει usable company profile / reference layer.")
    else:
        score -= 8
        reasons.append("Λείπει πλήρες company profile layer.")

    if dataset.raw_info.get("sec_fundamentals_available"):
        score += 15
        reasons.append("Υπάρχει SEC annual fundamentals coverage.")
    else:
        score -= 15
        reasons.append("Δεν υπάρχει πλήρες SEC annual fundamentals coverage.")

    if direct_history:
        score += 12
        reasons.append("Η ανάλυση βασίζεται σε direct daily price history του ίδιου του ticker.")
    elif proxy_history:
        score -= 8
        reasons.append("Το price history ήρθε μέσω proxy mode, άρα το αποτέλεσμα είναι πιο exploratory.")
    elif minimal_history:
        score -= 18
        reasons.append("Δεν υπάρχει κανονικό direct/proxy history, άρα το dataset είναι αδύναμο για πλήρη ανάλυση.")

    if len(dataset.price_history) >= 252:
        score += 8
        reasons.append("Υπάρχει τουλάχιστον ένα έτος usable daily history.")
    else:
        score -= 8
        reasons.append("Το διαθέσιμο daily history είναι αρκετά περιορισμένο.")

    if revenue_years >= 5 and margin_years >= 4:
        score += 10
        reasons.append("Υπάρχουν αρκετά χρόνια revenue / margin history για structural valuation work.")
    elif revenue_years >= 3:
        score += 4
        reasons.append("Υπάρχει μέτρια θεμελιώδης ιστορία, αλλά όχι ιδανική.")
    else:
        score -= 10
        reasons.append("Τα annual fundamentals είναι λίγα για robust long-form valuation.")

    if len(dataset.context_price_history) >= 4:
        score += 5
        reasons.append("Υπάρχει αρκετό market-context coverage από ETFs / proxies.")

    if classification["data_poor"]:
        score -= 12
        reasons.append("Το classifier τοποθετεί το ticker σε data-poor regime.")

    if dataset.market_cap > 0 and dataset.market_cap < 1_000_000_000:
        score -= 8
        reasons.append("Το ticker είναι μικρότερης κεφαλαιοποίησης, κάτι που συνήθως αυξάνει data fragility και model noise.")

    score = int(max(5, min(score, 95)))
    if score >= 80:
        label = "High"
    elif score >= 55:
        label = "Medium"
    else:
        label = "Low"

    return {
        "score": score,
        "label": label,
        "reasons": reasons,
    }


def build_valuation_report(dataset: StockDataset, config: ValuationConfig) -> dict[str, Any]:
    if dataset.current_price <= 0:
        raise ValuationError("Η τρέχουσα τιμή της μετοχής δεν είναι έγκυρη.")
    if dataset.shares_outstanding <= 0:
        raise ValuationError("Δεν βρέθηκαν αρκετά στοιχεία για shares outstanding.")

    discount_rates = _estimate_discount_rate(dataset, config)
    classification = _classify_company(dataset, config.tax_rate)
    framework, framework_reason = _choose_valuation_framework(classification)
    fcff_report: dict[str, Any] | None = None
    rim_report: dict[str, Any] | None = None
    errors: list[str] = []

    if framework == "hard_to_value":
        raise ValuationError(framework_reason)

    if framework in {"fcff", "growth_fcff", "normalized_fcff"}:
        try:
            fcff_report = _compute_fcff_report(dataset, config, discount_rates, classification, framework)
        except ValuationError as exc:
            errors.append(f"FCFF: {exc}")

        if classification["book_relevant"] and classification["has_positive_earnings"] and not classification["asset_light"]:
            try:
                rim_report = _compute_residual_income_report(dataset, config, discount_rates)
            except ValuationError as exc:
                errors.append(f"Residual income: {exc}")
    elif framework == "residual_income":
        try:
            rim_report = _compute_residual_income_report(dataset, config, discount_rates)
        except ValuationError as exc:
            errors.append(f"Residual income: {exc}")
    else:
        errors.append(f"Unsupported framework: {framework}")

    if fcff_report is None and rim_report is None:
        raise ValuationError(" | ".join(errors) if errors else framework_reason)

    if framework == "residual_income":
        primary = rim_report
        secondary = None
    else:
        primary = fcff_report or rim_report
        secondary = rim_report if fcff_report is not None and rim_report is not None else None

    if primary is None:
        raise ValuationError("Δεν βρέθηκε usable valuation output.")

    classification_summary = {
        "is_financial": classification["is_financial"],
        "is_reit": classification["is_reit"],
        "cyclical": classification["cyclical"],
        "asset_light": classification["asset_light"],
        "high_growth": classification["high_growth"],
        "has_positive_fcf": classification["has_positive_fcf"],
        "has_positive_earnings": classification["has_positive_earnings"],
        "book_relevant": classification["book_relevant"],
        "data_poor": classification["data_poor"],
        "revenue_cagr": classification["revenue_cagr"],
        "reasons": classification["reasons"],
    }

    primary["classification"] = classification_summary
    primary["framework_selected"] = framework
    primary["framework_reason"] = framework_reason
    primary["data_quality"] = _data_quality_summary(dataset, classification)
    primary["cross_check"] = _cross_check_summary(primary, secondary) if secondary is not None else None
    primary["confidence"] = _valuation_confidence(dataset, primary, secondary, classification)
    primary["secondary_method"] = secondary["method_used"] if secondary is not None else None
    primary["valuation_stack_note"] = (
        "Adaptive valuation stack: company classification -> framework selection -> probabilistic fair-value range. "
        "Primary anchor είναι το FCFF rail, ενώ το residual-income model εμφανίζεται μόνο ως cross-check όταν το book value έχει νόημα."
        if primary["method_family"] in {"fcff", "growth_fcff", "normalized_fcff"} and secondary is not None
        else "Adaptive valuation stack: το primary anchor είναι FCFF + WACC με revenue, margin και reinvestment drivers."
        if primary["method_family"] in {"fcff", "growth_fcff", "normalized_fcff"}
        else "Adaptive valuation stack: το primary anchor είναι residual income, επειδή το equity/book framework ταιριάζει περισσότερο σε αυτό το profile."
    )
    return primary
