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
_REVENUE_FIELDS = (
    "Total Revenue",
    "Operating Revenue",
)
_NET_INCOME_FIELDS = ("Net Income",)
_BOOK_VALUE_FIELDS = ("Stockholders Equity",)


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
    return _clamp(base_growth, -0.05, 0.18)


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
    cost_of_equity = dataset.risk_free_rate + max(dataset.beta, 0.6) * config.equity_risk_premium
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
    median_intrinsic: float,
    current_price: float,
    probability_undervalued: float,
) -> tuple[str, str]:
    gap = (median_intrinsic - current_price) / current_price

    if gap >= 0.15 and probability_undervalued >= 0.65:
        return "Undervalued", "Η κατανομή του valuation δείχνει αρκετά υψηλή πιθανότητα η αγορά να τιμολογεί κάτω από την εύλογη αξία."
    if gap <= -0.15 and probability_undervalued <= 0.35:
        return "Overvalued", "Η κατανομή του valuation δείχνει ότι η τρέχουσα τιμή ενσωματώνει ήδη αρκετά αισιόδοξες προσδοκίες."
    return "Fairly Priced", "Η τρέχουσα τιμή φαίνεται κοντά στη μέση ζώνη της εκτιμώμενης εύλογης αξίας."


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
    growth_path = np.clip(growth_path, -0.25, 0.30)
    scenario = _discounted_cash_flow(
        starting_fcf=starting_fcf,
        growth_path=growth_path,
        discount_rate=discount_rate,
        terminal_growth=min(terminal_growth, discount_rate - 0.005),
        net_cash=net_cash,
        shares_outstanding=shares_outstanding,
    )
    return float(scenario["intrinsic_value"])


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
            growth_rate=_clamp(growth, -0.10, 0.24),
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
            adjusted_growth = _clamp(growth_rate + growth_shift, -0.10, 0.24)
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


def _compute_dcf_report(
    dataset: StockDataset,
    config: ValuationConfig,
    discount_rates: dict[str, float],
) -> dict[str, Any]:
    revenue_series = _extract_series(dataset.annual_income_stmt, _REVENUE_FIELDS)
    fcf_series = _compute_fcf_series(dataset)
    starting_fcf, normalization_note = _normalize_starting_fcf(fcf_series)
    growth_rate = _estimate_growth_rate(dataset, fcf_series, revenue_series)
    terminal_growth = _clamp(min(0.03, max(0.01, dataset.risk_free_rate * 0.65)), 0.01, 0.03)
    net_cash = dataset.cash_and_equivalents - dataset.total_debt

    base_growth_path = np.linspace(growth_rate, terminal_growth, config.projection_years)
    deterministic_dcf = _discounted_cash_flow(
        starting_fcf=starting_fcf,
        growth_path=base_growth_path,
        discount_rate=discount_rates["wacc"],
        terminal_growth=terminal_growth,
        net_cash=net_cash,
        shares_outstanding=dataset.shares_outstanding,
    )

    rng = np.random.default_rng(42)
    growth_low = _clamp(growth_rate - 0.06, -0.10, 0.20)
    growth_high = _clamp(growth_rate + 0.06, -0.02, 0.25)
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

    fade = np.linspace(0.0, 1.0, config.projection_years)
    growth_paths = simulated_growth[:, None] + (simulated_terminal - simulated_growth)[:, None] * fade
    growth_paths = np.clip(growth_paths, -0.25, 0.30)

    fcf_paths = simulated_starting_fcf[:, None] * np.cumprod(1 + growth_paths, axis=1)
    years = np.arange(1, config.projection_years + 1)
    discount_factors = (1 + simulated_discount[:, None]) ** years
    pv_fcf = np.sum(fcf_paths / discount_factors, axis=1)
    terminal_fcf = fcf_paths[:, -1] * (1 + simulated_terminal)
    terminal_value = terminal_fcf / np.maximum(simulated_discount - simulated_terminal, 0.005)
    pv_terminal = terminal_value / ((1 + simulated_discount) ** config.projection_years)
    intrinsic_distribution = (pv_fcf + pv_terminal + net_cash) / dataset.shares_outstanding

    percentiles = {
        "p05": float(np.percentile(intrinsic_distribution, 5)),
        "p25": float(np.percentile(intrinsic_distribution, 25)),
        "p50": float(np.percentile(intrinsic_distribution, 50)),
        "p75": float(np.percentile(intrinsic_distribution, 75)),
        "p95": float(np.percentile(intrinsic_distribution, 95)),
    }
    probability_undervalued = float(np.mean(intrinsic_distribution > dataset.current_price))
    verdict, verdict_reason = _valuation_verdict(percentiles["p50"], dataset.current_price, probability_undervalued)
    scenario_table = _build_dcf_scenario_table(
        starting_fcf=starting_fcf,
        growth_rate=growth_rate,
        discount_rate=discount_rates["wacc"],
        terminal_growth=terminal_growth,
        net_cash=net_cash,
        shares_outstanding=dataset.shares_outstanding,
        projection_years=config.projection_years,
        current_price=dataset.current_price,
    )
    sensitivity_matrix = _build_dcf_sensitivity_matrix(
        starting_fcf=starting_fcf,
        growth_rate=growth_rate,
        discount_rate=discount_rates["wacc"],
        terminal_growth=terminal_growth,
        net_cash=net_cash,
        shares_outstanding=dataset.shares_outstanding,
        projection_years=config.projection_years,
    )
    uncertainty_band = percentiles["p95"] - percentiles["p05"]

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
            "cost_of_equity": discount_rates["cost_of_equity"],
            "cost_of_debt": discount_rates["cost_of_debt"],
            "wacc": discount_rates["wacc"],
            "net_cash": net_cash,
            "projection_years": config.projection_years,
        },
        "ratios": _base_ratios(dataset),
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
            ("Cost of equity", discount_rates["cost_of_equity"]),
            ("Cost of debt", discount_rates["cost_of_debt"]),
            ("WACC", discount_rates["wacc"]),
            ("Net cash / debt", net_cash),
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
        "p50": float(np.percentile(intrinsic_distribution, 50)),
        "p75": float(np.percentile(intrinsic_distribution, 75)),
        "p95": float(np.percentile(intrinsic_distribution, 95)),
    }
    probability_undervalued = float(np.mean(intrinsic_distribution > dataset.current_price))
    verdict, verdict_reason = _valuation_verdict(percentiles["p50"], dataset.current_price, probability_undervalued)
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
) -> dict[str, Any]:
    score = 35
    reasons: list[str] = []

    if primary["method_family"] == "dcf":
        fcf_years = len(primary["fundamental_series"].get("Free Cash Flow", pd.Series(dtype=float)).dropna())
        if fcf_years >= 4:
            score += 15
            reasons.append("Υπάρχουν αρκετά annual FCF observations για το primary DCF.")
        if fcf_years >= 3 and (primary["fundamental_series"]["Free Cash Flow"].tail(3) > 0).all():
            score += 10
            reasons.append("Τα τελευταία reported FCFs είναι θετικά, άρα το DCF anchor είναι πιο σταθερό.")
    else:
        roe_years = len(primary.get("roe_series", pd.Series(dtype=float)).dropna())
        if roe_years >= 3:
            score += 15
            reasons.append("Υπάρχουν αρκετά annual ROE observations για residual income anchor.")
        if primary["assumptions"]["starting_book_value"] > 0:
            score += 10
            reasons.append("Το book value per share είναι θετικό, κάτι που βοηθά το residual income model.")

    if dataset.raw_info.get("sec_fundamentals_available"):
        score += 10
        reasons.append("Υπάρχει SEC fundamentals coverage για το συγκεκριμένο ticker.")

    if secondary is not None:
        score += 10
        gap = abs(secondary["percentiles"]["p50"] / max(primary["percentiles"]["p50"], 1e-6) - 1)
        if gap <= 0.15:
            score += 15
            reasons.append("Το cross-check valuation model συμφωνεί σχετικά κοντά με το primary anchor.")
        elif gap <= 0.30:
            score += 8
            reasons.append("Το cross-check model είναι χρήσιμο αλλά δείχνει material dispersion έναντι του primary.")
        else:
            reasons.append("Το cross-check model διαφωνεί αρκετά, άρα το fair-value range θέλει μεγαλύτερη ταπεινότητα.")

    score = int(max(0, min(score, 95)))
    if score >= 80:
        label = "High"
    elif score >= 60:
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
    dcf_report: dict[str, Any] | None = None
    rim_report: dict[str, Any] | None = None
    errors: list[str] = []

    try:
        dcf_report = _compute_dcf_report(dataset, config, discount_rates)
    except ValuationError as exc:
        errors.append(f"DCF: {exc}")

    try:
        rim_report = _compute_residual_income_report(dataset, config, discount_rates)
    except ValuationError as exc:
        errors.append(f"Residual income: {exc}")

    if dcf_report is None and rim_report is None:
        raise ValuationError(" | ".join(errors))

    primary = dcf_report or rim_report
    secondary = rim_report if dcf_report is not None and rim_report is not None else None
    if primary is None:
        raise ValuationError("Δεν βρέθηκε usable valuation output.")

    primary["cross_check"] = _cross_check_summary(primary, secondary) if secondary is not None else None
    primary["confidence"] = _valuation_confidence(dataset, primary, secondary)
    primary["secondary_method"] = secondary["method_used"] if secondary is not None else None
    primary["valuation_stack_note"] = (
        "Primary valuation anchor: Monte Carlo FCF DCF. Cross-check: Residual Income."
        if secondary is not None and primary["method_family"] == "dcf"
        else "Primary valuation anchor: Monte Carlo Residual Income."
    )
    return primary
