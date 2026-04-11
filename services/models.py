from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class StockDataset:
    symbol: str
    company_name: str
    currency: str
    sector: str
    industry: str
    description: str
    current_price: float
    market_cap: float
    shares_outstanding: float
    beta: float
    revenue_growth: float | None
    earnings_growth: float | None
    profit_margin: float | None
    trailing_pe: float | None
    peg_ratio: float | None
    price_to_book: float | None
    enterprise_to_ebitda: float | None
    return_on_equity: float | None
    cash_and_equivalents: float
    total_debt: float
    risk_free_rate: float
    price_history: pd.DataFrame
    context_price_history: dict[str, pd.DataFrame]
    context_macro_series: dict[str, pd.Series]
    annual_cashflow: pd.DataFrame
    annual_balance_sheet: pd.DataFrame
    annual_income_stmt: pd.DataFrame
    raw_info: dict[str, Any]


@dataclass(slots=True)
class ValuationConfig:
    projection_years: int = 5
    simulations: int = 4000
    equity_risk_premium: float = 0.0525
    debt_spread: float = 0.0175
    tax_rate: float = 0.21


@dataclass(slots=True)
class ForecastConfig:
    simulations: int = 5000
    max_horizon: int = 21
    markov_regimes: int = 3
