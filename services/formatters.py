from __future__ import annotations

import math


_CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "EUR ",
    "GBP": "GBP ",
}


def format_currency(value: float | None, currency: str = "USD", decimals: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"

    symbol = _CURRENCY_SYMBOLS.get((currency or "USD").upper(), f"{currency or 'USD'} ")
    absolute_value = abs(value)

    if absolute_value >= 1_000_000_000_000:
        body = f"{value / 1_000_000_000_000:.2f}T"
    elif absolute_value >= 1_000_000_000:
        body = f"{value / 1_000_000_000:.2f}B"
    elif absolute_value >= 1_000_000:
        body = f"{value / 1_000_000:.2f}M"
    elif absolute_value >= 1_000:
        body = f"{value:,.{decimals}f}"
    else:
        body = f"{value:.{decimals}f}"

    return f"{symbol}{body}"


def format_percent(value: float | None, decimals: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_multiple(value: float | None, decimals: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{decimals}f}x"
