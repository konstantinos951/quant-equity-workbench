from __future__ import annotations

from pathlib import Path
import hashlib
import os
from typing import Any

_MPL_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from services.data_service import (
    DataRetrievalError,
    estimate_analysis_budget,
    fetch_stock_dataset,
)
from services.forecasting import ForecastModelError, build_forecast_report
from services.formatters import format_currency, format_multiple, format_percent
from services.models import ForecastConfig, ValuationConfig
from services.valuation_engine import ValuationError, build_valuation_report

load_dotenv()


st.set_page_config(
    page_title="Quant Equity Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #eef3f7;
            --bg-soft: rgba(250, 253, 255, 0.90);
            --ink: #12263a;
            --muted: #536579;
            --accent: #0f766e;
            --accent-soft: rgba(15, 118, 110, 0.12);
            --warm: #c26a12;
            --danger: #b91c1c;
            --info: #1d4ed8;
            --line: rgba(18, 38, 58, 0.10);
            --shadow: 0 24px 60px rgba(18, 38, 58, 0.12);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 24%),
                radial-gradient(circle at 85% 8%, rgba(29, 78, 216, 0.12), transparent 22%),
                radial-gradient(circle at 15% 90%, rgba(194, 106, 18, 0.10), transparent 20%),
                linear-gradient(180deg, #edf4fa 0%, #dfe9f4 100%);
        }
        section[data-testid="stMain"] {
            color: var(--ink);
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }
        section[data-testid="stMain"] p,
        section[data-testid="stMain"] li,
        section[data-testid="stMain"] label,
        section[data-testid="stMain"] span,
        section[data-testid="stMain"] div[data-testid="stMarkdownContainer"] {
            color: var(--ink);
        }
        section[data-testid="stMain"] h1,
        section[data-testid="stMain"] h2,
        section[data-testid="stMain"] h3,
        section[data-testid="stMain"] h4,
        section[data-testid="stMain"] h5,
        section[data-testid="stMain"] h6 {
            color: var(--ink);
        }
        div[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(49, 204, 191, 0.18), transparent 28%),
                radial-gradient(circle at 85% 12%, rgba(59, 130, 246, 0.20), transparent 20%),
                linear-gradient(180deg, #0f1c2f 0%, #14273b 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        div[data-testid="stSidebar"] * {
            color: #f4f8fb;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 1.1rem;
            margin-bottom: 1.2rem;
        }
        .panel {
            padding: 1.35rem 1.45rem;
            border-radius: 28px;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(247, 251, 255, 0.90));
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }
        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.20em;
            font-size: 0.76rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.45rem;
        }
        .hero-title {
            font-size: 2.45rem;
            line-height: 0.98;
            color: var(--ink);
            margin: 0 0 0.95rem 0;
            font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
            letter-spacing: -0.03em;
        }
        .hero-copy, .micro-copy {
            color: var(--muted);
            font-size: 0.99rem;
            line-height: 1.66;
        }
        .budget-card {
            padding: 1.05rem 1.05rem;
            border-radius: 22px;
            background: linear-gradient(145deg, rgba(15, 118, 110, 0.22), rgba(29, 78, 216, 0.18));
            border: 1px solid rgba(255, 255, 255, 0.12);
            margin-bottom: 1rem;
            box-shadow: 0 16px 36px rgba(4, 17, 32, 0.22);
        }
        .budget-number {
            font-size: 2.15rem;
            font-weight: 800;
            color: #f7fbff;
            line-height: 1.1;
        }
        .budget-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: rgba(244, 248, 251, 0.74);
            margin-bottom: 0.35rem;
        }
        .sidebar-note {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.10);
            margin-bottom: 0.75rem;
            font-size: 0.92rem;
            line-height: 1.45;
            backdrop-filter: blur(12px);
        }
        .summary-chip {
            display: inline-block;
            padding: 0.40rem 0.78rem;
            border-radius: 999px;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            background: linear-gradient(145deg, rgba(15, 118, 110, 0.10), rgba(29, 78, 216, 0.10));
            color: var(--ink);
            font-weight: 700;
            font-size: 0.82rem;
            border: 1px solid rgba(18, 38, 58, 0.08);
        }
        .workflow-step {
            padding: 0.78rem 0.9rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(238, 244, 250, 0.86), rgba(247, 251, 255, 0.80));
            border: 1px solid var(--line);
            margin-bottom: 0.6rem;
        }
        .metric-strip {
            padding: 0.95rem 1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.90);
            border: 1px solid var(--line);
            box-shadow: 0 16px 32px rgba(18, 38, 58, 0.07);
        }
        .section-title {
            font-size: 1.1rem;
            color: var(--ink);
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        .horizon-card {
            padding: 1.02rem 1.08rem;
            border-radius: 22px;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(245, 250, 255, 0.92));
            border: 1px solid var(--line);
            box-shadow: 0 16px 36px rgba(18, 38, 58, 0.08);
        }
        .horizon-card h4 {
            margin: 0 0 0.35rem 0;
            color: var(--ink);
        }
        .horizon-card p {
            margin: 0.2rem 0;
            color: var(--muted);
        }
        .guide-card {
            padding: 1.05rem 1.1rem;
            border-radius: 24px;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(245, 250, 255, 0.90));
            border: 1px solid var(--line);
            min-height: 100%;
            box-shadow: 0 18px 40px rgba(18, 38, 58, 0.07);
        }
        .guide-card h4 {
            margin-top: 0;
            color: var(--ink);
            letter-spacing: -0.01em;
        }
        .status-pill-good, .status-pill-neutral, .status-pill-warn {
            display: inline-block;
            padding: 0.36rem 0.74rem;
            border-radius: 999px;
            font-size: 0.80rem;
            font-weight: 800;
            margin-right: 0.45rem;
        }
        .status-pill-good {
            background: rgba(15, 118, 110, 0.15);
            color: #0f766e;
        }
        .status-pill-neutral {
            background: rgba(29, 78, 216, 0.13);
            color: #1d4ed8;
        }
        .status-pill-warn {
            background: rgba(194, 106, 18, 0.16);
            color: #b45309;
        }
        section[data-testid="stMain"] div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(245, 250, 255, 0.90));
            border: 1px solid rgba(18, 38, 58, 0.08);
            border-radius: 22px;
            padding: 0.9rem 1rem;
            box-shadow: 0 16px 34px rgba(18, 38, 58, 0.08);
        }
        section[data-testid="stMain"] div[data-testid="stMetric"] label,
        section[data-testid="stMain"] div[data-testid="stMetric"] div {
            color: var(--ink);
        }
        section[data-testid="stMain"] .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        section[data-testid="stMain"] .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(18, 38, 58, 0.08);
            border-radius: 16px;
            color: var(--ink);
            padding: 0.25rem 0.6rem;
        }
        section[data-testid="stMain"] .stTabs [aria-selected="true"] {
            background: linear-gradient(145deg, rgba(15, 118, 110, 0.16), rgba(29, 78, 216, 0.14));
            color: var(--ink);
        }
        section[data-testid="stMain"] div[data-testid="stAlert"],
        section[data-testid="stMain"] div[data-testid="stDataFrame"] {
            color: var(--ink);
        }
        @media (max-width: 1100px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _run_analysis(
    query: str,
    history_years: int,
    projection_years: int,
    valuation_simulations: int,
    forecast_simulations: int,
    equity_risk_premium: float,
    debt_spread: float,
    tax_rate: float,
    force_refresh: bool,
    use_live_quote: bool,
) -> tuple[object, dict, dict]:
    dataset = fetch_stock_dataset(
        query,
        history_years=history_years,
        force_refresh=force_refresh,
        use_live_quote=use_live_quote,
    )
    valuation: dict[str, Any]
    forecast: dict[str, Any]
    valuation_error: str | None = None
    forecast_error: str | None = None

    try:
        valuation = build_valuation_report(
            dataset,
            ValuationConfig(
                projection_years=projection_years,
                simulations=valuation_simulations,
                equity_risk_premium=equity_risk_premium,
                debt_spread=debt_spread,
                tax_rate=tax_rate,
            ),
        )
        valuation["available"] = True
    except ValuationError as exc:
        valuation_error = str(exc)
        valuation = {"available": False, "error": valuation_error}

    try:
        forecast = build_forecast_report(
            dataset,
            ForecastConfig(
                simulations=forecast_simulations,
                max_horizon=21,
                markov_regimes=3,
            ),
        )
        forecast["available"] = True
    except ForecastModelError as exc:
        forecast_error = str(exc)
        forecast = {"available": False, "error": forecast_error}

    if not valuation["available"] and not forecast["available"]:
        raise DataRetrievalError(
            f"Δεν ήταν δυνατό να βγει usable valuation ή forecast. "
            f"Valuation: {valuation_error or 'N/A'} | Forecast: {forecast_error or 'N/A'}"
        )
    return dataset, valuation, forecast


@st.cache_data(show_spinner=False, ttl=1800)
def _run_analysis_cached(
    query: str,
    history_years: int,
    projection_years: int,
    valuation_simulations: int,
    forecast_simulations: int,
    equity_risk_premium: float,
    debt_spread: float,
    tax_rate: float,
    use_live_quote: bool,
) -> tuple[object, dict, dict]:
    return _run_analysis(
        query=query,
        history_years=history_years,
        projection_years=projection_years,
        valuation_simulations=valuation_simulations,
        forecast_simulations=forecast_simulations,
        equity_risk_premium=equity_risk_premium,
        debt_spread=debt_spread,
        tax_rate=tax_rate,
        force_refresh=False,
        use_live_quote=use_live_quote,
    )


def run_analysis(
    query: str,
    history_years: int,
    projection_years: int,
    valuation_simulations: int,
    forecast_simulations: int,
    equity_risk_premium: float,
    debt_spread: float,
    tax_rate: float,
    force_refresh: bool,
    use_live_quote: bool,
) -> tuple[object, dict, dict]:
    if force_refresh:
        return _run_analysis(
            query=query,
            history_years=history_years,
            projection_years=projection_years,
            valuation_simulations=valuation_simulations,
            forecast_simulations=forecast_simulations,
            equity_risk_premium=equity_risk_premium,
            debt_spread=debt_spread,
            tax_rate=tax_rate,
            force_refresh=True,
            use_live_quote=use_live_quote,
        )

    return _run_analysis_cached(
        query=query,
        history_years=history_years,
        projection_years=projection_years,
        valuation_simulations=valuation_simulations,
        forecast_simulations=forecast_simulations,
        equity_risk_premium=equity_risk_premium,
        debt_spread=debt_spread,
        tax_rate=tax_rate,
        use_live_quote=use_live_quote,
    )


def _settings_signature(settings: dict[str, Any]) -> str:
    raw = "|".join(f"{key}={settings[key]}" for key in sorted(settings))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compute_profile(valuation_simulations: int, forecast_simulations: int, history_years: int) -> dict[str, str]:
    score = (valuation_simulations / 1000) * 0.9 + (forecast_simulations / 1000) * 1.1 + history_years * 0.45
    if score <= 11:
        return {
            "label": "Light",
            "detail": "Γρήγορο run, καλό για exploratory checks και δοκιμές.",
        }
    if score <= 18:
        return {
            "label": "Balanced",
            "detail": "Η πιο ισορροπημένη επιλογή για base M2 MacBook Air.",
        }
    return {
        "label": "Heavy",
        "detail": "Ακόμα M2-safe, αλλά θα θέλει πιο υπομονετικό run για smoother distributions.",
    }


def _settings_impact_lines(
    history_years: int,
    projection_years: int,
    valuation_simulations: int,
    forecast_simulations: int,
    use_live_quote: bool,
    force_refresh: bool,
) -> list[str]:
    history_note = (
        "Πιο πρόσφατο και πιο αντιδραστικό δείγμα."
        if history_years <= 4
        else "Πιο σταθερό history για regime detection και volatility calibration."
    )
    projection_note = (
        "Μικρότερη εξάρτηση από το terminal value."
        if projection_years <= 5
        else "Μεγαλύτερο βάρος στις μακροχρόνιες υποθέσεις του DCF."
    )
    valuation_note = (
        "Πιο γρήγορο valuation distribution."
        if valuation_simulations <= 4000
        else "Πιο λείο Monte Carlo valuation histogram και καλύτερα percentiles."
    )
    forecast_note = (
        "Γρήγορο fan chart για short-term testing."
        if forecast_simulations <= 5000
        else "Πιο σταθερά probabilistic bands και καλύτερο tail sampling."
    )
    quote_note = "Θα ζητήσει extra live quote call." if use_live_quote else "Θα χρησιμοποιήσει latest close από cached/FMP history."
    refresh_note = "Θα αγνοήσει πλήρως το cache." if force_refresh else "Θα αξιοποιήσει local cache όπου υπάρχει."
    return [
        f"`History {history_years}y`: {history_note}",
        f"`DCF {projection_years}y`: {projection_note}",
        f"`Valuation sims {valuation_simulations:,}`: {valuation_note}",
        f"`Forecast sims {forecast_simulations:,}`: {forecast_note}",
        f"`Live quote`: {quote_note}",
        f"`Refresh`: {refresh_note}",
    ]


def make_price_forecast_chart(price_history: pd.DataFrame, forecast: dict) -> go.Figure:
    history = price_history["Close"].dropna().tail(252 * 2)
    fan_chart = forecast["fan_chart"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="Historical price",
            line={"color": "#17342f", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fan_chart.index,
            y=fan_chart["p95"],
            mode="lines",
            line={"color": "rgba(15,118,110,0)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fan_chart.index,
            y=fan_chart["p05"],
            mode="lines",
            name="90% band",
            line={"color": "rgba(15,118,110,0)"},
            fill="tonexty",
            fillcolor="rgba(15,118,110,0.14)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fan_chart.index,
            y=fan_chart["p75"],
            mode="lines",
            line={"color": "rgba(180,83,9,0)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fan_chart.index,
            y=fan_chart["p25"],
            mode="lines",
            name="50% band",
            line={"color": "rgba(180,83,9,0)"},
            fill="tonexty",
            fillcolor="rgba(180,83,9,0.20)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fan_chart.index,
            y=fan_chart["p50"],
            mode="lines",
            name="Median forecast",
            line={"color": "#b45309", "width": 2.2, "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Historical price and probabilistic forecast fan",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        legend_orientation="h",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=470,
    )
    return fig


def make_valuation_distribution_chart(valuation: dict, current_price: float) -> go.Figure:
    distribution = valuation["intrinsic_value_distribution"]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=distribution,
            nbinsx=50,
            marker={"color": "#0f766e"},
            opacity=0.80,
            name="DCF simulations",
        )
    )
    fig.add_vline(
        x=current_price,
        line_dash="dot",
        line_color="#b91c1c",
        annotation_text="Market price",
        annotation_position="top right",
    )
    fig.add_vline(
        x=valuation["percentiles"]["p50"],
        line_dash="dash",
        line_color="#1d4ed8",
        annotation_text="Median intrinsic",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Monte Carlo DCF intrinsic value distribution",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        bargap=0.04,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=420,
    )
    return fig


def make_regime_chart(forecast: dict) -> go.Figure:
    probabilities = forecast["smoothed_probabilities"].tail(252)
    fig = go.Figure()
    palette = ["#b91c1c", "#b45309", "#0f766e"]

    for color, column in zip(palette, probabilities.columns):
        fig.add_trace(
            go.Scatter(
                x=probabilities.index,
                y=probabilities[column],
                stackgroup="one",
                mode="lines",
                line={"width": 1.2, "color": color},
                name=column,
            )
        )

    fig.update_layout(
        title="Regime probabilities over the last year",
        yaxis={"tickformat": ".0%", "range": [0, 1]},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=380,
    )
    return fig


def make_scenario_chart(valuation: dict, currency: str, current_price: float) -> go.Figure:
    scenarios = pd.DataFrame(valuation["scenario_table"])
    colors = ["#b91c1c", "#1d4ed8", "#0f766e"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=scenarios["scenario"],
            y=scenarios["intrinsic_value"],
            marker_color=colors,
            text=[format_currency(value, currency) for value in scenarios["intrinsic_value"]],
            textposition="outside",
            name="Intrinsic value",
        )
    )
    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="#b45309",
        annotation_text="Market price",
        annotation_position="top left",
    )
    fig.update_layout(
        title=valuation.get("scenario_title", "Bear / Base / Bull valuation scenarios"),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=380,
    )
    return fig


def make_sensitivity_heatmap(valuation: dict, currency: str) -> go.Figure:
    matrix = valuation["sensitivity_matrix"]
    text = [[format_currency(value, currency) for value in row] for row in matrix.values]
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=matrix.values,
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                colorscale="Tealgrn",
                text=text,
                texttemplate="%{text}",
                hovertemplate="Intrinsic value: %{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=valuation.get("sensitivity_title", "Valuation sensitivity"),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=420,
    )
    return fig


def make_fundamental_trend_chart(valuation: dict, currency: str) -> go.Figure:
    series_map: dict[str, pd.Series] = valuation.get("fundamental_series", {})
    entries = [(label, series) for label, series in series_map.items() if isinstance(series, pd.Series) and not series.empty]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if entries:
        first_label, first_series = entries[0]
        fig.add_trace(
            go.Bar(
                x=first_series.index,
                y=first_series.values,
                name=first_label,
                marker_color="#0f766e",
                opacity=0.75,
            ),
            secondary_y=False,
        )

    if len(entries) > 1:
        second_label, second_series = entries[1]
        fig.add_trace(
            go.Scatter(
                x=second_series.index,
                y=second_series.values,
                mode="lines+markers",
                name=second_label,
                line={"color": "#b45309", "width": 2.2},
            ),
            secondary_y=True,
        )

    if not entries:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No sufficient annual statement history returned by the provider.",
            showarrow=False,
        )

    fig.update_layout(
        title="Reported valuation anchors",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=400,
    )
    if entries:
        fig.update_yaxes(title_text=f"{entries[0][0]} ({currency})", secondary_y=False)
    if len(entries) > 1:
        fig.update_yaxes(title_text=f"{entries[1][0]} ({currency})", secondary_y=True)
    return fig


def format_mixed_value(label: str, value: Any, currency: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "N/A"

    label_lower = label.lower()
    if any(
        token in label_lower
        for token in (
            "rate",
            "growth",
            "margin",
            "probability",
            "roe",
            "wacc",
            "retention",
            "equity",
        )
    ):
        return format_percent(float(value))
    if "score" in label_lower and "valuation" not in label_lower:
        return f"{int(round(float(value)))}"
    return format_currency(float(value), currency)


def render_takeaways(title: str, items: list[str]) -> None:
    rendered = "".join(f"<p>{item}</p>" for item in items)
    st.markdown(
        f"""
        <div class="guide-card">
            <h4>{title}</h4>
            {rendered}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_horizon_cards(horizon_summary: list[dict], currency: str) -> None:
    columns = st.columns(3)
    for column, horizon in zip(columns, horizon_summary):
        with column:
            st.markdown(
                f"""
                <div class="horizon-card">
                    <h4>{horizon["label"]}</h4>
                    <p><strong>Median:</strong> {format_currency(horizon["median"], currency)}</p>
                    <p><strong>90% range:</strong> {format_currency(horizon["p05"], currency)} - {format_currency(horizon["p95"], currency)}</p>
                    <p><strong>P(upside):</strong> {format_percent(horizon["probability_upside"])}</p>
                    <p><strong>P(>+5%):</strong> {format_percent(horizon["probability_up_5pct"])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_sidebar_budget(budget: dict[str, Any], compute_profile: dict[str, str]) -> None:
    st.sidebar.markdown(
        f"""
        <div class="budget-card">
            <div class="budget-label">Before You Run</div>
            <div class="budget-number">{budget["label"]}</div>
            <div class="micro-copy" style="color: rgba(247,243,237,0.86);">
                Estimated fresh API calls now
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-note">
            <strong>Compute profile:</strong> {compute_profile["label"]}<br/>
            {compute_profile["detail"]}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(budget["note"])
    for item in budget["breakdown"]:
        state = "new call" if item["will_call"] else "cache"
        st.sidebar.caption(f"`{item['label']}`: {state}. {item['detail']}")


def render_intro(budget: dict[str, Any], compute_profile: dict[str, str]) -> None:
    st.markdown(
        f"""
        <div class="hero-grid">
            <div class="panel">
                <div class="hero-kicker">Hybrid Valuation + Forecasting</div>
                <div class="hero-title">Quant Equity Workbench</div>
                <div class="hero-copy">
                    Το app ενώνει <strong>DCF valuation</strong>, <strong>Monte Carlo uncertainty</strong>,
                    <strong>Markov + GARCH probabilistic forecasting</strong> και
                    <strong>cross-asset proxy context</strong> σε ένα ενιαίο dashboard για short-term research.
                </div>
                <div class="hero-copy" style="margin-top: 0.7rem;">
                    Όταν τρέξει η ανάλυση, θα δεις μαζί:
                    τρέχουσα τιμή, estimated intrinsic value, valuation scenarios, uncertainty bands,
                    forecast για 1 εβδομάδα, 3 εβδομάδες και 1 μήνα, το τρέχον regime της μετοχής,
                    καθώς και reliability diagnostics για valuation και forecasting.
                    Στο free mode, τα prices/context έρχονται κυρίως από <strong>FMP free</strong>, με <strong>Yahoo/Stooq fallback</strong> όπου χρειάζεται,
                    και τα fundamentals από <strong>SEC filings</strong> όπου υπάρχουν.
                </div>
            </div>
            <div class="panel">
                <div class="section-title">Run Checklist</div>
                <div class="workflow-step"><strong>1.</strong> Γράψε κατά προτίμηση ticker, π.χ. <code>AAPL</code>, για πιο ακριβές call budget.</div>
                <div class="workflow-step"><strong>2.</strong> Κοίτα το <strong>estimated API calls</strong> πριν πατήσεις run.</div>
                <div class="workflow-step"><strong>3.</strong> Διάλεξε αν θες live quote ή refresh από το API.</div>
                <div class="workflow-step"><strong>4.</strong> Πάτησε <strong>Run Analysis</strong> και διάβασε πρώτα το snapshot, μετά valuation και τέλος forecast.</div>
                <div class="workflow-step"><strong>Now:</strong> {budget["label"]} estimated calls | <strong>Compute:</strong> {compute_profile["label"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_guide_tab(
    settings_impact: list[str],
    budget: dict[str, Any],
    compute_profile: dict[str, str],
) -> None:
    guide_left, guide_right = st.columns(2)
    with guide_left:
        st.markdown(
            """
            <div class="guide-card">
                <h4>Τι κάνει το κάθε μοντέλο</h4>
                <p><strong>DCF:</strong> δίνει valuation anchor από future cash flows, WACC και terminal growth.</p>
                <p><strong>Residual income:</strong> λειτουργεί σαν fallback / cross-check όταν το FCF δεν είναι αρκετά καθαρό.</p>
                <p><strong>Monte Carlo DCF:</strong> δείχνει uncertainty αντί για μία μόνο “σωστή” τιμή.</p>
                <p><strong>Markov regimes:</strong> ξεχωρίζει bear / neutral / bull market states στα returns.</p>
                <p><strong>AR(1) + GARCH / GJR-GARCH-t:</strong> μοντελοποιεί short-term drift, volatility clustering, leverage asymmetry και fat tails.</p>
                <p><strong>Bootstrap + jumps:</strong> κρατά historical asymmetry και shock risk στα forecast paths.</p>
                <p><strong>Cross-asset proxies:</strong> χρησιμοποιεί SPY, QQQ, IWM, GLD, TLT, USO, UUP και sector ETF για market behavior, risk-off και geopolitical stress proxies.</p>
                <p><strong>Coverage label:</strong> δείχνει αν το συγκεκριμένο ticker έχει high, medium ή fragile data coverage στο free stack.</p>
                <p><strong>Open-access GPR index:</strong> όταν είναι διαθέσιμο, προσθέτει daily geopolitical risk signal από το dataset των Caldara-Iacoviello.</p>
                <p><strong>Calibration panel:</strong> ελέγχει αν το forecast engine ήταν πρόσφατα πιο αξιόπιστο ή πιο fragile.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with guide_right:
        st.markdown(
            f"""
            <div class="guide-card">
                <h4>Τι αλλάζει τώρα με τα settings σου</h4>
                <p><strong>Estimated calls:</strong> {budget["label"]}</p>
                <p><strong>Compute profile:</strong> {compute_profile["label"]}</p>
                <p class="micro-copy">{budget["note"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for line in settings_impact:
            st.markdown(f"- {line}")


def main() -> None:
    inject_styles()

    with st.sidebar:
        st.markdown("## Analysis Controls")
        query = st.text_input(
            "Ticker or company",
            value=st.session_state.get("query_value", "AAPL"),
            help="Ticker δίνει πιο ακριβές call estimate και αποφεύγει extra name-lookup ambiguity.",
        )
        history_years = st.slider(
            "Historical window (years)",
            min_value=3,
            max_value=10,
            value=5,
            help="Μεγαλύτερο history βοηθά τα regime και volatility models να σταθεροποιηθούν.",
        )
        projection_years = st.slider(
            "DCF forecast years",
            min_value=4,
            max_value=10,
            value=5,
            help="Περισσότερα projection years αυξάνουν το βάρος των assumptions στο DCF.",
        )
        valuation_simulations = st.slider(
            "DCF simulations",
            min_value=2000,
            max_value=12000,
            step=1000,
            value=5000,
            help="Περισσότερα simulations κάνουν πιο λείο το valuation distribution αλλά αυξάνουν τον χρόνο run.",
        )
        forecast_simulations = st.slider(
            "Forecast simulations",
            min_value=2000,
            max_value=15000,
            step=1000,
            value=6000,
            help="Περισσότερα paths δίνουν πιο σταθερό fan chart και tail-risk estimate.",
        )
        use_live_quote = st.toggle(
            "Use live quote",
            value=False,
            help="Κάνει ένα extra API call για πιο φρέσκια spot price.",
        )
        force_refresh = st.toggle(
            "Refresh from API now",
            value=False,
            help="Παρακάμπτει το local cache και ζητά νέα responses από το API.",
        )

        with st.expander("Advanced assumptions"):
            equity_risk_premium = st.slider(
                "Equity risk premium",
                min_value=0.03,
                max_value=0.08,
                value=0.0525,
                step=0.0025,
                format="%.4f",
                help="Υψηλότερο ERP ανεβάζει cost of equity και χαμηλώνει το DCF valuation.",
            )
            debt_spread = st.slider(
                "Debt spread above risk-free",
                min_value=0.005,
                max_value=0.05,
                value=0.0175,
                step=0.0025,
                format="%.4f",
                help="Ανεβάζει το cost of debt και πιέζει το WACC προς τα πάνω.",
            )
            tax_rate = st.slider(
                "Tax rate",
                min_value=0.10,
                max_value=0.35,
                value=0.21,
                step=0.01,
                format="%.2f",
                help="Χρησιμοποιείται στο after-tax cost of debt μέσα στο WACC.",
            )

    settings = {
        "query": query.strip(),
        "history_years": history_years,
        "projection_years": projection_years,
        "valuation_simulations": valuation_simulations,
        "forecast_simulations": forecast_simulations,
        "equity_risk_premium": equity_risk_premium,
        "debt_spread": debt_spread,
        "tax_rate": tax_rate,
        "force_refresh": force_refresh,
        "use_live_quote": use_live_quote,
    }
    current_signature = _settings_signature(settings)
    budget = estimate_analysis_budget(
        query=query,
        history_years=history_years,
        use_live_quote=use_live_quote,
        force_refresh=force_refresh,
    )
    compute_profile = _compute_profile(valuation_simulations, forecast_simulations, history_years)
    settings_impact = _settings_impact_lines(
        history_years=history_years,
        projection_years=projection_years,
        valuation_simulations=valuation_simulations,
        forecast_simulations=forecast_simulations,
        use_live_quote=use_live_quote,
        force_refresh=force_refresh,
    )

    render_sidebar_budget(budget, compute_profile)

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-note">
                <strong>Research note:</strong><br/>
                Το app προτιμά data-efficient μοντέλα που έχουν καλό trade-off ανάμεσα σε robustness,
                explainability και local runtime σε base M2 laptop.
            </div>
            """,
            unsafe_allow_html=True,
        )
        run_clicked = st.button("Run Analysis", use_container_width=True, type="primary")

    render_intro(budget, compute_profile)

    if run_clicked:
        if not settings["query"]:
            st.error("Πρώτα γράψε ticker ή όνομα εταιρείας.")
        else:
            try:
                with st.spinner("Loading FMP data, building valuation distribution, and fitting forecast models..."):
                    dataset, valuation, forecast = run_analysis(**settings)
                st.session_state["analysis_payload"] = {
                    "dataset": dataset,
                    "valuation": valuation,
                    "forecast": forecast,
                }
                st.session_state["analysis_signature"] = current_signature
                st.session_state["query_value"] = settings["query"]
            except (DataRetrievalError, ValuationError, ForecastModelError) as exc:
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover
                st.error(f"Unexpected error: {exc}")

    analysis_payload = st.session_state.get("analysis_payload")
    if not analysis_payload:
        st.info("Ρύθμισε τα controls αριστερά, δες πρώτα πόσα calls θα γίνουν, και μετά πάτα Run Analysis.")
        render_guide_tab(settings_impact, budget, compute_profile)
        return

    dataset = analysis_payload["dataset"]
    valuation = analysis_payload["valuation"]
    forecast = analysis_payload["forecast"]
    stored_signature = st.session_state.get("analysis_signature")
    valuation_available = bool(valuation.get("available"))
    forecast_available = bool(forecast.get("available"))

    if stored_signature != current_signature:
        st.warning("Τα αποτελέσματα που βλέπεις είναι από τα τελευταία submitted settings. Αν άλλαξες sliders, πάτα ξανά Run Analysis.")

    top_columns = st.columns(8)
    price_metric_label = "Live Quote" if dataset.raw_info.get("live_quote_used") else "Latest Close"
    top_columns[0].metric(price_metric_label, format_currency(dataset.current_price, dataset.currency))
    top_columns[1].metric(
        "Primary Value",
        format_currency(valuation["percentiles"]["p50"], dataset.currency) if valuation_available else "N/A",
    )
    top_columns[2].metric(
        "Margin of Safety",
        format_percent(valuation["margin_of_safety"]) if valuation_available else "N/A",
    )
    top_columns[3].metric(
        "P(Undervalued)",
        format_percent(valuation["probability_undervalued"]) if valuation_available else "N/A",
    )
    top_columns[4].metric(
        "Valuation Confidence",
        valuation["confidence"]["label"] if valuation_available else "N/A",
        delta=f"{valuation['confidence']['score']}/100" if valuation_available else None,
    )
    top_columns[5].metric(
        "Forecast Quality",
        forecast["calibration"]["label"] if forecast_available else "N/A",
        delta=f"{forecast['calibration']['score']}/100" if forecast_available else None,
    )
    top_columns[6].metric(
        "Current Regime",
        forecast["current_regime"] if forecast_available else "N/A",
        delta=format_percent(forecast["current_regime_probability"]) if forecast_available else None,
    )
    top_columns[7].metric("Coverage", dataset.raw_info.get("coverage_label", "N/A"))

    verdict_label = valuation["verdict"] if valuation_available else "Forecast Only"
    verdict_class = "status-pill-good" if verdict_label == "Undervalued" else "status-pill-warn" if verdict_label == "Overvalued" else "status-pill-neutral"
    st.markdown(
        f"""
        <span class="{verdict_class}">{verdict_label}</span>
        <span class="summary-chip">{dataset.symbol}</span>
        <span class="summary-chip">{dataset.sector}</span>
        <span class="summary-chip">{dataset.raw_info.get("coverage_label", "Coverage N/A")}</span>
        <span class="summary-chip">{dataset.raw_info.get("price_provider", "Provider N/A")} price feed</span>
        {f'<span class="summary-chip">Proxy via {dataset.raw_info.get("history_proxy_symbol")}</span>' if dataset.raw_info.get("history_mode") == "proxy" else ''}
        <span class="summary-chip">{dataset.raw_info.get("data_source", "Data provider")}</span>
        <span class="summary-chip">{valuation['method_used'] if valuation_available else 'No valuation anchor'}</span>
        <span class="summary-chip">{forecast["model_name"] if forecast_available else 'Forecast unavailable'}</span>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(dataset.company_name)
    if valuation_available:
        st.write(valuation["verdict_reason"])
        st.caption(valuation.get("valuation_stack_note", ""))
    else:
        st.warning(valuation.get("error", "Δεν βγήκε valuation για αυτό το ticker με τα διαθέσιμα free fundamentals."))
    if dataset.description:
        st.caption(dataset.description)
    if dataset.raw_info.get("history_mode") == "proxy":
        st.warning(
            f"Δεν βρέθηκε direct history για το {dataset.symbol}. Το forecast τρέχει σε proxy mode με βάση το {dataset.raw_info.get('history_proxy_symbol')}. "
            "Το αποτέλεσμα είναι usable για exploratory research, αλλά όχι τόσο ισχυρό όσο direct symbol history."
        )
    if dataset.raw_info.get("history_mode") == "minimal":
        st.warning(
            f"Δεν βρέθηκε usable direct ή proxy history για το {dataset.symbol}. "
            "Το app συνεχίζει με current-price anchor και fundamentals όπου υπάρχουν, αλλά όχι με κανονικό forecast."
        )
    st.caption(
        f"Price basis: {dataset.raw_info.get('price_basis', 'N/A')} | "
        f"Coverage: {dataset.raw_info.get('coverage_note', 'N/A')} | "
        f"Actual fresh calls: {dataset.raw_info.get('actual_network_calls', 'N/A')} | "
        f"Cache hints: {', '.join(dataset.raw_info.get('cache_messages', [])) or 'Fresh API responses'}"
    )

    tabs = st.tabs(["Overview", "Valuation", "Forecast", "Reliability", "Guide"])

    with tabs[0]:
        st.markdown("### Snapshot")
        overview_left, overview_right = st.columns([1.35, 1.0])
        with overview_left:
            if forecast_available:
                st.plotly_chart(make_price_forecast_chart(dataset.price_history, forecast), use_container_width=True)
            else:
                st.warning(forecast.get("error", "Forecast unavailable."))
        with overview_right:
            if valuation_available:
                st.plotly_chart(
                    make_valuation_distribution_chart(valuation, dataset.current_price),
                    use_container_width=True,
                )
            else:
                st.info("Το valuation panel θα εμφανιστεί μόνο όταν υπάρχουν αρκετά SEC fundamentals για αυτό το ticker.")

        st.markdown("### Short-Term Forecast")
        if forecast_available:
            render_horizon_cards(forecast["horizon_summary"], dataset.currency)
        else:
            st.warning(forecast.get("error", "Forecast unavailable."))

        summary_left, summary_right = st.columns(2)
        with summary_left:
            if forecast_available:
                render_takeaways("Model Takeaways", forecast.get("takeaways", []))
            else:
                st.info("Μόλις βγει forecast, εδώ θα εμφανιστούν τα βασικά takeaways του μοντέλου.")
        with summary_right:
            overview_lines = [
                "Το valuation λειτουργεί ως anchor για τη θεωρητική αξία και όχι σαν μία μοναδική βεβαιότητα.",
                "Το short-term forecast λειτουργεί ως probability map. Το κέντρο του fan chart είναι πιθανότερο path, όχι υπόσχεση.",
                "Όταν valuation και forecast συμφωνούν, το signal είναι πιο καθαρό. Όταν διαφωνούν, θέλει μικρότερο position size και μεγαλύτερη πειθαρχία.",
            ]
            if valuation_available and valuation.get("cross_check"):
                overview_lines.append(
                    f"Το secondary valuation model ({valuation['cross_check']['method_used']}) κάθεται περίπου {format_percent(valuation['cross_check']['gap_vs_primary'])} από το primary anchor."
                )
            render_takeaways("Research Posture", overview_lines)

    with tabs[1]:
        st.markdown("### Valuation Anchor")
        if not valuation_available:
            st.warning(valuation.get("error", "Δεν βγήκε fair value για αυτό το ticker με το free-mode fundamentals stack."))
            st.info("Το free valuation δουλεύει καλύτερα σε U.S. companies με καθαρά SEC filings και αρκετό historical companyfacts coverage.")
        else:
            st.markdown(
                f"""
                <span class="summary-chip">{valuation['method_used']}</span>
                <span class="summary-chip">Confidence {valuation['confidence']['label']} ({valuation['confidence']['score']}/100)</span>
                {f'<span class="summary-chip">Cross-check: {valuation["cross_check"]["method_used"]}</span>' if valuation.get("cross_check") else ''}
                """,
                unsafe_allow_html=True,
            )
            valuation_left, valuation_right = st.columns([1.0, 1.25])
            with valuation_left:
                st.plotly_chart(
                    make_scenario_chart(valuation, dataset.currency, dataset.current_price),
                    use_container_width=True,
                )
            with valuation_right:
                st.plotly_chart(
                    make_sensitivity_heatmap(valuation, dataset.currency),
                    use_container_width=True,
                )

            lower_left, lower_right = st.columns([1.05, 1.0])
            with lower_left:
                st.plotly_chart(
                    make_fundamental_trend_chart(valuation, dataset.currency),
                    use_container_width=True,
                )
            with lower_right:
                assumptions = valuation["assumptions"]
                st.markdown("### Valuation Inputs")
                st.dataframe(
                    pd.DataFrame(
                        [
                            (label, format_mixed_value(label, value, dataset.currency))
                            for label, value in valuation.get("input_rows", [])
                        ],
                        columns=["Input", "Value"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### Market Ratios")
                ratios = valuation["ratios"]
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("P/E", format_multiple(ratios["trailing_pe"])),
                            ("PEG", format_multiple(ratios["peg_ratio"])),
                            ("P/B", format_multiple(ratios["price_to_book"])),
                            ("EV/EBITDA", format_multiple(ratios["enterprise_to_ebitda"])),
                            ("ROE", format_percent(ratios["return_on_equity"])),
                            ("Profit margin", format_percent(ratios["profit_margin"])),
                        ],
                        columns=["Ratio", "Value"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                if valuation.get("cross_check"):
                    cross_check = valuation["cross_check"]
                    st.markdown("### Cross-Check")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                ("Method", cross_check["method_used"]),
                                ("Median intrinsic", format_currency(cross_check["median_intrinsic"], dataset.currency)),
                                ("Margin vs market", format_percent(cross_check["margin_vs_market"])),
                                ("Gap vs primary", format_percent(cross_check["gap_vs_primary"])),
                            ],
                            columns=["Metric", "Value"],
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.markdown("### Scenario Table")
            scenario_frame = pd.DataFrame(valuation["scenario_table"]).copy()
            scenario_frame["intrinsic_value"] = scenario_frame["intrinsic_value"].map(lambda value: format_currency(value, dataset.currency))
            scenario_frame["margin_vs_market"] = scenario_frame["margin_vs_market"].map(format_percent)
            st.dataframe(
                scenario_frame.rename(
                    columns={
                        "scenario": "Scenario",
                        "intrinsic_value": "Intrinsic Value",
                        "margin_vs_market": "Margin vs Market",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(valuation["normalization_note"])

    with tabs[2]:
        st.markdown("### Forecast Diagnostics")
        if not forecast_available:
            st.warning(forecast.get("error", "Forecast unavailable."))
        else:
            forecast_left, forecast_right = st.columns([1.0, 1.15])
            with forecast_left:
                st.plotly_chart(make_regime_chart(forecast), use_container_width=True)
            with forecast_right:
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("Model", forecast["model_name"]),
                            ("Current regime", forecast["current_regime"]),
                            ("Current regime probability", format_percent(forecast["current_regime_probability"])),
                            ("Forecast quality", f"{forecast['calibration']['label']} ({forecast['calibration']['score']}/100)"),
                            ("Daily jump probability", format_percent(forecast["jump_process"]["jump_probability"])),
                            ("Context drift bias", format_percent(forecast["context_bias"])),
                            ("Opportunity score", f"{forecast['context_scores']['opportunity']:.2f}"),
                            ("Stress score", f"{forecast['context_scores']['stress']:.2f}"),
                            ("Geopolitical proxy score", f"{forecast['context_scores']['geopolitical']:.2f}"),
                            ("Behavior score", f"{forecast['context_scores']['behavior']:.2f}"),
                            ("AR(1) intercept", f"{forecast['mean_model']['intercept']:.5f}"),
                            ("AR(1) phi", f"{forecast['mean_model']['phi']:.3f}"),
                        ],
                        columns=["Diagnostic", "Value"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("### Regime Transition Matrix")
                st.dataframe(forecast["transition_matrix"].style.format("{:.1%}"), use_container_width=True)

            regime_stats = pd.DataFrame(forecast["regime_stats"]).copy()
            regime_stats["mean_return"] = regime_stats["mean_return"].map(format_percent)
            regime_stats["std_return"] = regime_stats["std_return"].map(format_percent)
            st.markdown("### Regime Stats")
            st.dataframe(
                regime_stats.rename(
                    columns={
                        "label": "Regime",
                        "mean_return": "Mean Daily Return",
                        "std_return": "Daily Volatility",
                    }
                )[["Regime", "Mean Daily Return", "Daily Volatility"]],
                use_container_width=True,
                hide_index=True,
            )

            context_left, context_right = st.columns([1.0, 1.0])
            with context_left:
                st.markdown("### Proxy Snapshot")
                st.caption(forecast["proxy_methodology"])
                st.caption(
                    "Υψηλότερο `Geopolitical proxy score` συνήθως σημαίνει ότι το μοντέλο βλέπει πιο έντονο risk-off περιβάλλον, "
                    "με περισσότερη πιθανότητα για volatility spikes και αμυντική συμπεριφορά τιμής."
                )
                if not forecast["feature_snapshot"].empty:
                    st.dataframe(forecast["feature_snapshot"], use_container_width=True)
                else:
                    st.info("Δεν ήταν διαθέσιμα αρκετά proxy features για snapshot.")
            with context_right:
                st.markdown("### Context Drivers")
                st.caption(
                    f"Assets used: {', '.join(forecast['context_assets_used']) or 'none'} | "
                    f"Macro signals: {', '.join(forecast.get('macro_context_used', [])) or 'none'}"
                )
                st.caption(
                    "Αν ανέβει το `GPR`, ο χρυσός, το πετρέλαιο και το δολάριο μαζί, το μοντέλο το διαβάζει σαν "
                    "πιθανή κλιμάκωση κρίσης ή πολεμικού stress. Αντίθετα, χαμηλότερο GPR και καλύτερο breadth "
                    "δίνουν πιο risk-on bias."
                )
                if not forecast["context_driver_table"].empty:
                    driver_table = forecast["context_driver_table"].copy()
                    driver_table["beta"] = driver_table["beta"].map(lambda value: f"{value:.4f}")
                    driver_table["current_z"] = driver_table["current_z"].map(lambda value: f"{value:.2f}")
                    driver_table["drift_contribution"] = driver_table["drift_contribution"].map(format_percent)
                    st.dataframe(driver_table, use_container_width=True, hide_index=True)
                else:
                    st.info("Δεν βγήκαν σταθεροί exogenous drivers από τα διαθέσιμα proxies.")

    with tabs[3]:
        st.markdown("### Reliability & Evidence")
        reliability_left, reliability_right = st.columns([1.0, 1.05])
        with reliability_left:
            if valuation_available:
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("Primary valuation method", valuation["method_used"]),
                            ("Confidence", f"{valuation['confidence']['label']} ({valuation['confidence']['score']}/100)"),
                            ("Verdict", valuation["verdict"]),
                            ("90% intrinsic range", format_currency(valuation["uncertainty"]["intrinsic_range_90"], dataset.currency)),
                            ("Stack note", valuation.get("valuation_stack_note", "N/A")),
                        ],
                        columns=["Valuation evidence", "Value"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                if valuation["confidence"]["reasons"]:
                    render_takeaways("Why The Valuation Confidence Looks Like This", valuation["confidence"]["reasons"])
            else:
                st.info("Δεν υπάρχουν αρκετά fundamentals για valuation reliability diagnostics σε αυτό το ticker.")

        with reliability_right:
            if forecast_available:
                calibration = forecast["calibration"]
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("Forecast quality", f"{calibration['label']} ({calibration['score']}/100)"),
                            ("Calibration sample", f"{calibration['sample_days']} days"),
                            ("Sign accuracy", format_percent(calibration["sign_accuracy"])),
                            ("90% interval coverage", format_percent(calibration["interval_90_coverage"])),
                            ("Left-tail hit rate", format_percent(calibration["left_tail_hit_rate"])),
                            ("Predicted daily vol", format_percent(calibration["predicted_daily_vol"])),
                            ("Realized daily vol", format_percent(calibration["realized_daily_vol"])),
                            ("Realized / predicted vol", f"{calibration['vol_ratio']:.2f}x" if calibration["vol_ratio"] is not None else "N/A"),
                        ],
                        columns=["Forecast evidence", "Value"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(calibration["summary"])
            else:
                st.info("Δεν υπάρχουν forecast reliability diagnostics για αυτό το run.")

        if forecast_available:
            evidence_left, evidence_right = st.columns([1.0, 1.0])
            with evidence_left:
                st.markdown("### Volatility Ensemble")
                if not forecast["volatility_models"].empty:
                    volatility_models = forecast["volatility_models"].copy()
                    volatility_models["weight"] = volatility_models["weight"].map(format_percent)
                    volatility_models["aic"] = volatility_models["aic"].map(lambda value: f"{value:.1f}" if pd.notna(value) else "N/A")
                    volatility_models["bic"] = volatility_models["bic"].map(lambda value: f"{value:.1f}" if pd.notna(value) else "N/A")
                    st.dataframe(volatility_models, use_container_width=True, hide_index=True)
                else:
                    st.info("Δεν ήταν διαθέσιμο volatility ensemble breakdown.")
            with evidence_right:
                st.markdown("### Data Coverage")
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("Coverage label", dataset.raw_info.get("coverage_label", "N/A")),
                            ("Coverage note", dataset.raw_info.get("coverage_note", "N/A")),
                            ("Primary price provider", dataset.raw_info.get("price_provider", "N/A")),
                            ("SEC fundamentals available", "Yes" if dataset.raw_info.get("sec_fundamentals_available") else "No"),
                            ("Context assets used", ", ".join(forecast.get("context_assets_used", [])) or "none"),
                            ("Macro context used", ", ".join(forecast.get("macro_context_used", [])) or "none"),
                            ("Sector ETF proxy", dataset.raw_info.get("sector_etf") or "none"),
                            ("Cache hints", ", ".join(dataset.raw_info.get("cache_messages", [])) or "Fresh API responses"),
                        ],
                        columns=["Coverage item", "Value"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[4]:
        render_guide_tab(settings_impact, budget, compute_profile)


if __name__ == "__main__":
    main()
