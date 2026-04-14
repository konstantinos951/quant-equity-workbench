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

from services.decision_engine import build_long_term_report, build_short_term_report
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
    page_title="Equity Research Desk",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #081423;
            --navy-soft: #10253b;
            --paper: #f6f1e8;
            --paper-strong: #fbf8f1;
            --ink: #172437;
            --muted: #5e6c7c;
            --gold: #ae8450;
            --gold-soft: rgba(174, 132, 80, 0.12);
            --teal: #0e6665;
            --teal-soft: rgba(14, 102, 101, 0.12);
            --danger: #a53d32;
            --info: #234d84;
            --line: rgba(23, 36, 55, 0.12);
            --line-strong: rgba(23, 36, 55, 0.18);
            --shadow: 0 22px 48px rgba(7, 18, 33, 0.14);
        }
        .stApp {
            background:
                linear-gradient(180deg, #e8e0d3 0%, #ece5da 14%, #f3ede4 34%, #f6f1e8 100%);
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
                linear-gradient(180deg, #07121f 0%, #0c1c2f 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        div[data-testid="stSidebar"] * {
            color: #eef3f8;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 1rem;
            margin-bottom: 1.15rem;
        }
        .panel {
            padding: 1.4rem 1.45rem;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(251, 248, 241, 0.96), rgba(247, 242, 232, 0.94));
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
        }
        .panel-hero {
            background:
                linear-gradient(160deg, rgba(8, 20, 35, 0.98), rgba(14, 33, 52, 0.96));
            border: 1px solid rgba(174, 132, 80, 0.18);
            color: #f4ede2;
        }
        .panel-ops {
            background:
                linear-gradient(180deg, rgba(251, 248, 241, 0.98), rgba(245, 239, 230, 0.96));
        }
        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-size: 0.72rem;
            font-weight: 700;
            color: #d2b180;
            margin-bottom: 0.55rem;
        }
        .hero-title {
            font-size: 2.7rem;
            line-height: 0.98;
            color: #f9f3ea;
            margin: 0 0 0.9rem 0;
            font-family: "Canela", "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
            letter-spacing: -0.035em;
        }
        .hero-copy, .micro-copy {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.68;
        }
        .panel-hero .hero-copy,
        .panel-hero .micro-copy {
            color: rgba(244, 237, 226, 0.84);
        }
        .hero-band {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }
        .hero-band-item {
            padding: 0.9rem 0.95rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(210, 177, 128, 0.14);
        }
        .hero-band-label {
            display: block;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: rgba(244, 237, 226, 0.58);
            margin-bottom: 0.35rem;
        }
        .hero-band-value {
            font-size: 0.95rem;
            color: #f4ede2;
            font-weight: 700;
            line-height: 1.35;
        }
        .section-overline {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            color: var(--gold);
            font-weight: 800;
            margin-bottom: 0.45rem;
        }
        .panel-title {
            font-size: 1.35rem;
            font-family: "Iowan Old Style", "Palatino Linotype", serif;
            letter-spacing: -0.02em;
            color: var(--ink);
            margin-bottom: 0.7rem;
        }
        .budget-card {
            padding: 1.05rem 1.05rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
            border: 1px solid rgba(210, 177, 128, 0.16);
            margin-bottom: 1rem;
            box-shadow: 0 16px 36px rgba(2, 10, 18, 0.22);
        }
        .budget-number {
            font-size: 2.2rem;
            font-weight: 800;
            color: #f7efe3;
            line-height: 1.1;
        }
        .budget-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: rgba(242, 233, 221, 0.62);
            margin-bottom: 0.35rem;
        }
        .sidebar-note {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(210, 177, 128, 0.12);
            margin-bottom: 0.75rem;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .summary-chip {
            display: inline-block;
            padding: 0.40rem 0.78rem;
            border-radius: 999px;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            background: linear-gradient(145deg, rgba(174, 132, 80, 0.10), rgba(8, 20, 35, 0.04));
            color: var(--ink);
            font-weight: 700;
            font-size: 0.82rem;
            border: 1px solid rgba(23, 36, 55, 0.10);
        }
        .workflow-step {
            padding: 0.85rem 0.92rem;
            border-radius: 15px;
            background: linear-gradient(180deg, rgba(255, 251, 245, 0.98), rgba(246, 240, 229, 0.92));
            border: 1px solid var(--line);
            margin-bottom: 0.6rem;
        }
        .metric-strip {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(251, 248, 241, 0.92);
            border: 1px solid var(--line);
            box-shadow: 0 16px 32px rgba(18, 38, 58, 0.06);
        }
        .section-title {
            font-size: 1.0rem;
            color: var(--ink);
            margin-bottom: 0.5rem;
            font-weight: 800;
            letter-spacing: 0.02em;
        }
        .horizon-card {
            padding: 1.08rem 1.08rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(251, 248, 241, 0.98), rgba(245, 239, 230, 0.94));
            border: 1px solid var(--line);
            box-shadow: 0 14px 28px rgba(18, 38, 58, 0.06);
        }
        .horizon-card h4 {
            margin: 0 0 0.35rem 0;
            color: var(--ink);
            font-family: "Iowan Old Style", "Palatino Linotype", serif;
        }
        .horizon-card p {
            margin: 0.2rem 0;
            color: var(--muted);
        }
        .guide-card {
            padding: 1.05rem 1.1rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(251, 248, 241, 0.96), rgba(245, 239, 230, 0.94));
            border: 1px solid var(--line);
            min-height: 100%;
            box-shadow: 0 18px 32px rgba(18, 38, 58, 0.06);
        }
        .guide-card h4 {
            margin-top: 0;
            color: var(--ink);
            letter-spacing: -0.01em;
            font-family: "Iowan Old Style", "Palatino Linotype", serif;
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
            background: rgba(14, 102, 101, 0.13);
            color: #0e6665;
        }
        .status-pill-neutral {
            background: rgba(35, 77, 132, 0.12);
            color: #234d84;
        }
        .status-pill-warn {
            background: rgba(165, 61, 50, 0.12);
            color: #a53d32;
        }
        section[data-testid="stMain"] div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(251, 248, 241, 0.98), rgba(244, 237, 228, 0.95));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 14px 26px rgba(18, 38, 58, 0.06);
        }
        section[data-testid="stMain"] div[data-testid="stMetric"] label,
        section[data-testid="stMain"] div[data-testid="stMetric"] div {
            color: var(--ink);
        }
        section[data-testid="stMain"] div[data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
        }
        section[data-testid="stMain"] .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        section[data-testid="stMain"] .stTabs [data-baseweb="tab"] {
            background: rgba(248, 243, 234, 0.92);
            border: 1px solid var(--line);
            border-radius: 14px;
            color: var(--ink);
            padding: 0.32rem 0.72rem;
        }
        section[data-testid="stMain"] .stTabs [aria-selected="true"] {
            background: linear-gradient(145deg, rgba(174, 132, 80, 0.16), rgba(8, 20, 35, 0.08));
            color: var(--ink);
            border-color: rgba(174, 132, 80, 0.26);
        }
        section[data-testid="stMain"] div[data-testid="stAlert"],
        section[data-testid="stMain"] div[data-testid="stDataFrame"] {
            color: var(--ink);
        }
        section[data-testid="stMain"] div[data-testid="stAlert"] *,
        section[data-testid="stMain"] div[data-testid="stDataFrame"] *,
        section[data-testid="stMain"] div[data-testid="stCaptionContainer"] *,
        section[data-testid="stMain"] .stCaption,
        section[data-testid="stMain"] code {
            color: var(--ink) !important;
        }
        .panel-hero h1,
        .panel-hero h2,
        .panel-hero h3,
        .panel-hero h4,
        .panel-hero p,
        .panel-hero span,
        .panel-hero div {
            color: #f4ede2 !important;
        }
        section[data-testid="stMain"] div[data-testid="stPlotlyChart"] {
            background: linear-gradient(180deg, rgba(251, 248, 241, 0.94), rgba(244, 237, 228, 0.90));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.35rem 0.35rem 0.1rem 0.35rem;
            box-shadow: 0 14px 24px rgba(18, 38, 58, 0.05);
        }
        div[data-testid="stButton"] > button {
            border-radius: 14px;
            border: 1px solid rgba(210, 177, 128, 0.22);
            background: linear-gradient(180deg, #f4ede2, #e9decd);
            color: #142033;
            font-weight: 800;
        }
        div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background: linear-gradient(180deg, #c4a06f, #a97e4b);
            color: #0b1626;
            border: 1px solid rgba(255, 255, 255, 0.10);
        }
        @media (max-width: 1100px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .hero-band {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _run_analysis(
    query: str,
    analysis_mode: str,
    history_years: int,
    projection_years: int,
    valuation_simulations: int,
    forecast_simulations: int,
    equity_risk_premium: float,
    debt_spread: float,
    tax_rate: float,
    force_refresh: bool,
    use_live_quote: bool,
) -> tuple[object, dict, dict, dict]:
    dataset = fetch_stock_dataset(
        query,
        history_years=history_years,
        force_refresh=force_refresh,
        use_live_quote=use_live_quote,
        analysis_mode=analysis_mode,
    )
    valuation: dict[str, Any]
    forecast: dict[str, Any]
    mode_report: dict[str, Any]
    valuation_error: str | None = None
    forecast_error: str | None = None

    if analysis_mode == "long_term":
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
    else:
        valuation = {
            "available": False,
            "error": "Το short-term mode δεν απαιτεί full intrinsic valuation και αποφεύγει επίτηδες το βαρύτερο SEC-dependent rail.",
        }

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

    if analysis_mode == "short_term":
        mode_report = build_short_term_report(dataset, forecast, valuation if valuation.get("available") else None)
    else:
        mode_report = build_long_term_report(dataset, valuation if valuation.get("available") else None)

    if analysis_mode == "short_term" and not forecast["available"]:
        raise DataRetrievalError(
            f"Δεν ήταν δυνατό να βγει usable short-term forecast. Forecast: {forecast_error or 'N/A'}"
        )
    if analysis_mode == "long_term" and not valuation["available"] and not forecast["available"]:
        raise DataRetrievalError(
            f"Δεν ήταν δυνατό να βγει usable valuation ή forecast. "
            f"Valuation: {valuation_error or 'N/A'} | Forecast: {forecast_error or 'N/A'}"
        )
    return dataset, valuation, forecast, mode_report


@st.cache_data(show_spinner=False, ttl=1800)
def _run_analysis_cached(
    query: str,
    analysis_mode: str,
    history_years: int,
    projection_years: int,
    valuation_simulations: int,
    forecast_simulations: int,
    equity_risk_premium: float,
    debt_spread: float,
    tax_rate: float,
    use_live_quote: bool,
) -> tuple[object, dict, dict, dict]:
    return _run_analysis(
        query=query,
        analysis_mode=analysis_mode,
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
    analysis_mode: str,
    history_years: int,
    projection_years: int,
    valuation_simulations: int,
    forecast_simulations: int,
    equity_risk_premium: float,
    debt_spread: float,
    tax_rate: float,
    force_refresh: bool,
    use_live_quote: bool,
) -> tuple[object, dict, dict, dict]:
    if force_refresh:
        return _run_analysis(
            query=query,
            analysis_mode=analysis_mode,
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
        analysis_mode=analysis_mode,
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
    analysis_mode: str,
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
        f"`Mode`: {'Short Term' if analysis_mode == 'short_term' else 'Long Term'}",
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
        font={"color": "#172437", "family": "Avenir Next, Segoe UI, sans-serif"},
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
        font={"color": "#172437", "family": "Avenir Next, Segoe UI, sans-serif"},
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
        font={"color": "#172437", "family": "Avenir Next, Segoe UI, sans-serif"},
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
        font={"color": "#172437", "family": "Avenir Next, Segoe UI, sans-serif"},
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
        font={"color": "#172437", "family": "Avenir Next, Segoe UI, sans-serif"},
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
        font={"color": "#172437", "family": "Avenir Next, Segoe UI, sans-serif"},
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
    if any(token in label_lower for token in ("year", "years", "horizon")):
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


def render_section_intro(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="guide-card" style="margin-bottom: 0.85rem;">
            <h4>{title}</h4>
            <p>{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _summary_table(
    dataset: object,
    valuation: dict[str, Any],
    forecast: dict[str, Any],
    mode_report: dict[str, Any],
    analysis_mode: str,
) -> pd.DataFrame:
    valuation_available = bool(valuation.get("available"))
    forecast_available = bool(forecast.get("available"))
    rows: list[tuple[str, str, str]] = [
        (
            "Current price",
            format_currency(dataset.current_price, dataset.currency),
            "Η τελευταία usable τιμή πάνω στην οποία πατά όλη η ανάλυση.",
        ),
    ]

    if analysis_mode == "short_term":
        rows.extend(
            [
                (
                    "Trade setup",
                    mode_report.get("setup", "N/A") if mode_report.get("available") else "N/A",
                    "Η συνολική short-term ποιότητα του setup αφού συνδυαστούν forecast edge, stress, jump risk και event risk.",
                ),
                (
                    "Action",
                    mode_report.get("action", "N/A") if mode_report.get("available") else "N/A",
                    "Η πειθαρχημένη έξοδος του short-term layer: actionable, watch ή no-trade.",
                ),
                (
                    "Best horizon",
                    mode_report.get("best_horizon", "N/A") if mode_report.get("available") else "N/A",
                    "Το horizon όπου το μοντέλο βλέπει το καλύτερο risk-adjusted edge αυτή τη στιγμή.",
                ),
                (
                    "Expected edge",
                    format_percent(mode_report.get("expected_edge")) if mode_report.get("available") else "N/A",
                    "Ένας συντηρητικός edge proxy που συνδυάζει mean path, upside probabilities, downside tails και event penalties.",
                ),
                (
                    "Signal confidence",
                    f"{mode_report.get('signal_confidence', 'N/A')}/100" if mode_report.get("available") else "N/A",
                    "Πόσο πολύ αξίζει να εμπιστευτούμε το short-term signal αφού λάβουμε υπόψη calibration και data quality.",
                ),
            ]
        )
    else:
        rows.extend(
            [
                (
                    "Fundamental value",
                    format_currency(valuation["percentiles"]["p50"], dataset.currency) if valuation_available else "N/A",
                    "Η median εκτίμηση του intrinsic value distribution. Δεν είναι υπόσχεση τιμής, αλλά το κεντρικό fair-value anchor.",
                ),
                (
                    "Valuation verdict",
                    valuation["verdict"] if valuation_available else "N/A",
                    "Το συμπέρασμα του valuation engine αφού δει range, probability και uncertainty.",
                ),
                (
                    "Long-term stance",
                    mode_report.get("stance", "N/A") if mode_report.get("available") else "N/A",
                    "Η long-term σύνθεση intrinsic value, profitability, balance-sheet resilience και forward estimate anchor.",
                ),
                (
                    "Margin of safety",
                    format_percent(valuation["margin_of_safety"]) if valuation_available else "N/A",
                    "Πόσο πάνω ή κάτω κάθεται το fair-value anchor σε σχέση με την τρέχουσα τιμή.",
                ),
                (
                    "Probability undervalued",
                    format_percent(valuation["probability_undervalued"]) if valuation_available else "N/A",
                    "Το ποσοστό των simulated fair values που βγαίνουν πάνω από τη σημερινή τιμή.",
                ),
                (
                    "Valuation confidence",
                    f"{valuation['confidence']['label']} ({valuation['confidence']['score']}/100)" if valuation_available else "N/A",
                    "Πόσο πολύ εμπιστευόμαστε το valuation output με βάση data quality, model suitability και dispersion.",
                ),
                (
                    "Long-term score",
                    f"{mode_report.get('overall_score', 'N/A')}/100" if mode_report.get("available") else "N/A",
                    "Η συνολική long-term κρίση του app αφού συνδυάσει valuation και ποιοτικά financial pillars.",
                ),
            ]
        )

    rows.extend(
        [
            (
                "Data quality",
                f"{valuation['data_quality']['label']} ({valuation['data_quality']['score']}/100)" if valuation_available and valuation.get("data_quality") else dataset.raw_info.get("coverage_label", "N/A"),
                "Πόσο δυνατή είναι η βάση δεδομένων της συγκεκριμένης ανάλυσης πριν καν μπούμε στα μαθηματικά μοντέλα.",
            ),
        (
            "Forecast quality",
            f"{forecast['calibration']['label']} ({forecast['calibration']['score']}/100)" if forecast_available else "N/A",
            "Πόσο καλά έχει σταθεί πρόσφατα το probabilistic forecast engine σε direction, volatility και interval coverage.",
        ),
        (
            "Current regime",
            (
                f"{forecast['current_regime']} ({format_percent(forecast['current_regime_probability'])})"
                if forecast_available
                else "N/A"
            ),
            "Το regime που το μοντέλο θεωρεί πιο πιθανό αυτή τη στιγμή για τη μετοχή.",
        ),
        (
            "Coverage",
            dataset.raw_info.get("coverage_label", "N/A"),
            "Μια γρήγορη ένδειξη για το πόσο σταθερή ή εύθραυστη είναι η κάλυψη του ticker στο free stack.",
        ),
    ])
    return pd.DataFrame(rows, columns=["Item", "Value", "What It Means"])


def _input_meaning(label: str) -> str:
    mapping = {
        "Method": "Το valuation rail που επιλέχθηκε για αυτή την εταιρεία.",
        "Starting revenue": "Το πιο πρόσφατο annual revenue που χρησιμοποιείται σαν βάση για το projection.",
        "Starting EBIT margin": "Το operating profitability point από το οποίο ξεκινά το explicit forecast.",
        "Target EBIT margin": "Το margin προς το οποίο υποθέτει το μοντέλο ότι συγκλίνει η εταιρεία.",
        "Revenue growth": "Ο αρχικός ρυθμός ανάπτυξης που τροφοδοτεί την explicit phase του FCFF model.",
        "Sales to capital": "Δείχνει πόσο αποδοτικά η εταιρεία μετατρέπει reinvestment σε νέο revenue.",
        "Terminal growth": "Ο μακροχρόνιος ρυθμός ανάπτυξης μετά την explicit forecast περίοδο.",
        "Risk-free rate": "Η βάση πάνω στην οποία χτίζεται το cost of capital.",
        "Beta used": "Η beta μετά από shrinkage, ώστε να μη γίνεται υπερευαίσθητη σε noisy market history.",
        "Cost of equity": "Η απαιτούμενη απόδοση των μετόχων με βάση risk-free και equity risk premium.",
        "Cost of debt": "Το κόστος δανεισμού που χρησιμοποιείται μέσα στο WACC.",
        "WACC": "Το blended discount rate για FCFF valuation.",
        "Net cash / debt": "Προστίθεται ή αφαιρείται όταν περνάμε από enterprise value σε equity value.",
        "User-selected DCF years": "Τα explicit forecast years που διάλεξες από το sidebar.",
        "Effective DCF years": "Τα years που τελικά χρησιμοποίησε το model μετά την προσαρμογή για growth profile.",
        "Market-implied stage-1 growth": "Το growth που φαίνεται να ζητά η αγορά για να δικαιολογεί την τωρινή τιμή.",
        "90% intrinsic range": "Το εύρος μεταξύ p05 και p95. Όσο πιο φαρδύ, τόσο μεγαλύτερη η αβεβαιότητα.",
        "Book value / share": "Η λογιστική καθαρή θέση ανά μετοχή.",
        "Starting ROE": "Η απόδοση ιδίων κεφαλαίων από την οποία ξεκινά το residual-income model.",
        "Terminal ROE": "Το ROE προς το οποίο συγκλίνει το residual-income projection.",
        "Retention rate": "Το ποσοστό κερδών που θεωρείται ότι μένει μέσα στην επιχείρηση.",
    }
    return mapping.get(label, "Input του valuation engine που επηρεάζει την τελική εκτίμηση fair value.")


def _ratio_meaning(label: str) -> str:
    mapping = {
        "P/E": "Πόσες φορές τα κέρδη πληρώνει σήμερα η αγορά.",
        "PEG": "Συνδέει το P/E με το growth για να δείξει αν το premium στηρίζεται από ανάπτυξη.",
        "P/B": "Πόσες φορές τη λογιστική καθαρή θέση αποτιμά η αγορά.",
        "EV/EBITDA": "Σχετικός πολλαπλασιαστής για λειτουργική αξία πριν από D&A και capital structure.",
        "ROE": "Πόσο αποδοτικά μετατρέπει η εταιρεία τα ίδια κεφάλαια σε κέρδη.",
        "Profit margin": "Τι ποσοστό του revenue μένει τελικά ως καθαρό κέρδος.",
    }
    return mapping.get(label, "Σχετικός δείκτης για το πώς τιμολογείται ή αποδίδει η εταιρεία.")


def _forecast_metric_meaning(label: str) -> str:
    mapping = {
        "Model": "Ο συνδυασμός μοντέλων που χρησιμοποιήθηκε για το short-term forecast.",
        "Current regime": "Η πιθανότερη φάση αγοράς/συμπεριφοράς για τη μετοχή αυτή τη στιγμή.",
        "Current regime probability": "Πόσο σίγουρο είναι το regime model για το τρέχον state.",
        "Forecast quality": "Συνοπτική αξιολόγηση του calibration του forecast engine.",
        "Daily jump probability": "Η πιθανότητα να εμφανιστεί ακραίο ημερήσιο shock στο simulation.",
        "Context drift bias": "Η καθαρή μετατόπιση στο drift από market-context features και proxies.",
        "Opportunity score": "Composite score που μετρά πόσο risk-on / favorable είναι το setup.",
        "Stress score": "Composite score που μετρά market stress, volatility pressure και αμυντική συμπεριφορά.",
        "Geopolitical proxy score": "Score που διαβάζει risk-off σήματα από oil, gold, dollar, rates και GPR proxies.",
        "Behavior score": "Score που διαβάζει momentum/reversal/breadth συμπεριφορά.",
        "AR(1) intercept": "Η σταθερή συνιστώσα του mean model στα daily returns.",
        "AR(1) phi": "Το πόσο persistent είναι το χθεσινό return στο σημερινό forecast drift.",
    }
    return mapping.get(label, "Forecast diagnostic που βοηθά να διαβαστεί το probabilistic αποτέλεσμα.")


def render_horizon_cards(horizon_summary: list[dict], currency: str) -> None:
    frame = pd.DataFrame(
        [
            {
                "Horizon": horizon["label"],
                "Median": format_currency(horizon["median"], currency),
                "90% Range": f"{format_currency(horizon['p05'], currency)} - {format_currency(horizon['p95'], currency)}",
                "P(Upside)": format_percent(horizon["probability_upside"]),
                "P(>+5%)": format_percent(horizon["probability_up_5pct"]),
            }
            for horizon in horizon_summary
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_short_term_report(mode_report: dict[str, Any], currency: str) -> None:
    if not mode_report.get("available"):
        st.warning(mode_report.get("error", "Το short-term report δεν ήταν διαθέσιμο."))
        return

    st.markdown("### Trading Stance")
    st.dataframe(
        pd.DataFrame(
            [
                ("Setup", mode_report["setup"], "Η συνολική ποιότητα του short-term setup."),
                ("Action", mode_report["action"], "Η πρακτική έξοδος του trade-decision layer."),
                ("Trade setup score", f"{mode_report['trade_setup_score']}/100", "Ο συνολικός short-term score αφού συνδυαστούν edge, calibration, stress και event risk."),
                ("Signal confidence", f"{mode_report['signal_confidence']}/100", "Το πόσο εμπιστευόμαστε ότι το short-term signal έχει χρηστική αξία."),
                ("Best horizon", mode_report["best_horizon"], "Το horizon όπου το μοντέλο βλέπει το καθαρότερο risk-adjusted edge."),
                ("Expected edge", format_percent(mode_report["expected_edge"]), "Συντηρητικό edge proxy μετά από downside και event penalties."),
            ],
            columns=["Trading item", "Value", "What It Means"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Horizon Edge Map")
    horizon_frame = pd.DataFrame(mode_report["horizon_table"]).copy()
    horizon_frame["mean_return"] = horizon_frame["mean_return"].map(format_percent)
    horizon_frame["median_return"] = horizon_frame["median_return"].map(format_percent)
    horizon_frame["probability_upside"] = horizon_frame["probability_upside"].map(format_percent)
    horizon_frame["probability_up_5pct"] = horizon_frame["probability_up_5pct"].map(format_percent)
    horizon_frame["probability_down_5pct"] = horizon_frame["probability_down_5pct"].map(format_percent)
    horizon_frame["expected_max_drawdown"] = horizon_frame["expected_max_drawdown"].map(format_percent)
    horizon_frame["expected_edge"] = horizon_frame["expected_edge"].map(format_percent)
    horizon_frame["edge_score"] = horizon_frame["edge_score"].map(lambda value: f"{value:.0f}/100")
    horizon_frame["event_penalty"] = horizon_frame["event_penalty"].map(format_percent)
    st.dataframe(
        horizon_frame.rename(
            columns={
                "horizon": "Horizon",
                "days": "Days",
                "mean_return": "Mean Return",
                "median_return": "Median Return",
                "probability_upside": "P(Upside)",
                "probability_up_5pct": "P(>+5%)",
                "probability_down_5pct": "P(<-5%)",
                "expected_max_drawdown": "Expected Max Drawdown",
                "expected_edge": "Expected Edge",
                "edge_score": "Edge Score",
                "event_penalty": "Event Penalty",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    event = mode_report.get("event_risk", {})
    st.markdown("### Event Risk")
    st.dataframe(
        pd.DataFrame(
            [
                ("Event date", event.get("event_date", "N/A"), "Η πιο κοντινή earnings/event ημερομηνία που βρήκε το free data stack."),
                ("Days to event", str(event.get("days_to_event", "N/A")), "Πόσο κοντά ή μακριά είναι το event από το σήμερα."),
                ("Release timing", event.get("time", "N/A"), "Αν ο provider έδωσε timing τύπου BMO/AMC ή παρόμοιο marker."),
                ("Event window active", "Yes" if event.get("event_window_active") else "No", "Αν είμαστε μέσα στο παράθυρο όπου το μοντέλο κόβει score λόγω event risk."),
            ],
            columns=["Event item", "Value", "What It Means"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    exit_window = mode_report.get("exit_window", {})
    if exit_window.get("best_window"):
        st.markdown("### Exit Timing Window")
        st.dataframe(
            pd.DataFrame(
                [
                    ("Target price", format_currency(exit_window.get("target_price"), currency) if exit_window.get("target_price") is not None else "N/A", "Το tactical target που χρησιμοποιεί το exit engine για να μετρήσει first-passage probabilities."),
                    ("Stop reference", format_currency(exit_window.get("stop_price"), currency) if exit_window.get("stop_price") is not None else "N/A", "Το downside reference του ίδιου timing engine."),
                    ("Best exit day", f"Day {exit_window.get('best_day')}" if exit_window.get("best_day") is not None else "N/A", "Η ημέρα με το καλύτερο συνδυαστικό exit score."),
                    ("Best exit window", exit_window.get("best_window", "N/A"), "Ένα πιο ανθρώπινο window εξόδου γύρω από την κορυφή του exit score."),
                    ("Median days to target", f"{exit_window['median_days_to_target']:.1f}" if exit_window.get("median_days_to_target") is not None else "N/A", "Σε πόσες περίπου συνεδριάσεις φτάνουν το target τα paths που το πετυχαίνουν."),
                ],
                columns=["Timing item", "Value", "What It Means"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        exit_rows = pd.DataFrame(exit_window.get("rows", [])).copy()
        if not exit_rows.empty:
            exit_rows = exit_rows.head(10)
            for column in (
                "target_hit_probability",
                "stop_hit_probability",
                "target_first_probability",
                "stop_first_probability",
                "expected_return",
                "expected_max_drawdown",
                "regime_deterioration_probability",
                "event_hazard",
            ):
                exit_rows[column] = exit_rows[column].map(format_percent)
            exit_rows["exit_score"] = exit_rows["exit_score"].map(lambda value: f"{value:.3f}")
            st.dataframe(
                exit_rows.rename(
                    columns={
                        "day": "Day",
                        "target_hit_probability": "P(Target By Day)",
                        "stop_hit_probability": "P(Stop By Day)",
                        "target_first_probability": "P(Target Before Stop)",
                        "stop_first_probability": "P(Stop Before Target)",
                        "expected_return": "Expected Return",
                        "expected_max_drawdown": "Expected Max Drawdown",
                        "regime_deterioration_probability": "P(Regime Turns Bearish)",
                        "event_hazard": "Event Hazard",
                        "exit_score": "Exit Score",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    render_takeaways("Short-Term Read", mode_report.get("reasons", []))
    render_takeaways("Risk Controls", mode_report.get("risk_controls", []))


def render_long_term_report(mode_report: dict[str, Any], currency: str) -> None:
    if not mode_report.get("available"):
        st.warning(mode_report.get("error", "Το long-term report δεν ήταν διαθέσιμο."))
        return

    st.markdown("### Investment Stance")
    st.dataframe(
        pd.DataFrame(
            [
                ("Long-term stance", mode_report["stance"], "Η συνολική long-duration κρίση του app."),
                ("Holding action", mode_report.get("holding_action", "N/A"), "Η πιο πρακτική long-term ανάγνωση του app για το αν αξίζει accumulation, απλό hold ή περισσότερη αποχή."),
                ("Overall score", f"{mode_report['overall_score']}/100", "Η σύνθεση intrinsic value, profitability, balance-sheet quality και forward estimate anchor."),
                ("Forward revenue growth", format_percent(mode_report.get("forward_revenue_growth")) if mode_report.get("forward_revenue_growth") is not None else "N/A", "Το κοντινό growth anchor από τα δωρεάν estimate layers όταν είναι διαθέσιμα."),
                ("Forward EPS growth", format_percent(mode_report.get("forward_eps_growth")) if mode_report.get("forward_eps_growth") is not None else "N/A", "Το earnings-growth anchor από το ίδιο estimate layer."),
                ("Owner earnings", format_currency(mode_report.get("owner_earnings_value"), currency) if mode_report.get("owner_earnings_value") is not None else "N/A", "Συμπληρωματικός cash-flow lens για πιο ώριμες επιχειρήσεις."),
                ("Mean price target", format_currency(mode_report.get("price_target_mean"), currency) if mode_report.get("price_target_mean") is not None else "N/A", "Consensus-like internet-backed reference point, μόνο ως secondary lens και όχι σαν intrinsic truth."),
            ],
            columns=["Investment item", "Value", "What It Means"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    pillar_frame = pd.DataFrame(mode_report.get("pillar_rows", [])).copy()
    if not pillar_frame.empty:
        pillar_frame["score"] = pillar_frame["score"].map(lambda value: f"{value:.0f}/100")
        st.markdown("### Long-Term Pillars")
        st.dataframe(
            pillar_frame.rename(
                columns={
                    "pillar": "Pillar",
                    "score": "Score",
                    "commentary": "What It Means",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    render_takeaways("Long-Term Read", mode_report.get("reasons", []))


def render_sidebar_budget(budget: dict[str, Any], compute_profile: dict[str, str]) -> None:
    st.sidebar.markdown(
        f"""
        <div class="budget-card">
            <div class="budget-label">Fresh Calls Budget</div>
            <div class="budget-number">{budget["label"]}</div>
            <div class="micro-copy" style="color: rgba(247,243,237,0.86);">
                Estimated new requests for this run
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-note">
            <strong>Compute posture:</strong> {compute_profile["label"]}<br/>
            {compute_profile["detail"]}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(budget["note"])
    for item in budget["breakdown"]:
        state = "new call" if item["will_call"] else "cache"
        st.sidebar.caption(f"`{item['label']}`: {state}. {item['detail']}")


def render_intro(budget: dict[str, Any], compute_profile: dict[str, str], analysis_mode: str) -> None:
    mandate_copy = (
        "Ένα προσωπικό research terminal για disciplined short-horizon equity work, με "
        "<strong>trade-decision filtering</strong>, <strong>probabilistic short-term forecasting</strong> και "
        "<strong>event-aware market context</strong> σε ένα ενιαίο briefing."
        if analysis_mode == "short_term"
        else "Ένα προσωπικό research terminal για disciplined long-duration equity work, με "
        "<strong>adaptive intrinsic valuation</strong>, <strong>market-implied expectations</strong> και "
        "<strong>internet-backed fundamental cross-checks</strong> σε ένα ενιαίο briefing."
    )
    second_copy = (
        "Στο short-term mode το app δίνει βάρος σε <strong>edge</strong>, <strong>no-trade discipline</strong>, "
        "<strong>event risk</strong> και <strong>forecast calibration</strong>, ώστε να αποφεύγονται trades χωρίς καθαρό πλεονέκτημα."
        if analysis_mode == "short_term"
        else "Στο long-term mode το app προσπαθεί πρώτα να αναγνωρίσει <strong>το valuation regime της εταιρείας</strong>, "
        "μετά να επιλέξει το καταλληλότερο framework, και στο τέλος να επιστρέψει "
        "<strong>fair-value range</strong>, <strong>profitability / balance-sheet lenses</strong> και "
        "<strong>confidence diagnostics</strong>."
    )
    st.markdown(
        f"""
        <div class="hero-grid">
            <div class="panel panel-hero">
                <div class="hero-kicker">Private Research Environment</div>
                <div class="hero-title">Equity Research Desk</div>
                <div class="hero-copy">
                    {mandate_copy}
                </div>
                <div class="hero-copy" style="margin-top: 0.7rem;">
                    {second_copy}
                </div>
                <div class="hero-band">
                    <div class="hero-band-item">
                        <span class="hero-band-label">Mandate</span>
                        <span class="hero-band-value">{"Short-term execution discipline" if analysis_mode == "short_term" else "Long-term intrinsic research"}</span>
                    </div>
                    <div class="hero-band-item">
                        <span class="hero-band-label">Core Engine</span>
                        <span class="hero-band-value">{"Trade setup scoring, event risk, regime-aware forecasting" if analysis_mode == "short_term" else "Adaptive valuation, Monte Carlo, regime-aware forecasting"}</span>
                    </div>
                    <div class="hero-band-item">
                        <span class="hero-band-label">Current Run</span>
                        <span class="hero-band-value">{budget["label"]} estimated calls | {compute_profile["label"]} compute load</span>
                    </div>
                </div>
            </div>
            <div class="panel panel-ops">
                <div class="section-overline">Pre-Trade Protocol</div>
                <div class="panel-title">Execution Discipline</div>
                <div class="workflow-step"><strong>1.</strong> Δώσε κατά προτίμηση ticker, π.χ. <code>AAPL</code>, ώστε να έχουμε πιο καθαρό symbol mapping και call budgeting.</div>
                <div class="workflow-step"><strong>2.</strong> Δες πρώτα το <strong>estimated fresh calls</strong>. Για προσωπική χρήση, το σωστό είναι λίγα νέα requests και πολύ cache reuse.</div>
                <div class="workflow-step"><strong>3.</strong> Χρησιμοποίησε <strong>live quote</strong> μόνο όταν θες spot refresh και άφηνε το <strong>refresh</strong> κλειστό όταν δεν υπάρχει λόγος να κάψεις νέο fetch.</div>
                <div class="workflow-step"><strong>4.</strong> Μετά το run, διάβασε πρώτα το <strong>{"Trading Brief" if analysis_mode == "short_term" else "Investment Brief"}</strong>, μετά το <strong>{"Signal Desk" if analysis_mode == "short_term" else "Valuation Desk"}</strong> και τέλος το <strong>Forecast Desk</strong>.</div>
                <div class="workflow-step"><strong>Current posture:</strong> {budget["label"]} estimated calls | <strong>Compute:</strong> {compute_profile["label"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_guide_tab(
    settings_impact: list[str],
    budget: dict[str, Any],
    compute_profile: dict[str, str],
    analysis_mode: str,
) -> None:
    guide_left, guide_right = st.columns(2)
    with guide_left:
        st.markdown(
            f"""
            <div class="guide-card">
                <h4>Research Stack</h4>
                <p><strong>Selected mode:</strong> {"Short Term" if analysis_mode == "short_term" else "Long Term"}.</p>
                <p><strong>Adaptive valuation:</strong> το app δεν περνά όλα τα tickers από ίδιο μοντέλο. Προσπαθεί πρώτα να καταλάβει τι εταιρεία έχει μπροστά του.</p>
                <p><strong>FCFF rail:</strong> βασίζεται σε revenue, operating margin, reinvestment discipline και WACC, όχι σε τυφλό extrapolation του historical FCF.</p>
                <p><strong>Residual income rail:</strong> ενεργοποιείται όταν το equity/book framework είναι πιο κατάλληλο από ένα generic operating-company DCF.</p>
                <p><strong>Reverse DCF:</strong> λέει τι growth ζητά ήδη η αγορά για να στέκει η τωρινή τιμή.</p>
                <p><strong>Monte Carlo:</strong> μετατρέπει το fair value από single number σε πιθανό range.</p>
                <p><strong>Markov + GARCH:</strong> χαρτογραφεί regime state, volatility clustering, asymmetry και shock risk στα short-term paths.</p>
                <p><strong>Cross-asset context:</strong> χρησιμοποιεί SPY, QQQ, IWM, GLD, TLT, USO, UUP και sector ETF σαν market-behavior and stress proxies.</p>
                <p><strong>Confidence layer:</strong> το αποτέλεσμα δεν είναι μόνο αριθμός. Συνοδεύεται και από κρίση για το πόσο robust είναι.</p>
                <p><strong>Mode logic:</strong> στο short-term δίνει βάρος σε edge, no-trade filtering και event risk. Στο long-term δίνει βάρος σε intrinsic value, profitability και capital structure.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with guide_right:
        st.markdown(
            f"""
            <div class="guide-card">
                <h4>Current Operating Setup</h4>
                <p><strong>Estimated fresh calls:</strong> {budget["label"]}</p>
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
        st.markdown("## Research Controls")
        analysis_mode_label = st.radio(
            "Research horizon",
            options=["Short Term", "Long Term"],
            horizontal=True,
            help="Το short-term δίνει βάρος σε trade setup, no-trade gating και event risk. Το long-term δίνει βάρος σε intrinsic valuation και fundamental quality.",
        )
        analysis_mode = "short_term" if analysis_mode_label == "Short Term" else "long_term"
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
            max_value=12,
            value=6,
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
        "analysis_mode": analysis_mode,
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
        analysis_mode=analysis_mode,
        force_refresh=force_refresh,
    )
    compute_profile = _compute_profile(valuation_simulations, forecast_simulations, history_years)
    settings_impact = _settings_impact_lines(
        analysis_mode=analysis_mode,
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
                <strong>Desk note:</strong><br/>
                Το setup είναι ρυθμισμένο ώστε να παραμένει γρήγορο, explainable και βιώσιμο στο free-data stack,
                με έμφαση στο disciplined προσωπικό research και όχι στο αχρείαστα βαρύ compute.
            </div>
            """,
            unsafe_allow_html=True,
        )
        run_clicked = st.button(
            "Run Trading Brief" if analysis_mode == "short_term" else "Run Investment Brief",
            use_container_width=True,
            type="primary",
        )

    render_intro(budget, compute_profile, analysis_mode)

    if run_clicked:
        if not settings["query"]:
            st.error("Πρώτα γράψε ticker ή όνομα εταιρείας.")
        else:
            try:
                with st.spinner("Loading market data, selecting valuation framework, and fitting forecast models..."):
                    dataset, valuation, forecast, mode_report = run_analysis(**settings)
                st.session_state["analysis_payload"] = {
                    "dataset": dataset,
                    "valuation": valuation,
                    "forecast": forecast,
                    "mode_report": mode_report,
                    "analysis_mode": analysis_mode,
                }
                st.session_state["analysis_signature"] = current_signature
                st.session_state["query_value"] = settings["query"]
            except (DataRetrievalError, ValuationError, ForecastModelError) as exc:
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover
                st.error(f"Unexpected error: {exc}")

    analysis_payload = st.session_state.get("analysis_payload")
    if not analysis_payload:
        st.info(
            "Ρύθμισε τα controls αριστερά, δες πρώτα πόσα calls θα γίνουν, και μετά πάτα "
            f"{'Run Trading Brief' if analysis_mode == 'short_term' else 'Run Investment Brief'}."
        )
        render_guide_tab(settings_impact, budget, compute_profile, analysis_mode)
        return

    dataset = analysis_payload["dataset"]
    valuation = analysis_payload["valuation"]
    forecast = analysis_payload["forecast"]
    mode_report = analysis_payload.get("mode_report", {})
    analysis_mode = analysis_payload.get("analysis_mode", analysis_mode)
    stored_signature = st.session_state.get("analysis_signature")
    valuation_available = bool(valuation.get("available"))
    forecast_available = bool(forecast.get("available"))

    if stored_signature != current_signature:
        st.warning(
            "Τα αποτελέσματα που βλέπεις είναι από τα τελευταία submitted settings. "
            f"Αν άλλαξες sliders, πάτα ξανά {'Run Trading Brief' if analysis_mode == 'short_term' else 'Run Investment Brief'}."
        )

    render_section_intro(
        "Executive Summary",
        "Αυτό είναι το πιο σύντομο και αναγνώσιμο snapshot του run. Κάθε γραμμή λέει όχι μόνο τι αριθμό πήραμε, αλλά και τι σημαίνει πρακτικά για την ανάλυση.",
    )
    st.dataframe(
        _summary_table(dataset, valuation, forecast, mode_report, analysis_mode),
        use_container_width=True,
        hide_index=True,
    )

    if analysis_mode == "short_term":
        verdict_label = mode_report.get("action", "Short-Term")
        verdict_class = (
            "status-pill-good"
            if verdict_label == "Actionable"
            else "status-pill-warn"
            if verdict_label == "No-Trade"
            else "status-pill-neutral"
        )
    else:
        verdict_label = valuation["verdict"] if valuation_available else "Long-Term Review"
        verdict_class = "status-pill-good" if verdict_label == "Undervalued" else "status-pill-warn" if verdict_label == "Overvalued" else "status-pill-neutral"
    st.markdown(
        f"""
        <span class="{verdict_class}">{verdict_label}</span>
        <span class="summary-chip">{dataset.symbol}</span>
        <span class="summary-chip">{dataset.sector}</span>
        <span class="summary-chip">{'Short Term' if analysis_mode == 'short_term' else 'Long Term'}</span>
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
    if analysis_mode == "short_term" and mode_report.get("available"):
        st.write(
            f"Το short-term decision layer βγάζει `{mode_report.get('setup', 'N/A')}` setup με action "
            f"`{mode_report.get('action', 'N/A')}` και καλύτερο horizon το `{mode_report.get('best_horizon', 'N/A')}`."
        )
    elif valuation_available:
        st.write(valuation["verdict_reason"])
        st.caption(valuation.get("valuation_stack_note", ""))
        if valuation.get("market_implied"):
            market_implied = valuation["market_implied"]
            upper_text = " και ξεπερνά το ανώτατο debug range του μοντέλου" if market_implied.get("upper_bound_hit") else ""
            st.info(
                f"Reverse DCF: για να δικαιολογηθεί η τρέχουσα τιμή, η αγορά χρειάζεται περίπου "
                f"{format_percent(market_implied['required_initial_growth'])} stage-1 FCF growth "
                f"σε horizon {market_implied['projection_years']} ετών{upper_text}. "
                f"{market_implied['note']}"
            )
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
    risk_free_date = dataset.raw_info.get("risk_free_rate_date")
    risk_free_suffix = f" as of {risk_free_date}" if risk_free_date else ""
    risk_free_text = (
        f"{format_percent(dataset.risk_free_rate)} ({dataset.raw_info.get('risk_free_rate_source', 'N/A')}"
        f"{risk_free_suffix})"
    )
    st.caption(
        f"Price basis: {dataset.raw_info.get('price_basis', 'N/A')} | "
        f"Coverage: {dataset.raw_info.get('coverage_note', 'N/A')} | "
        f"Risk-free: {risk_free_text} | "
        f"Actual fresh calls: {dataset.raw_info.get('actual_network_calls', 'N/A')} | "
        f"Cache hints: {', '.join(dataset.raw_info.get('cache_messages', [])) or 'Fresh API responses'}"
    )

    tabs = st.tabs(
        ["Trading Brief", "Signal Desk", "Forecast Desk", "Reliability", "Method"]
        if analysis_mode == "short_term"
        else ["Investment Brief", "Valuation Desk", "Forecast Desk", "Reliability", "Method"]
    )

    with tabs[0]:
        st.markdown("### Trading Brief" if analysis_mode == "short_term" else "### Investment Brief")
        render_section_intro(
            "What This Section Shows",
            "Το short-term briefing συνδυάζει trade setup, probability map, event risk και no-trade discipline."
            if analysis_mode == "short_term"
            else "Το investment briefing συνδυάζει intrinsic value, long-term stance και το probabilistic market backdrop ώστε να δεις αν η μετοχή φαίνεται ελκυστική, απαιτητική ή απλώς αβέβαιη.",
        )
        if analysis_mode == "short_term":
            render_short_term_report(mode_report, dataset.currency)
        else:
            render_long_term_report(mode_report, dataset.currency)

        if forecast_available:
            st.plotly_chart(make_price_forecast_chart(dataset.price_history, forecast), use_container_width=True)
            st.caption("Το fan chart δείχνει πιθανές διαδρομές τιμής. Η median γραμμή είναι το κέντρο της κατανομής και οι ζώνες γύρω της δείχνουν uncertainty και tail risk.")
        else:
            st.warning(forecast.get("error", "Forecast unavailable."))
        if valuation_available:
            st.plotly_chart(
                make_valuation_distribution_chart(valuation, dataset.current_price),
                use_container_width=True,
            )
            st.caption("Η valuation distribution δείχνει πολλά πιθανά fair values. Η μπλε γραμμή είναι το median intrinsic estimate και η κόκκινη η αγορά.")
        elif analysis_mode == "long_term":
            st.info("Το valuation panel θα εμφανιστεί μόνο όταν υπάρχουν αρκετά SEC fundamentals για αυτό το ticker.")

        st.markdown("### Short-Horizon Outlook" if analysis_mode == "short_term" else "### Market Backdrop")
        if forecast_available:
            render_horizon_cards(forecast["horizon_summary"], dataset.currency)
        else:
            st.warning(forecast.get("error", "Forecast unavailable."))

        if forecast_available:
            render_takeaways("Desk Takeaways", forecast.get("takeaways", []))
        else:
            st.info("Μόλις βγει forecast, εδώ θα εμφανιστούν τα βασικά takeaways του μοντέλου.")
        if analysis_mode == "short_term":
            overview_lines = [
                "Το short-term forecast λειτουργεί σαν probability map και όχι σαν υπόσχεση τιμής.",
                "Το πιο χρήσιμο output δεν είναι πάντα buy signal. Πολύ συχνά είναι το `No-Trade` όταν το edge είναι μικρό ή το event risk πολύ κοντινό.",
                "Όταν το καλύτερο horizon έχει καλό edge αλλά το calibration είναι αδύναμο, το σωστό διάβασμα είναι watchlist και όχι επιθετική είσοδος.",
            ]
        else:
            overview_lines = [
                "Το valuation λειτουργεί ως anchor για τη θεωρητική αξία και όχι σαν μία μοναδική βεβαιότητα.",
                "Το short-term forecast λειτουργεί ως probability map. Το κέντρο του fan chart είναι πιθανότερο path, όχι υπόσχεση.",
                "Όταν valuation και forecast συμφωνούν, το signal είναι πιο καθαρό. Όταν διαφωνούν, θέλει μικρότερο position size και μεγαλύτερη πειθαρχία.",
            ]
            if valuation_available and valuation.get("cross_check"):
                overview_lines.append(
                    f"Το secondary valuation model ({valuation['cross_check']['method_used']}) κάθεται περίπου {format_percent(valuation['cross_check']['gap_vs_primary'])} από το primary anchor."
                )
            if valuation_available and valuation.get("market_implied"):
                market_implied = valuation["market_implied"]
                overview_lines.append(
                    f"Το reverse DCF δείχνει ότι η αγορά προεξοφλεί περίπου {format_percent(market_implied['required_initial_growth'])} αρχικό FCF growth σε horizon {market_implied['projection_years']} ετών."
                )
        render_takeaways("Research Posture", overview_lines)

    with tabs[1]:
        st.markdown("### Signal Desk" if analysis_mode == "short_term" else "### Intrinsic Valuation")
        if analysis_mode == "short_term":
            render_section_intro(
                "How To Read This Section",
                "Το Signal Desk είναι το σημείο όπου το app μετατρέπει το forecast σε πρακτική απόφαση: edge, action, event risk και no-trade discipline.",
            )
            render_short_term_report(mode_report, dataset.currency)
            if valuation_available:
                st.markdown("### Secondary Long-Term Context")
                st.info("Παρότι το mode είναι short-term, αν υπήρχε usable valuation anchor το δείχνουμε μόνο ως background context και όχι ως κύριο trade trigger.")
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("Valuation verdict", valuation["verdict"], "Το intrinsic stance, μόνο σαν background context."),
                            ("Margin of safety", format_percent(valuation["margin_of_safety"]), "Πόσο premium ή discount βγαίνει η αγορά απέναντι στο fair-value anchor."),
                            ("Valuation confidence", f"{valuation['confidence']['label']} ({valuation['confidence']['score']}/100)", "Πόσο πολύ αξίζει να βαρύνει το valuation μέσα σε ένα short-term setup."),
                        ],
                        columns=["Context item", "Value", "What It Means"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Στο short-term mode δεν απαιτείται full SEC-backed intrinsic valuation για να βγει trading brief.")
        elif not valuation_available:
            st.warning(valuation.get("error", "Δεν βγήκε fair value για αυτό το ticker με το free-mode fundamentals stack."))
            st.info("Το free valuation δουλεύει καλύτερα σε U.S. companies με καθαρά SEC filings και αρκετό historical companyfacts coverage.")
        else:
            render_section_intro(
                "How To Read This Section",
                "Εδώ φαίνεται ποιο valuation framework διάλεξε το app, ποιες assumptions το κινούν, πού βγαίνει η αξία σε διαφορετικά σενάρια, και πόσο απαιτητικές είναι οι προσδοκίες που έχει ήδη ενσωματώσει η αγορά.",
            )
            render_long_term_report(mode_report, dataset.currency)
            st.markdown(
                f"""
                <span class="summary-chip">{valuation['method_used']}</span>
                <span class="summary-chip">Confidence {valuation['confidence']['label']} ({valuation['confidence']['score']}/100)</span>
                <span class="summary-chip">Framework {valuation.get('framework_selected', 'N/A')}</span>
                {f'<span class="summary-chip">Cross-check: {valuation["cross_check"]["method_used"]}</span>' if valuation.get("cross_check") else ''}
                """,
                unsafe_allow_html=True,
            )
            st.info(valuation.get("framework_reason", ""))
            st.plotly_chart(
                make_scenario_chart(valuation, dataset.currency, dataset.current_price),
                use_container_width=True,
            )
            st.caption("Το scenario chart δείχνει πού βγαίνει η αξία σε bear, base και bull assumptions. Αν η αγορά είναι ήδη πάνω από το bull case, το premium είναι πολύ απαιτητικό.")

            st.plotly_chart(
                make_sensitivity_heatmap(valuation, dataset.currency),
                use_container_width=True,
            )
            st.caption("Το sensitivity heatmap δείχνει πόσο γρήγορα μετακινείται το fair value όταν αλλάξουν growth και WACC, δηλαδή οι δύο πιο κρίσιμες παράμετροι.")

            st.plotly_chart(
                make_fundamental_trend_chart(valuation, dataset.currency),
                use_container_width=True,
            )
            st.caption("Το historical anchor chart βοηθά να δεις αν το valuation πατά σε σταθερή θεμελιώδη ιστορία ή σε απότομη πρόσφατη μεταβολή.")

            st.markdown("### Valuation Inputs")
            st.dataframe(
                pd.DataFrame(
                    [
                        (label, format_mixed_value(label, value, dataset.currency), _input_meaning(label))
                        for label, value in valuation.get("input_rows", [])
                    ],
                    columns=["Input", "Value", "What It Means"],
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Market Ratios")
            ratios = valuation["ratios"]
            st.dataframe(
                pd.DataFrame(
                    [
                        ("P/E", format_multiple(ratios["trailing_pe"]), _ratio_meaning("P/E")),
                        ("PEG", format_multiple(ratios["peg_ratio"]), _ratio_meaning("PEG")),
                        ("P/B", format_multiple(ratios["price_to_book"]), _ratio_meaning("P/B")),
                        ("EV/EBITDA", format_multiple(ratios["enterprise_to_ebitda"]), _ratio_meaning("EV/EBITDA")),
                        ("ROE", format_percent(ratios["return_on_equity"]), _ratio_meaning("ROE")),
                        ("Profit margin", format_percent(ratios["profit_margin"]), _ratio_meaning("Profit margin")),
                    ],
                    columns=["Ratio", "Value", "What It Means"],
                ),
                use_container_width=True,
                hide_index=True,
            )

            if valuation.get("market_implied"):
                market_implied = valuation["market_implied"]
                st.markdown("### Market-Implied Expectations")
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("Label", market_implied["label"], "Συνοπτική κρίση για το πόσο απαιτητικές είναι οι παραδοχές που ήδη πληρώνει η αγορά."),
                            ("Required stage-1 growth", format_percent(market_implied["required_initial_growth"]), "Ο growth ρυθμός που χρειάζεται για να δικαιολογείται η σημερινή τιμή."),
                            ("Horizon", f"{market_implied['projection_years']} years", "Για πόσα χρόνια περίπου πρέπει να στηριχθεί το βασικό growth phase."),
                            ("Upper bound hit", "Yes" if market_implied["upper_bound_hit"] else "No", "Αν είναι Yes, η αγορά είναι ακόμη πιο επιθετική από το πάνω όριο του debug range."),
                        ],
                        columns=["Metric", "Value", "What It Means"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(market_implied["note"])

            if valuation.get("cross_check"):
                cross_check = valuation["cross_check"]
                st.markdown("### Cross-Check")
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("Method", cross_check["method_used"], "Το δεύτερο valuation lens που χρησιμοποιήθηκε σαν έλεγχος λογικότητας."),
                            ("Median intrinsic", format_currency(cross_check["median_intrinsic"], dataset.currency), "Η median fair-value εκτίμηση του secondary model."),
                            ("Margin vs market", format_percent(cross_check["margin_vs_market"]), "Πόσο πάνω ή κάτω βγάζει τη μετοχή το secondary lens σε σχέση με την αγορά."),
                            ("Gap vs primary", format_percent(cross_check["gap_vs_primary"]), "Πόσο κοντά ή μακριά είναι το secondary model από το κύριο valuation anchor."),
                        ],
                        columns=["Metric", "Value", "What It Means"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            if valuation.get("classification"):
                classification = valuation["classification"]
                class_lines = list(classification.get("reasons", []))
                class_lines.append(
                    "Data-rich case." if not classification.get("data_poor") else "Data-poor case, άρα το valuation range θέλει μεγαλύτερη προσοχή."
                )
                render_takeaways("Why This Framework Was Selected", class_lines)

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
        st.markdown("### Probability Forecasting")
        if not forecast_available:
            st.warning(forecast.get("error", "Forecast unavailable."))
        else:
            render_section_intro(
                "How To Read This Section",
                "Το forecast engine δεν προσπαθεί να μαντέψει μία τιμή. Προσπαθεί να χαρτογραφήσει πιθανότητες, regimes, volatility, jump risk και market-context bias για το κοντινό μέλλον.",
            )
            st.plotly_chart(make_regime_chart(forecast), use_container_width=True)
            st.caption("Το regime chart δείχνει πώς κατανέμεται η πιθανότητα ανάμεσα στα βασικά market states μέσα στον χρόνο.")

            st.dataframe(
                pd.DataFrame(
                    [
                        ("Model", forecast["model_name"], _forecast_metric_meaning("Model")),
                        ("Current regime", forecast["current_regime"], _forecast_metric_meaning("Current regime")),
                        ("Current regime probability", format_percent(forecast["current_regime_probability"]), _forecast_metric_meaning("Current regime probability")),
                        ("Forecast quality", f"{forecast['calibration']['label']} ({forecast['calibration']['score']}/100)", _forecast_metric_meaning("Forecast quality")),
                        ("Daily jump probability", format_percent(forecast["jump_process"]["jump_probability"]), _forecast_metric_meaning("Daily jump probability")),
                        ("Context drift bias", format_percent(forecast["context_bias"]), _forecast_metric_meaning("Context drift bias")),
                        ("Opportunity score", f"{forecast['context_scores']['opportunity']:.2f}", _forecast_metric_meaning("Opportunity score")),
                        ("Stress score", f"{forecast['context_scores']['stress']:.2f}", _forecast_metric_meaning("Stress score")),
                        ("Geopolitical proxy score", f"{forecast['context_scores']['geopolitical']:.2f}", _forecast_metric_meaning("Geopolitical proxy score")),
                        ("Behavior score", f"{forecast['context_scores']['behavior']:.2f}", _forecast_metric_meaning("Behavior score")),
                        ("AR(1) intercept", f"{forecast['mean_model']['intercept']:.5f}", _forecast_metric_meaning("AR(1) intercept")),
                        ("AR(1) phi", f"{forecast['mean_model']['phi']:.3f}", _forecast_metric_meaning("AR(1) phi")),
                    ],
                    columns=["Diagnostic", "Value", "What It Means"],
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
        st.markdown("### Model Governance")
        render_section_intro(
            "Why This Section Matters",
            "Εδώ το app σταματά να σου δίνει απλώς outputs και σου δείχνει πόσο αξίζει να τα εμπιστευτείς. Είναι το section που ξεχωρίζει καλή ανάλυση από ψευδαίσθηση ακρίβειας.",
        )
        if valuation_available:
            st.dataframe(
                pd.DataFrame(
                    [
                        ("Primary valuation method", valuation["method_used"], "Το κύριο valuation framework που χρησιμοποιήθηκε."),
                        ("Confidence", f"{valuation['confidence']['label']} ({valuation['confidence']['score']}/100)", "Η συνολική εμπιστοσύνη στο valuation output."),
                        ("Data quality", f"{valuation['data_quality']['label']} ({valuation['data_quality']['score']}/100)", "Η ποιότητα του dataset πριν καν εφαρμοστεί το valuation model."),
                        ("Verdict", valuation["verdict"], "Το τελικό intrinsic stance του valuation engine."),
                        ("90% intrinsic range", format_currency(valuation["uncertainty"]["intrinsic_range_90"], dataset.currency), "Πόσο φαρδύ είναι το πιθανό fair-value band."),
                        ("Stack note", valuation.get("valuation_stack_note", "N/A"), "Σύντομη περιγραφή του valuation stack."),
                    ],
                    columns=["Valuation evidence", "Value", "What It Means"],
                ),
                use_container_width=True,
                hide_index=True,
            )
            if valuation["confidence"]["reasons"]:
                render_takeaways("Why The Valuation Confidence Looks Like This", valuation["confidence"]["reasons"])
            if valuation.get("data_quality", {}).get("reasons"):
                render_takeaways("Why The Data Quality Looks Like This", valuation["data_quality"]["reasons"])
        else:
            st.info("Δεν υπάρχουν αρκετά fundamentals για valuation reliability diagnostics σε αυτό το ticker.")

        if forecast_available:
            calibration = forecast["calibration"]
            st.dataframe(
                pd.DataFrame(
                    [
                        ("Forecast quality", f"{calibration['label']} ({calibration['score']}/100)", "Η συνολική στατιστική ποιότητα του forecast engine."),
                        ("Calibration sample", f"{calibration['sample_days']} days", "Πόσο πρόσφατο history χρησιμοποιήθηκε για calibration diagnostics."),
                        ("Sign accuracy", format_percent(calibration["sign_accuracy"]), "Πόσο συχνά έπιανε σωστά την κατεύθυνση."),
                        ("90% interval coverage", format_percent(calibration["interval_90_coverage"]), "Πόσο συχνά το realized αποτέλεσμα έμενε μέσα στο 90% band."),
                        ("Left-tail hit rate", format_percent(calibration["left_tail_hit_rate"]), "Πόσο συχνά το realized return έσπαγε το αριστερό tail του band."),
                        ("Predicted daily vol", format_percent(calibration["predicted_daily_vol"]), "Η volatility που περίμενε το μοντέλο."),
                        ("Realized daily vol", format_percent(calibration["realized_daily_vol"]), "Η volatility που τελικά είδαμε."),
                        ("Realized / predicted vol", f"{calibration['vol_ratio']:.2f}x" if calibration["vol_ratio"] is not None else "N/A", "Δείχνει αν το μοντέλο υποτίμησε ή υπερεκτίμησε τη μεταβλητότητα."),
                    ],
                    columns=["Forecast evidence", "Value", "What It Means"],
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(calibration["summary"])
        else:
            st.info("Δεν υπάρχουν forecast reliability diagnostics για αυτό το run.")

        if forecast_available:
            st.markdown("### Volatility Ensemble")
            if not forecast["volatility_models"].empty:
                volatility_models = forecast["volatility_models"].copy()
                volatility_models["weight"] = volatility_models["weight"].map(format_percent)
                volatility_models["aic"] = volatility_models["aic"].map(lambda value: f"{value:.1f}" if pd.notna(value) else "N/A")
                volatility_models["bic"] = volatility_models["bic"].map(lambda value: f"{value:.1f}" if pd.notna(value) else "N/A")
                st.dataframe(volatility_models, use_container_width=True, hide_index=True)
            else:
                st.info("Δεν ήταν διαθέσιμο volatility ensemble breakdown.")

            st.markdown("### Data Coverage")
            st.dataframe(
                pd.DataFrame(
                    [
                        ("Coverage label", dataset.raw_info.get("coverage_label", "N/A"), "Γρήγορη ένδειξη σταθερότητας του free-data stack για το ticker."),
                        ("Coverage note", dataset.raw_info.get("coverage_note", "N/A"), "Σύντομη ερμηνεία για το πού είναι δυνατή και πού όχι η κάλυψη."),
                        ("Primary price provider", dataset.raw_info.get("price_provider", "N/A"), "Ο provider που τελικά έδωσε το usable price history."),
                        ("SEC fundamentals available", "Yes" if dataset.raw_info.get("sec_fundamentals_available") else "No", "Αν υπάρχουν annual SEC fundamentals για valuation."),
                        ("Context assets used", ", ".join(forecast.get("context_assets_used", [])) or "none", "Τα proxy assets που χρησιμοποιήθηκαν στο context model."),
                        ("Macro context used", ", ".join(forecast.get("macro_context_used", [])) or "none", "Τα macro / geopolitical series που μπήκαν στο forecast bias layer."),
                        ("Sector ETF proxy", dataset.raw_info.get("sector_etf") or "none", "Το ETF που χρησιμοποιήθηκε σαν sector-relative context proxy."),
                        ("Cache hints", ", ".join(dataset.raw_info.get("cache_messages", [])) or "Fresh API responses", "Σου λέει ποια parts ήρθαν από cache και ποια όχι."),
                    ],
                    columns=["Coverage item", "Value", "What It Means"],
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[4]:
        render_guide_tab(settings_impact, budget, compute_profile, analysis_mode)


if __name__ == "__main__":
    main()
