from __future__ import annotations

from typing import Any

import numpy as np

from services.models import StockDataset


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, "", "None", "null", "-"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_first_value(mapping: dict[str, Any], candidates: tuple[str, ...]) -> float | None:
    for candidate in candidates:
        if candidate not in mapping:
            continue
        value = _to_float(mapping.get(candidate))
        if value is not None and np.isfinite(value):
            return float(value)
    return None


def _earnings_event_info(dataset: StockDataset) -> dict[str, Any]:
    info = dict(dataset.raw_info.get("earnings_summary") or {})
    days_to_event = _to_float(info.get("days_to_event"))
    if days_to_event is not None:
        days_to_event = int(round(days_to_event))
    return {
        "event_date": info.get("event_date"),
        "days_to_event": days_to_event,
        "time": info.get("time") or "N/A",
        "event_window_active": bool(info.get("event_window_active")),
        "eps_estimate": _to_float(info.get("eps_estimate")),
        "revenue_estimate": _to_float(info.get("revenue_estimate")),
        "is_upcoming": bool(info.get("is_upcoming")),
    }


def _transition_probability_to_bear(forecast: dict[str, Any], horizon_days: int) -> float:
    transition_df = forecast.get("transition_matrix")
    current_regime = str(forecast.get("current_regime") or "")
    if transition_df is None or horizon_days <= 0:
        return 0.0
    try:
        matrix = transition_df.to_numpy(dtype=float)
        columns = list(transition_df.columns)
        rows = list(transition_df.index)
    except Exception:
        return 0.0

    from_label = f"From {current_regime}"
    bear_label = "To Bear"
    if from_label not in columns or bear_label not in rows:
        return 0.0

    current_index = columns.index(from_label)
    state = np.zeros(matrix.shape[1], dtype=float)
    state[current_index] = 1.0
    step = matrix.copy()
    horizon_matrix = np.linalg.matrix_power(step, max(horizon_days, 1))
    ending = horizon_matrix @ state
    bear_index = rows.index(bear_label)
    return float(np.clip(ending[bear_index], 0.0, 1.0))


def _build_tactical_levels(current_price: float, horizon_rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    best = max(horizon_rows, key=lambda item: item["edge_score"])
    tactical_target_return = float(np.clip(max(best["median_return"], 0.03), 0.03, 0.10))
    stop_return = float(np.clip(max(0.04, 0.70 * tactical_target_return), 0.04, 0.08))
    return (
        current_price * (1.0 + tactical_target_return),
        current_price * (1.0 - stop_return),
        tactical_target_return,
        stop_return,
    )


def _build_exit_window_analysis(
    dataset: StockDataset,
    forecast: dict[str, Any],
    horizon_rows: list[dict[str, Any]],
    event: dict[str, Any],
) -> dict[str, Any]:
    paths = np.asarray(forecast.get("paths"), dtype=float)
    if paths.ndim != 2 or paths.shape[1] < 2:
        return {
            "target_price": None,
            "stop_price": None,
            "target_return": None,
            "stop_return": None,
            "best_day": None,
            "best_window": None,
            "rows": [],
        }

    current_price = float(dataset.current_price)
    target_price, stop_price, target_return, stop_return = _build_tactical_levels(current_price, horizon_rows)
    horizon = paths.shape[1] - 1
    target_hits = paths[:, 1:] >= target_price
    stop_hits = paths[:, 1:] <= stop_price

    first_target = np.where(target_hits.any(axis=1), target_hits.argmax(axis=1) + 1, horizon + 1)
    first_stop = np.where(stop_hits.any(axis=1), stop_hits.argmax(axis=1) + 1, horizon + 1)
    running_peaks = np.maximum.accumulate(paths, axis=1)
    drawdowns = paths / np.maximum(running_peaks, 1e-8) - 1.0

    rows: list[dict[str, Any]] = []
    for day in range(1, horizon + 1):
        target_hit_prob = float(np.mean(first_target <= day))
        stop_hit_prob = float(np.mean(first_stop <= day))
        target_first_prob = float(np.mean((first_target <= day) & (first_target < first_stop)))
        stop_first_prob = float(np.mean((first_stop <= day) & (first_stop < first_target)))
        day_returns = paths[:, day] / current_price - 1.0
        expected_return = float(np.mean(day_returns))
        expected_drawdown = float(np.mean(np.min(drawdowns[:, : day + 1], axis=1)))
        regime_deterioration = _transition_probability_to_bear(forecast, day)

        event_hazard = 0.0
        if event.get("days_to_event") is not None:
            event_days = int(event["days_to_event"])
            if 0 <= event_days <= day:
                event_hazard = 0.18 if event_days <= 2 else 0.10
            elif -1 <= event_days <= 1:
                event_hazard = 0.07

        score = (
            1.25 * target_hit_prob
            + 1.10 * target_first_prob
            + 1.65 * expected_return
            - 1.05 * abs(expected_drawdown)
            - 0.75 * stop_first_prob
            - 0.45 * regime_deterioration
            - event_hazard
        )
        rows.append(
            {
                "day": day,
                "target_hit_probability": target_hit_prob,
                "stop_hit_probability": stop_hit_prob,
                "target_first_probability": target_first_prob,
                "stop_first_probability": stop_first_prob,
                "expected_return": expected_return,
                "expected_max_drawdown": expected_drawdown,
                "regime_deterioration_probability": regime_deterioration,
                "event_hazard": event_hazard,
                "exit_score": score,
            }
        )

    best_row = max(rows, key=lambda item: item["exit_score"])
    window_start = max(1, best_row["day"] - 2)
    window_end = min(horizon, best_row["day"] + 2)
    hit_days = first_target[first_target <= horizon]
    median_days_to_target = float(np.median(hit_days)) if len(hit_days) else None

    return {
        "target_price": target_price,
        "stop_price": stop_price,
        "target_return": target_return,
        "stop_return": stop_return,
        "best_day": best_row["day"],
        "best_window": f"Day {window_start} to Day {window_end}",
        "median_days_to_target": median_days_to_target,
        "rows": rows,
    }


def _short_term_horizon_table(dataset: StockDataset, forecast: dict[str, Any]) -> list[dict[str, Any]]:
    current_price = float(dataset.current_price)
    event = _earnings_event_info(dataset)
    rows: list[dict[str, Any]] = []

    for horizon in forecast.get("horizon_summary", []):
        mean_return = float(horizon["mean"] / current_price - 1.0)
        median_return = float(horizon["median"] / current_price - 1.0)
        downside_tail = float(max(0.0, 1.0 - horizon["p05"] / current_price))
        moderate_drawdown = float(max(0.0, 1.0 - horizon["p25"] / current_price))
        upside_tail = float(max(0.0, horizon["p95"] / current_price - 1.0))
        probability_upside = float(horizon["probability_upside"])
        probability_up_5pct = float(horizon["probability_up_5pct"])
        probability_down_5pct = float(horizon["probability_down_5pct"])

        event_penalty = 0.0
        if event["days_to_event"] is not None and 0 <= event["days_to_event"] <= horizon["days"]:
            event_penalty = 0.0075 if event["days_to_event"] > 2 else 0.015
        elif event["days_to_event"] is not None and abs(event["days_to_event"]) <= 2:
            event_penalty = 0.01

        expected_edge = (
            0.55 * mean_return
            + 0.20 * median_return
            + 0.20 * probability_up_5pct * upside_tail
            - 0.55 * probability_down_5pct * downside_tail
            - 0.20 * moderate_drawdown
            - event_penalty
        )
        edge_score = float(
            np.clip(
                50
                + 90 * (probability_upside - 0.50)
                + 220 * expected_edge
                - 90 * probability_down_5pct * downside_tail,
                5,
                95,
            )
        )
        rows.append(
            {
                "horizon": horizon["label"],
                "days": int(horizon["days"]),
                "mean_return": mean_return,
                "median_return": median_return,
                "probability_upside": probability_upside,
                "probability_up_5pct": probability_up_5pct,
                "probability_down_5pct": probability_down_5pct,
                "expected_edge": expected_edge,
                "edge_score": edge_score,
                "event_penalty": event_penalty,
                "expected_max_drawdown": float(horizon.get("expected_max_drawdown") or 0.0),
            }
        )

    return rows


def build_short_term_report(
    dataset: StockDataset,
    forecast: dict[str, Any],
    valuation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not forecast.get("available"):
        return {
            "mode": "short_term",
            "available": False,
            "error": forecast.get("error", "Το short-term report χρειάζεται usable forecast."),
        }

    event = _earnings_event_info(dataset)
    horizon_rows = _short_term_horizon_table(dataset, forecast)
    if not horizon_rows:
        return {
            "mode": "short_term",
            "available": False,
            "error": "Δεν βγήκαν usable horizon rows από το forecast engine.",
        }

    best_horizon = max(horizon_rows, key=lambda item: item["edge_score"])
    exit_window = _build_exit_window_analysis(dataset, forecast, horizon_rows, event)
    calibration_score = float(forecast.get("calibration", {}).get("score") or 0.0)
    opportunity = float(forecast.get("context_scores", {}).get("opportunity") or 0.0)
    stress = float(forecast.get("context_scores", {}).get("stress") or 0.0)
    geopolitical = float(forecast.get("context_scores", {}).get("geopolitical") or 0.0)
    jump_probability = float(forecast.get("jump_process", {}).get("jump_probability") or 0.0)
    regime_probability = float(forecast.get("current_regime_probability") or 0.0)

    score = 42.0
    score += 0.34 * calibration_score
    score += 0.30 * (best_horizon["edge_score"] - 50.0)
    score += 8.0 * max(0.0, opportunity)
    score -= 7.0 * max(0.0, stress)
    score -= 4.0 * max(0.0, geopolitical)
    score -= 140.0 * jump_probability
    score += 10.0 * max(0.0, regime_probability - 0.45)

    if dataset.raw_info.get("history_mode") == "proxy":
        score -= 8.0
    elif dataset.raw_info.get("history_mode") == "minimal":
        score -= 18.0

    if valuation and valuation.get("available"):
        margin_of_safety = float(valuation.get("margin_of_safety") or 0.0)
        score += float(np.clip(margin_of_safety * 25.0, -6.0, 6.0))

    if event["days_to_event"] is not None:
        if 0 <= event["days_to_event"] <= 2:
            score -= 18.0
        elif 0 <= event["days_to_event"] <= 7:
            score -= 10.0
        elif -1 <= event["days_to_event"] <= 1:
            score -= 8.0

    score = float(np.clip(score, 5.0, 95.0))

    no_trade_reasons: list[str] = []
    if calibration_score < 48:
        no_trade_reasons.append("Το πρόσφατο calibration του forecast engine είναι αδύναμο.")
    if best_horizon["expected_edge"] <= 0.0025:
        no_trade_reasons.append("Το expected edge βγαίνει πολύ μικρό για να δικαιολογεί νέο trade.")
    if dataset.raw_info.get("history_mode") == "minimal":
        no_trade_reasons.append("Δεν υπάρχει κανονικό direct/proxy history για robust short-term signal.")
    if event["days_to_event"] is not None and 0 <= event["days_to_event"] <= 2:
        no_trade_reasons.append("Υπάρχει πολύ κοντινό earnings/event window, άρα το near-term distribution είναι πιο binary.")
    if stress >= 1.15 and best_horizon["probability_upside"] < 0.57:
        no_trade_reasons.append("Το stress regime είναι ανεβασμένο χωρίς αρκετά καθαρό ανοδικό probability edge.")

    no_trade = len(no_trade_reasons) > 0
    if no_trade:
        action = "No-Trade"
    elif score >= 72 and best_horizon["probability_upside"] >= 0.58:
        action = "Actionable"
    elif score >= 56:
        action = "Watch"
    else:
        action = "No-Trade"

    if score >= 72 and action != "No-Trade":
        setup = "Favorable"
    elif score >= 56:
        setup = "Mixed"
    elif score >= 42:
        setup = "Fragile"
    else:
        setup = "Adverse"

    reasons = [
        f"Το καλύτερο horizon αυτή τη στιγμή βγαίνει το `{best_horizon['horizon']}` με edge score περίπου {best_horizon['edge_score']:.0f}/100.",
        f"Το forecast calibration είναι `{forecast['calibration']['label']}` ({forecast['calibration']['score']}/100), κάτι που επηρεάζει άμεσα το πόσο επιθετικά διαβάζουμε το signal.",
        f"Το regime αυτή τη στιγμή είναι `{forecast['current_regime']}` με πιθανότητα περίπου {forecast['current_regime_probability']:.0%}.",
    ]
    if exit_window.get("best_window"):
        reasons.append(
            f"Το exit-timing layer δείχνει καλύτερο tactical window γύρω από το `{exit_window['best_window']}`."
        )
    if event["days_to_event"] is not None:
        if event["days_to_event"] >= 0:
            reasons.append(
                f"Υπάρχει earnings/event marker σε περίπου {event['days_to_event']} ημέρες, άρα το short-term αποτέλεσμα θέλει περισσότερη πειθαρχία."
            )
        else:
            reasons.append(
                f"Το earnings/event window είναι πολύ πρόσφατο ({event['days_to_event']} ημέρες), άρα το price process μπορεί να παραμένει ακόμη σε post-event digestion."
            )
    if dataset.raw_info.get("history_mode") == "proxy":
        reasons.append("Το short-term signal πατά σε proxy history fallback, άρα το διαβάζουμε πιο ταπεινά.")

    risk_controls = [
        "Μην κυνηγάς trade όταν το `Action` βγαίνει `No-Trade`, ακόμα κι αν το fan chart δείχνει θεωρητικό upside.",
        "Όσο πιο κοντά είναι earnings ή άλλο event window, τόσο περισσότερο μετρά το volatility risk και λιγότερο το central path.",
        "Αν `Stress score` και `Geopolitical score` είναι και τα δύο ψηλά, το πιο σωστό διάβασμα είναι μικρότερο size ή πλήρης αποχή.",
    ]
    if no_trade_reasons:
        risk_controls.extend(no_trade_reasons)

    return {
        "mode": "short_term",
        "available": True,
        "setup": setup,
        "action": action,
        "trade_setup_score": int(round(score)),
        "best_horizon": best_horizon["horizon"],
        "best_horizon_days": best_horizon["days"],
        "expected_edge": best_horizon["expected_edge"],
        "signal_confidence": int(round(max(5.0, min(95.0, 0.65 * calibration_score + 0.35 * score)))),
        "no_trade": no_trade,
        "horizon_table": horizon_rows,
        "event_risk": event,
        "exit_window": exit_window,
        "reasons": reasons,
        "risk_controls": risk_controls,
    }


def build_long_term_report(
    dataset: StockDataset,
    valuation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ratios = dict(dataset.raw_info.get("ratios_ttm") or {})
    key_metrics = dict(dataset.raw_info.get("key_metrics_ttm") or {})
    estimates = dict(dataset.raw_info.get("analyst_estimates") or {})
    price_target = dict(dataset.raw_info.get("price_target_summary") or {})
    owner_earnings = dict(dataset.raw_info.get("owner_earnings_summary") or {})

    profitability_score = 45.0
    balance_sheet_score = 45.0
    estimate_anchor_score = 35.0

    gross_margin = _extract_first_value(ratios, ("grossProfitMarginTTM", "grossProfitMargin"))
    operating_margin = _extract_first_value(ratios, ("operatingProfitMarginTTM", "operatingProfitMargin"))
    roe = _extract_first_value(ratios, ("returnOnEquityTTM", "returnOnEquity", "roeTTM")) or dataset.return_on_equity
    roic = _extract_first_value(key_metrics, ("roicTTM", "roic", "returnOnInvestedCapital"))
    debt_to_equity = _extract_first_value(ratios, ("debtEquityRatioTTM", "debtEquityRatio"))
    current_ratio = _extract_first_value(ratios, ("currentRatioTTM", "currentRatio"))
    interest_coverage = _extract_first_value(ratios, ("interestCoverageTTM", "interestCoverage"))
    forward_revenue_growth = _to_float(estimates.get("forward_revenue_growth"))
    forward_eps_growth = _to_float(estimates.get("forward_eps_growth"))
    owner_earnings_value = _extract_first_value(owner_earnings, ("ownersEarnings", "ownerEarnings", "ownerEarningsTTM"))
    price_target_mean = _extract_first_value(
        price_target,
        ("allTimeAveragePriceTarget", "avgPriceTarget", "averagePriceTarget", "priceTargetAverage"),
    )

    for value, weight in (
        (gross_margin, 12.0),
        (operating_margin, 14.0),
        (roe, 12.0),
        (roic, 12.0),
    ):
        if value is None:
            continue
        profitability_score += float(np.clip(value * weight, -8.0, 12.0))

    if debt_to_equity is not None:
        balance_sheet_score += float(np.clip((1.0 - debt_to_equity) * 12.0, -10.0, 10.0))
    if current_ratio is not None:
        balance_sheet_score += float(np.clip((current_ratio - 1.0) * 8.0, -8.0, 10.0))
    if interest_coverage is not None:
        balance_sheet_score += float(np.clip(interest_coverage * 1.2, -6.0, 12.0))

    if forward_revenue_growth is not None:
        estimate_anchor_score += float(np.clip(forward_revenue_growth * 70.0, -10.0, 16.0))
    if forward_eps_growth is not None:
        estimate_anchor_score += float(np.clip(forward_eps_growth * 50.0, -10.0, 14.0))
    if price_target_mean is not None and dataset.current_price > 0:
        estimate_anchor_score += float(np.clip((price_target_mean / dataset.current_price - 1.0) * 18.0, -8.0, 8.0))

    profitability_score = float(np.clip(profitability_score, 5.0, 95.0))
    balance_sheet_score = float(np.clip(balance_sheet_score, 5.0, 95.0))
    estimate_anchor_score = float(np.clip(estimate_anchor_score, 5.0, 95.0))

    valuation_score = 35.0
    valuation_context = "Δεν υπάρχει πλήρες intrinsic valuation anchor για αυτό το run."
    if valuation and valuation.get("available"):
        valuation_confidence = float(valuation.get("confidence", {}).get("score") or 0.0)
        margin_of_safety = float(valuation.get("margin_of_safety") or 0.0)
        valuation_score = float(np.clip(45.0 + 0.35 * valuation_confidence + 28.0 * margin_of_safety, 5.0, 95.0))
        valuation_context = (
            f"Το primary fair-value anchor βγαίνει `{valuation['verdict']}` με confidence "
            f"{valuation['confidence']['score']}/100."
        )

    overall = float(
        np.clip(
            0.40 * valuation_score
            + 0.25 * profitability_score
            + 0.20 * balance_sheet_score
            + 0.15 * estimate_anchor_score,
            5.0,
            95.0,
        )
    )

    if valuation and valuation.get("available"):
        verdict = valuation.get("verdict")
        if overall >= 72 and verdict == "Undervalued":
            stance = "Attractive"
        elif overall >= 60 and verdict in {"Undervalued", "Fairly Priced", "Fair / Uncertain"}:
            stance = "Balanced"
        elif verdict == "Overvalued":
            stance = "Demanding"
        else:
            stance = "Selective"
    else:
        stance = "Insufficient Evidence"

    if stance == "Attractive":
        holding_action = "Accumulate"
    elif stance in {"Balanced", "Selective"}:
        holding_action = "Hold / Selective Add"
    elif stance == "Demanding":
        holding_action = "Hold / Avoid Chasing"
    else:
        holding_action = "Avoid Strong Conclusion"

    reasons = [
        valuation_context,
        f"Το profitability lens βγαίνει περίπου {profitability_score:.0f}/100 και το balance-sheet lens περίπου {balance_sheet_score:.0f}/100.",
        f"Το forward-estimate / market-consensus lens βγαίνει περίπου {estimate_anchor_score:.0f}/100, βασισμένο στα δωρεάν internet-backed estimate layers που ήταν διαθέσιμα.",
    ]
    if owner_earnings_value is not None:
        reasons.append("Υπήρχε owner-earnings lens, που βοηθά σαν συμπληρωματικός cash-flow cross-check.")
    if dataset.raw_info.get("sec_warning"):
        reasons.append("Το SEC layer δεν ήταν πλήρως διαθέσιμο σε αυτό το run, άρα το long-term confidence πέφτει.")
    if not dataset.raw_info.get("sec_fundamentals_available"):
        reasons.append("Χωρίς SEC annual fundamentals, το long-term conclusion πρέπει να διαβαστεί πιο προσεκτικά.")

    pillar_rows = [
        {
            "pillar": "Intrinsic valuation",
            "score": valuation_score,
            "commentary": valuation_context,
        },
        {
            "pillar": "Profitability quality",
            "score": profitability_score,
            "commentary": "Εξετάζει margins, ROE και όπου υπάρχει ROIC, για να δει αν η επιχείρηση δημιουργεί οικονομική αξία με επαναληψιμότητα.",
        },
        {
            "pillar": "Balance-sheet resilience",
            "score": balance_sheet_score,
            "commentary": "Εξετάζει leverage, liquidity και interest coverage ώστε να φανεί αν η κεφαλαιακή δομή αφήνει περιθώριο για execution.",
        },
        {
            "pillar": "Forward estimate anchor",
            "score": estimate_anchor_score,
            "commentary": "Χρησιμοποιεί forward internet-backed estimate layers όταν είναι διαθέσιμα, ώστε να μη μένουμε μόνο στο backward-looking picture.",
        },
    ]

    return {
        "mode": "long_term",
        "available": True,
        "stance": stance,
        "holding_action": holding_action,
        "overall_score": int(round(overall)),
        "pillar_rows": pillar_rows,
        "reasons": reasons,
        "owner_earnings_value": owner_earnings_value,
        "price_target_mean": price_target_mean,
        "forward_revenue_growth": forward_revenue_growth,
        "forward_eps_growth": forward_eps_growth,
    }
