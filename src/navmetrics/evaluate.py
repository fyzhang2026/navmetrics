"""Public evaluation entry point."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from navmetrics.catalog import GROUPS, get_metric, list_metrics, stem_of
from navmetrics.engine import compute
from navmetrics.errors import PerformanceEvaluationError
from navmetrics.inputs import prepare
from navmetrics.report import PerformanceReport


def evaluate(
    nav: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free: float | pd.Series = 0.0,
    mar: float | pd.Series = 0.0,
    confidence: float | Sequence[float] = 0.95,
    periods_per_year: float = 252,
    metrics: str | Sequence[str] | None = None,
    *,
    rate_convention: str = "period",
    day_count: float = 365.25,
    sterling_drawdowns: int = 5,
    kappa_order: int = 3,
    farinelli_tibiletti_orders: Sequence[tuple[int, int]] = ((1, 1), (2, 1)),
    rolling_windows: Sequence[str | int] = ("6M", "1Y", "3Y"),
    regime_threshold: float = 0.0,
    psr_threshold: float = 0.0,
    min_rolling_periods: int = 10,
    align: str = "strict",
) -> PerformanceReport:
    """Evaluate a strategy NAV curve against an optional benchmark.

    Parameters
    ----------
    nav, benchmark:
        Positive net-asset-value series indexed by date. Both are performance
        indexes: external cash flows have already been removed. They must be
        denominated in the same currency and include reinvested distributions.
    risk_free, mar:
        A scalar or a series on the same dates. With ``rate_convention='period'``
        each value is the simple return over the interval ending on that date.
        With ``rate_convention='annualized'`` each value is an annualized rate
        and is converted with actual days over ``day_count``.
    confidence:
        One or more tail probabilities strictly between 0.5 and 1.
    periods_per_year:
        Annualization factor ``q``. Daily data conventionally uses 252, weekly
        52, and monthly 12.
    metrics:
        ``None`` returns every statistic the inputs can support. Pass metric
        ids (``"sharpe"``), expanded ids (``"historical_var_95"``), stems
        (``"historical_var"``), or groups (``"risk_adjusted"``).
    psr_threshold:
        Annualized Sharpe hurdle for the probabilistic Sharpe ratio. Zero by
        default. It is converted to a per-period hurdle by dividing by ``sqrt(q)``.
    regime_threshold:
        A benchmark simple return above this value is an up-market period.
    """
    requested = _normalize_metrics(metrics)
    if benchmark is None and requested is not None:
        blocked = [item for item in requested if item != "all" and _requires_benchmark(item)]
        if blocked:
            joined = ", ".join(blocked)
            raise PerformanceEvaluationError(
                f"These selections need a benchmark NAV series: {joined}."
            )

    data = prepare(
        nav,
        benchmark=benchmark,
        risk_free=risk_free,
        mar=mar,
        confidence=confidence,
        periods_per_year=periods_per_year,
        rate_convention=rate_convention,
        day_count=day_count,
        sterling_drawdowns=sterling_drawdowns,
        kappa_order=kappa_order,
        farinelli_tibiletti_orders=tuple(farinelli_tibiletti_orders),
        rolling_windows=tuple(rolling_windows),
        regime_threshold=regime_threshold,
        psr_threshold=psr_threshold,
        min_rolling_periods=min_rolling_periods,
        align=align,
    )
    scalars, frames = compute(data)
    scalars, frames = _select(requested, scalars, frames)
    meta = {
        "start": data.nav.index[0],
        "end": data.nav.index[-1],
        "n_periods": int(data.returns.shape[0]),
        "periods_per_year": data.periods_per_year,
        "rate_convention": data.rate_convention,
        "day_count": data.day_count,
        "confidence": data.confidence,
        "sterling_drawdowns": data.sterling_drawdowns,
        "kappa_order": data.kappa_order,
        "farinelli_tibiletti_orders": data.ft_orders,
        "rolling_windows": tuple(window.name for window in data.windows),
        "regime_threshold": data.regime_threshold,
        "psr_threshold": data.psr_threshold,
        "min_rolling_periods": data.min_rolling_periods,
        "align": data.align,
        "has_benchmark": data.benchmark is not None,
    }
    return PerformanceReport(pd.Series(scalars, dtype=float), frames, meta)


def _normalize_metrics(metrics: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if metrics is None:
        return None
    if isinstance(metrics, str):
        requested = (metrics,)
    else:
        requested = tuple(str(item) for item in metrics)
    if not requested:
        raise PerformanceEvaluationError("metrics must contain at least one id, or be None.")
    return requested


def _requires_benchmark(item: str) -> bool:
    if item == "relative":
        return True
    stem = stem_of(item)
    if stem is None:
        return False
    metric = get_metric(stem)
    return bool(metric and metric.requires_benchmark)


def _group_of(metric_id: str) -> str | None:
    metric = get_metric(metric_id)
    if metric is not None:
        return metric.group
    stem = stem_of(metric_id)
    if stem is None:
        return None
    parent = get_metric(stem)
    return None if parent is None else parent.group


def _select(
    requested: tuple[str, ...] | None,
    scalars: dict[str, float],
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    if requested is None or "all" in requested:
        return scalars, frames
    chosen_scalars: list[str] = []
    chosen_frames: list[str] = []
    for item in requested:
        scalar_hits, frame_hits = _expand(item, scalars, frames)
        chosen_scalars.extend(scalar_hits)
        chosen_frames.extend(frame_hits)
    scalar_ids = list(dict.fromkeys(chosen_scalars))
    frame_ids = list(dict.fromkeys(chosen_frames))
    return (
        {key: scalars[key] for key in scalar_ids},
        {key: frames[key] for key in frame_ids},
    )


def _expand(
    item: str,
    scalars: dict[str, float],
    frames: dict[str, pd.DataFrame],
) -> tuple[list[str], list[str]]:
    if item in GROUPS:
        return (
            [key for key in scalars if _group_of(key) == item],
            [key for key in frames if _group_of(key) == item],
        )
    scalar_hits = [item] if item in scalars else []
    frame_hits = [item] if item in frames else []
    spec = get_metric(item)
    if spec is not None and spec.kind != "scalar":
        scalar_hits.extend(key for key in scalars if stem_of(key) == item and key not in scalar_hits)
        if spec.kind == "frame" and item in frames and item not in frame_hits:
            frame_hits.append(item)
    if not scalar_hits and not frame_hits:
        stem = stem_of(item)
        if stem is not None and item != stem:
            raise PerformanceEvaluationError(
                f"{item!r} was not computed from these inputs. "
                "Check confidence, kappa_order, farinelli_tibiletti_orders, and rolling_windows."
            )
        known = ", ".join(list_metrics()["id"].head(8).tolist())
        raise PerformanceEvaluationError(
            f"Unknown metric {item!r}. Call list_metrics() for the catalog. "
            f"The first catalog ids are: {known}."
        )
    return scalar_hits, frame_hits
