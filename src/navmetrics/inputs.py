"""Validate and align NAV, benchmark, risk-free, and MAR inputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from navmetrics.errors import PerformanceEvaluationError

_WINDOW = re.compile(r"^(\d+)([MY])$")


@dataclass(frozen=True)
class Window:
    name: str
    offset: pd.DateOffset | None
    periods: int | None


@dataclass(frozen=True)
class Prepared:
    nav: pd.Series
    benchmark: pd.Series | None
    returns: pd.Series
    benchmark_returns: pd.Series | None
    risk_free: pd.Series
    mar: pd.Series
    confidence: tuple[float, ...]
    periods_per_year: float
    day_count: float
    rate_convention: str
    sterling_drawdowns: int
    kappa_order: int
    ft_orders: tuple[tuple[int, int], ...]
    windows: tuple[Window, ...]
    regime_threshold: float
    psr_threshold: float
    min_rolling_periods: int
    align: str


def prepare(
    nav,
    benchmark=None,
    risk_free: float | pd.Series = 0.0,
    mar: float | pd.Series = 0.0,
    confidence: float | tuple[float, ...] | list[float] = 0.95,
    periods_per_year: float = 252,
    rate_convention: str = "period",
    day_count: float = 365.25,
    sterling_drawdowns: int = 5,
    kappa_order: int = 3,
    farinelli_tibiletti_orders: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (
        (1, 1),
        (2, 1),
    ),
    rolling_windows: tuple[str | int, ...] | list[str | int] = ("6M", "1Y", "3Y"),
    regime_threshold: float = 0.0,
    psr_threshold: float = 0.0,
    min_rolling_periods: int = 10,
    align: str = "strict",
) -> Prepared:
    if align not in {"strict", "inner"}:
        raise PerformanceEvaluationError("align must be 'strict' or 'inner'.")
    if rate_convention not in {"period", "annualized"}:
        raise PerformanceEvaluationError("rate_convention must be 'period' or 'annualized'.")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise PerformanceEvaluationError("periods_per_year must be a positive finite number.")
    if not np.isfinite(day_count) or day_count <= 0:
        raise PerformanceEvaluationError("day_count must be a positive finite number.")
    if isinstance(sterling_drawdowns, bool) or int(sterling_drawdowns) < 1:
        raise PerformanceEvaluationError("sterling_drawdowns must be an integer >= 1.")
    if isinstance(kappa_order, bool) or int(kappa_order) < 1:
        raise PerformanceEvaluationError("kappa_order must be an integer >= 1.")
    if isinstance(min_rolling_periods, bool) or int(min_rolling_periods) < 2:
        raise PerformanceEvaluationError("min_rolling_periods must be an integer >= 2.")
    if not np.isfinite(regime_threshold):
        raise PerformanceEvaluationError("regime_threshold must be finite.")
    if not np.isfinite(psr_threshold):
        raise PerformanceEvaluationError("psr_threshold must be finite.")

    levels = _confidence_levels(confidence)
    orders = _ft_orders(farinelli_tibiletti_orders)
    windows = _unique_windows(rolling_windows)

    nav_series = _as_nav(nav, "nav")
    benchmark_series = None if benchmark is None else _as_nav(benchmark, "benchmark")
    risk_free_series = _as_rate(risk_free, "risk_free")
    mar_series = _as_rate(mar, "mar")

    common = _common_index(
        nav_series,
        benchmark_series,
        risk_free_series,
        mar_series,
        align,
    )
    nav_series = nav_series.reindex(common)
    if benchmark_series is not None:
        benchmark_series = benchmark_series.reindex(common)
    risk_free_on_nav = _broadcast_rate(risk_free_series, common, "risk_free")
    mar_on_nav = _broadcast_rate(mar_series, common, "mar")
    if len(nav_series) < 2:
        raise PerformanceEvaluationError("At least two NAV dates are required.")

    period_rf = _to_period(risk_free_on_nav, rate_convention, day_count, "risk_free")
    period_mar = _to_period(mar_on_nav, rate_convention, day_count, "mar")
    returns = nav_series.pct_change().iloc[1:]
    benchmark_returns = None
    if benchmark_series is not None:
        benchmark_returns = benchmark_series.pct_change().iloc[1:]

    _reject_nonfinite(returns, "strategy returns")
    if benchmark_returns is not None:
        _reject_nonfinite(benchmark_returns, "benchmark returns")
    _reject_nonfinite(period_rf, "risk-free rates")
    _reject_nonfinite(period_mar, "MAR rates")

    return Prepared(
        nav=nav_series,
        benchmark=benchmark_series,
        returns=returns,
        benchmark_returns=benchmark_returns,
        risk_free=period_rf,
        mar=period_mar,
        confidence=levels,
        periods_per_year=float(periods_per_year),
        day_count=float(day_count),
        rate_convention=rate_convention,
        sterling_drawdowns=int(sterling_drawdowns),
        kappa_order=int(kappa_order),
        ft_orders=orders,
        windows=windows,
        regime_threshold=float(regime_threshold),
        psr_threshold=float(psr_threshold),
        min_rolling_periods=int(min_rolling_periods),
        align=align,
    )


def _confidence_levels(confidence) -> tuple[float, ...]:
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        values = [float(confidence)]
    else:
        values = [float(level) for level in confidence]
    if not values:
        raise PerformanceEvaluationError("confidence must contain at least one level.")
    cleaned: list[float] = []
    for level in values:
        if not np.isfinite(level) or not 0.5 < level < 1:
            raise PerformanceEvaluationError(
                "Each confidence level must be strictly between 0.5 and 1."
            )
        if level not in cleaned:
            cleaned.append(level)
    return tuple(cleaned)


def _ft_orders(orders) -> tuple[tuple[int, int], ...]:
    cleaned: list[tuple[int, int]] = []
    for order in orders:
        if len(order) != 2:
            raise PerformanceEvaluationError(
                "Each Farinelli-Tibiletti order must be a pair (p, q)."
            )
        p_order, q_order = int(order[0]), int(order[1])
        if p_order < 1 or q_order < 1:
            raise PerformanceEvaluationError(
                "Farinelli-Tibiletti orders must be integers >= 1."
            )
        pair = (p_order, q_order)
        if pair not in cleaned:
            cleaned.append(pair)
    if not cleaned:
        raise PerformanceEvaluationError("farinelli_tibiletti_orders must not be empty.")
    return tuple(cleaned)


def _unique_windows(rolling_windows) -> tuple[Window, ...]:
    windows: list[Window] = []
    seen: set[str] = set()
    for window in rolling_windows:
        parsed = _parse_window(window)
        if parsed.name in seen:
            continue
        seen.add(parsed.name)
        windows.append(parsed)
    if not windows:
        raise PerformanceEvaluationError("rolling_windows must contain at least one window.")
    return tuple(windows)


def _parse_window(window: str | int) -> Window:
    if isinstance(window, bool):
        raise PerformanceEvaluationError(f"Invalid rolling window {window!r}.")
    if isinstance(window, (int, np.integer)):
        periods = int(window)
        if periods < 2:
            raise PerformanceEvaluationError("A period rolling window must be >= 2.")
        return Window(name=str(periods), offset=None, periods=periods)
    text = str(window).strip().upper()
    match = _WINDOW.fullmatch(text)
    if match is None:
        raise PerformanceEvaluationError(
            f"Invalid rolling window {window!r}. Use '6M', '1Y', or an integer period count."
        )
    count = int(match.group(1))
    if count < 1:
        raise PerformanceEvaluationError(f"Invalid rolling window {window!r}.")
    unit = match.group(2)
    offset = pd.DateOffset(months=count) if unit == "M" else pd.DateOffset(years=count)
    return Window(name=f"{count}{unit}", offset=offset, periods=None)


def _as_nav(obj, name: str) -> pd.Series:
    series = _as_series(obj, name)
    if (series <= 0).any() or not np.isfinite(series.to_numpy()).all():
        raise PerformanceEvaluationError(f"{name} must be strictly positive and finite.")
    return series


def _as_rate(obj, name: str):
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if not np.isfinite(obj):
            raise PerformanceEvaluationError(f"{name} must be finite.")
        return float(obj)
    return _as_series(obj, name)


def _as_series(obj, name: str) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise PerformanceEvaluationError(f"{name} DataFrame must have exactly one column.")
        series = obj.iloc[:, 0].copy()
    elif isinstance(obj, pd.Series):
        series = obj.copy()
    else:
        raise PerformanceEvaluationError(
            f"{name} must be a pandas Series with a DatetimeIndex."
        )
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.DatetimeIndex(pd.to_datetime(series.index))
        except (TypeError, ValueError) as exc:
            raise PerformanceEvaluationError(f"{name} index must be datetimes.") from exc
    if series.index.tz is not None:
        series.index = series.index.tz_convert("UTC").tz_localize(None)
    series = series.sort_index()
    if series.index.has_duplicates:
        raise PerformanceEvaluationError(f"{name} has duplicate dates.")
    try:
        series = series.astype(float)
    except (TypeError, ValueError) as exc:
        raise PerformanceEvaluationError(f"{name} values must be numeric.") from exc
    if series.isna().any():
        raise PerformanceEvaluationError(f"{name} contains missing values.")
    series.name = name
    return series


def _common_index(nav, benchmark, risk_free, mar, align: str) -> pd.DatetimeIndex:
    indexes = [nav.index]
    if benchmark is not None:
        indexes.append(benchmark.index)
    if isinstance(risk_free, pd.Series):
        indexes.append(risk_free.index)
    if isinstance(mar, pd.Series):
        indexes.append(mar.index)
    if align == "inner":
        common = indexes[0]
        for index in indexes[1:]:
            common = common.intersection(index)
        return pd.DatetimeIndex(common).sort_values()
    common = nav.index
    for index, label in (
        (benchmark.index, "benchmark") if benchmark is not None else (None, None),
        (risk_free.index, "risk_free") if isinstance(risk_free, pd.Series) else (None, None),
        (mar.index, "mar") if isinstance(mar, pd.Series) else (None, None),
    ):
        if index is None:
            continue
        missing = common.difference(index)
        if len(missing):
            raise PerformanceEvaluationError(
                f"{label} is missing {len(missing)} NAV dates. "
                f"The first missing date is {missing.min().date()}. "
                "Pass align='inner' to evaluate the overlapping dates only."
            )
    return common


def _broadcast_rate(rate, index: pd.DatetimeIndex, name: str) -> pd.Series:
    if isinstance(rate, float):
        return pd.Series(rate, index=index, name=name)
    return rate.reindex(index)


def _to_period(
    level_on_nav: pd.Series,
    convention: str,
    day_count: float,
    name: str,
) -> pd.Series:
    if convention == "period":
        return level_on_nav.iloc[1:].rename(name)
    days = level_on_nav.index.to_series().diff().dt.days.iloc[1:].astype(float)
    annual = level_on_nav.iloc[1:]
    if (annual <= -1).any():
        raise PerformanceEvaluationError(
            f"{name} annualized rates must be greater than -100%."
        )
    if (days <= 0).any():
        raise PerformanceEvaluationError(f"{name} dates must be strictly increasing.")
    period = (1.0 + annual.to_numpy()) ** (days.to_numpy() / day_count) - 1.0
    return pd.Series(period, index=annual.index, name=name)


def _reject_nonfinite(series: pd.Series, name: str) -> None:
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise PerformanceEvaluationError(f"{name} are not finite.")


def confidence_tag(level: float) -> str:
    """Format 0.95 as '95' and 0.975 as '97.5'."""
    percent = level * 100
    text = f"{percent:.6f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    if not math.isfinite(percent):
        raise PerformanceEvaluationError(f"Invalid confidence level {level}.")
    return text
