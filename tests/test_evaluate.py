from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from navmetrics import (
    PerformanceEvaluationError,
    evaluate,
    list_metrics,
)
from navmetrics.catalog import METRICS


def _nav_from_returns(returns: np.ndarray, start: str = "2020-01-01") -> pd.Series:
    index = pd.bdate_range(start, periods=len(returns) + 1)
    values = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + np.asarray(returns, dtype=float))))
    return pd.Series(values, index=index)


def test_sharpe_sortino_and_var_match_definitions():
    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005])
    nav = _nav_from_returns(returns)
    report = evaluate(nav, risk_free=0.0, mar=0.0, periods_per_year=252, metrics=["sharpe", "sortino", "gaussian_var", "historical_var"])

    expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    downside = np.sqrt(np.sum(np.minimum(returns, 0.0) ** 2) / (len(returns) - 1))
    expected_sortino = returns.mean() / downside * np.sqrt(252)
    expected_gaussian = -returns.mean() + stats.norm.ppf(0.95) * returns.std(ddof=1)
    expected_historical = -np.quantile(returns, 0.05)

    assert report.scalars["sharpe"] == pytest.approx(expected_sharpe)
    assert report.scalars["sortino"] == pytest.approx(expected_sortino)
    assert report.scalars["gaussian_var_95"] == pytest.approx(expected_gaussian)
    assert report.scalars["historical_var_95"] == pytest.approx(expected_historical)


def test_drawdown_and_calmar():
    index = pd.bdate_range("2020-01-01", periods=4)
    nav = pd.Series([100.0, 150.0, 120.0, 180.0], index=index)
    report = evaluate(nav, periods_per_year=252, metrics=["max_drawdown", "calmar", "drawdown_summary"])
    returns = nav.pct_change().dropna().to_numpy()
    cagr = (180.0 / 100.0) ** (252 / len(returns)) - 1.0

    assert report.scalars["max_drawdown"] == pytest.approx(0.2)
    assert report.scalars["calmar"] == pytest.approx(cagr / 0.2)
    assert report.frames["drawdown_summary"].iloc[0]["depth"] == pytest.approx(0.2)
    assert "sharpe" not in report.scalars


def test_benchmark_regression_and_geometric_excess():
    rng = np.random.default_rng(7)
    benchmark_returns = rng.normal(0.0004, 0.01, 300)
    strategy_returns = 0.0001 + 1.5 * benchmark_returns
    strategy = _nav_from_returns(strategy_returns)
    benchmark = _nav_from_returns(benchmark_returns)
    report = evaluate(
        strategy,
        benchmark,
        risk_free=0.0,
        metrics=["beta", "alpha", "geometric_excess_return", "tracking_error"],
    )
    assert report.scalars["beta"] == pytest.approx(1.5, rel=1e-8, abs=1e-8)
    assert report.scalars["alpha"] == pytest.approx(0.0001 * 252, rel=1e-6, abs=1e-8)
    growth_s = strategy.iloc[-1] / strategy.iloc[0]
    growth_b = benchmark.iloc[-1] / benchmark.iloc[0]
    assert report.scalars["geometric_excess_return"] == pytest.approx(growth_s / growth_b - 1.0)


def test_identical_curves_have_zero_active_return():
    returns = np.array([0.01, -0.02, 0.005, 0.004, -0.001, 0.008])
    nav = _nav_from_returns(returns)
    report = evaluate(nav, nav, metrics=["beta", "alpha", "tracking_error", "information_ratio"])
    assert report.scalars["beta"] == pytest.approx(1.0)
    assert report.scalars["alpha"] == pytest.approx(0.0, abs=1e-10)
    assert report.scalars["tracking_error"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(report.scalars["information_ratio"])


def test_annualized_risk_free_uses_actual_day_count():
    index = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"])
    nav = pd.Series([100.0, 101.0, 102.0, 100.0], index=index)
    report = evaluate(
        nav,
        risk_free=0.04,
        rate_convention="annualized",
        day_count=365.25,
        periods_per_year=252,
        metrics=["sharpe"],
    )
    day_count = np.array([3.0, 1.0, 1.0])
    rf = (1.04) ** (day_count / 365.25) - 1.0
    returns = nav.pct_change().dropna().to_numpy()
    excess = returns - rf
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    assert report.scalars["sharpe"] == pytest.approx(expected)
    assert report.meta["rate_convention"] == "annualized"


def test_metric_selection_groups_and_unknown_names():
    returns = np.linspace(-0.01, 0.02, 30)
    nav = _nav_from_returns(returns)
    benchmark = _nav_from_returns(returns * 0.5)
    selected = evaluate(nav, benchmark, confidence=(0.95, 0.99), metrics=["historical_var", "path"])
    assert set(selected.scalars.index) >= {"historical_var_95", "historical_var_99", "stability"}
    assert "rolling" in selected.frames
    assert "sharpe" not in selected.scalars

    with pytest.raises(PerformanceEvaluationError):
        evaluate(nav, metrics=["not_a_real_metric"])
    with pytest.raises(PerformanceEvaluationError):
        evaluate(nav, metrics=["beta"])


def test_strict_alignment_and_inner_join():
    nav = _nav_from_returns(np.array([0.01, 0.02, -0.01, 0.005]))
    benchmark = nav.iloc[:-1]
    with pytest.raises(PerformanceEvaluationError, match="missing"):
        evaluate(nav, benchmark, metrics=["cumulative_return"])
    joined = evaluate(nav, benchmark, align="inner", metrics=["cumulative_return"])
    assert joined.meta["n_periods"] == 3


def test_full_catalog_is_populated_on_a_long_sample():
    rng = np.random.default_rng(11)
    n = 252 * 6
    market = rng.normal(0.0003, 0.01, n)
    strategy = 0.00005 + 0.8 * market + rng.normal(0, 0.002, n)
    nav = _nav_from_returns(strategy, start="2018-01-01")
    benchmark = _nav_from_returns(market, start="2018-01-01")
    report = evaluate(
        nav,
        benchmark,
        risk_free=0.02,
        mar=0.0,
        rate_convention="annualized",
        confidence=(0.95, 0.99),
        periods_per_year=252,
    )
    fixed = [metric.id for metric in METRICS if metric.kind == "scalar"]
    assert set(fixed) <= set(report.scalars.index)
    for stem in ("historical_var", "cvar", "probabilistic_sharpe", "kappa", "farinelli_tibiletti"):
        assert any(str(key).startswith(stem + "_") for key in report.scalars.index)
    assert "rolling_cagr_min_6M" in report.scalars
    assert np.isfinite(report.scalars["sharpe"])
    assert np.isfinite(report.scalars["hurst"])
    assert np.isfinite(report.scalars["calmar_36m"])
    assert np.isfinite(report.scalars["trailing_5y"])
    assert np.isnan(report.scalars["trailing_10y"])
    assert {"strategy", "benchmark", "geometric_excess"} <= set(report.frames["calendar_year_returns"].columns)
    catalog = list_metrics()
    assert {"sharpe", "rolling", "calendar_year_returns"} <= set(catalog["id"])


def test_optional_parameters_change_named_output():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0002, 0.01, 80)
    nav = _nav_from_returns(returns)
    benchmark = _nav_from_returns(rng.normal(0.0002, 0.01, 80))
    report = evaluate(
        nav,
        benchmark,
        kappa_order=4,
        farinelli_tibiletti_orders=((1, 2),),
        rolling_windows=("3M",),
        psr_threshold=0.5,
        regime_threshold=0.001,
        sterling_drawdowns=2,
        metrics=["kappa", "farinelli_tibiletti", "rolling", "up_capture"],
    )
    assert "kappa_4" in report.scalars
    assert "farinelli_tibiletti_1_2" in report.scalars
    assert "rolling_sharpe_median_3M" in report.scalars
    assert set(report.frames["rolling"]["window"].unique()) == {"3M"}
    assert np.isfinite(report.scalars["up_capture"]) or np.isnan(report.scalars["up_capture"])
