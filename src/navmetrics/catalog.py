"""Metric identifiers, groups, and descriptions.

``list_metrics`` is the authoritative catalog. Numeric definitions live in
``engine.py`` and are restated in the package README.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Metric:
    id: str
    group: str
    kind: str
    requires_benchmark: bool
    description: str


def _s(
    metric_id: str,
    group: str,
    description: str,
    *,
    benchmark: bool = False,
    kind: str = "scalar",
) -> Metric:
    return Metric(metric_id, group, kind, benchmark, description)


# kind:
#   scalar                  one float
#   scalar_per_confidence   one float per confidence level, id suffix _95
#   scalar_per_order        kappa and Farinelli-Tibiletti
#   frame                   a DataFrame on the report

METRICS: tuple[Metric, ...] = (
    _s("cumulative_return", "return", "Simple total return, V_T / V_0 - 1."),
    _s("cagr", "return", "Compound annual growth, (V_T / V_0) ** (q / n) - 1."),
    _s(
        "annualized_arithmetic_return",
        "return",
        "Arithmetic mean of simple returns multiplied by q.",
    ),
    _s(
        "annualized_excess_return",
        "return",
        "Arithmetic mean of r - r_f, multiplied by q.",
    ),
    _s("mean_return", "return", "Arithmetic mean of per-period simple returns."),
    _s("median_return", "return", "Median per-period simple return."),
    _s("mean_positive_return", "return", "Mean of strictly positive simple returns."),
    _s("mean_negative_return", "return", "Mean of strictly negative simple returns."),
    _s("win_rate", "return", "Fraction of periods with a positive simple return."),
    _s("win_rate_monthly", "return", "Fraction of calendar months with a positive compounded return."),
    _s("win_rate_quarterly", "return", "Fraction of calendar quarters with a positive compounded return."),
    _s("win_rate_yearly", "return", "Fraction of calendar years with a positive compounded return."),
    _s("payoff_ratio", "return", "Mean positive return divided by the absolute mean negative return."),
    _s("profit_factor", "return", "Sum of positive returns divided by the absolute sum of negative returns."),
    _s("gain_pain_ratio", "return", "Sum of all returns divided by the absolute sum of negative returns."),
    _s("expectancy", "return", "Per-period expected simple return. Equal to mean_return."),
    _s("sqn", "return", "System quality number, sqrt(n) * mean(r) / sample std(r)."),
    _s("longest_win_streak", "return", "Longest run of strictly positive returns."),
    _s("longest_loss_streak", "return", "Longest run of strictly negative returns."),
    _s("best_period", "return", "Largest per-period simple return."),
    _s("worst_period", "return", "Smallest per-period simple return."),
    _s("best_month", "return", "Largest compounded calendar-month return."),
    _s("worst_month", "return", "Smallest compounded calendar-month return."),
    _s("best_quarter", "return", "Largest compounded calendar-quarter return."),
    _s("worst_quarter", "return", "Smallest compounded calendar-quarter return."),
    _s("best_year", "return", "Largest compounded calendar-year return."),
    _s("worst_year", "return", "Smallest compounded calendar-year return."),
    _s("mtd", "return", "Total return from the last NAV before this month to the last NAV."),
    _s("qtd", "return", "Total return from the last NAV before this quarter to the last NAV."),
    _s("ytd", "return", "Total return from the last NAV before this year to the last NAV."),
    _s("trailing_6m", "return", "CAGR from the last NAV on or before six months ago."),
    _s("trailing_1y", "return", "CAGR from the last NAV on or before one year ago."),
    _s("trailing_3y", "return", "CAGR from the last NAV on or before three years ago."),
    _s("trailing_5y", "return", "CAGR from the last NAV on or before five years ago."),
    _s("trailing_10y", "return", "CAGR from the last NAV on or before ten years ago."),
    _s(
        "geometric_excess_return",
        "return",
        "Compounded outperformance, (1 + R_p) / (1 + R_b) - 1.",
        benchmark=True,
    ),
    _s(
        "annualized_geometric_excess",
        "return",
        "Geometric excess return annualized with exponent q / n.",
        benchmark=True,
    ),
    _s(
        "cumulative_arithmetic_excess",
        "return",
        "Sum of per-period arithmetic active returns r_p - r_b.",
        benchmark=True,
    ),
    _s(
        "annualized_arithmetic_excess",
        "return",
        "Mean of r_p - r_b, multiplied by q.",
        benchmark=True,
    ),
    _s("annualized_volatility", "risk", "Sample standard deviation of simple returns multiplied by sqrt(q)."),
    _s(
        "annualized_mean_absolute_deviation",
        "risk",
        "Mean absolute deviation from the mean return, multiplied by sqrt(q).",
    ),
    _s(
        "downside_deviation",
        "risk",
        "Sample root-mean-square of min(r - MAR, 0), multiplied by sqrt(q).",
    ),
    _s(
        "upside_deviation",
        "risk",
        "Sample root-mean-square of max(r - MAR, 0), multiplied by sqrt(q).",
    ),
    _s(
        "semivariance",
        "risk",
        "Sum of squared min(r - MAR, 0) divided by n - 1. Not annualized.",
    ),
    _s("lpm_0", "risk", "Fraction of periods with r < MAR. Denominator is n."),
    _s("lpm_1", "risk", "Lower partial moment of order 1 around MAR, denominator n - 1."),
    _s("lpm_2", "risk", "Lower partial moment of order 2 around MAR, denominator n - 1."),
    _s("lpm_3", "risk", "Lower partial moment of order 3 around MAR, denominator n - 1."),
    _s("upm_0", "risk", "Fraction of periods with r > MAR. Denominator is n."),
    _s("upm_1", "risk", "Upper partial moment of order 1 around MAR, denominator n - 1."),
    _s("upm_2", "risk", "Upper partial moment of order 2 around MAR, denominator n - 1."),
    _s("upm_3", "risk", "Upper partial moment of order 3 around MAR, denominator n - 1."),
    _s("skewness", "risk", "Unbiased sample skewness of simple returns."),
    _s("excess_kurtosis", "risk", "Unbiased sample excess kurtosis of simple returns."),
    _s("jarque_bera", "risk", "Jarque-Bera statistic from scipy.stats.jarque_bera."),
    _s("jarque_bera_pvalue", "risk", "Two-sided p-value of the Jarque-Bera statistic."),
    _s("autocorrelation_lag1", "risk", "Lag-1 autocorrelation of simple returns."),
    _s("ljung_box", "risk", "Ljung-Box Q statistic, lags = min(10, n // 2 - 1)."),
    _s("ljung_box_pvalue", "risk", "Chi-square p-value of the Ljung-Box statistic."),
    _s("hurst", "risk", "Rescaled-range Hurst exponent of simple returns."),
    _s(
        "historical_var",
        "tail",
        "Historical Value at Risk as a positive loss: minus the empirical quantile at 1 - alpha.",
        kind="scalar_per_confidence",
    ),
    _s(
        "gaussian_var",
        "tail",
        "Gaussian VaR as a positive loss: -mean + z_alpha * sample std.",
        kind="scalar_per_confidence",
    ),
    _s(
        "modified_var",
        "tail",
        "Cornish-Fisher modified VaR as a positive loss. Requires at least 8 returns.",
        kind="scalar_per_confidence",
    ),
    _s(
        "cvar",
        "tail",
        "Historical expected shortfall as a positive loss: minus the mean of returns at or below the VaR quantile.",
        kind="scalar_per_confidence",
    ),
    _s("tail_ratio", "tail", "Absolute ratio of the 95% return quantile to the 5% return quantile."),
    _s(
        "outlier_win_ratio",
        "tail",
        "Share of winning periods whose return exceeds mean + 2 sample standard deviations.",
    ),
    _s(
        "outlier_loss_ratio",
        "tail",
        "Share of losing periods whose return is below mean - 2 sample standard deviations.",
    ),
    _s("max_drawdown", "drawdown", "Largest peak-to-trough loss as a positive fraction of the peak NAV."),
    _s("average_drawdown", "drawdown", "Mean of the maximum depth of each drawdown episode."),
    _s("median_drawdown", "drawdown", "Median of the maximum depth of each drawdown episode."),
    _s("max_drawdown_duration_periods", "drawdown", "Longest drawdown measured in return periods, peak to recovery."),
    _s("average_drawdown_duration_periods", "drawdown", "Mean drawdown length in return periods, peak to recovery or sample end."),
    _s("max_drawdown_duration_days", "drawdown", "Longest drawdown measured in calendar days."),
    _s("average_drawdown_duration_days", "drawdown", "Mean drawdown length in calendar days."),
    _s(
        "max_drawdown_recovery_periods",
        "drawdown",
        "Return periods from the deepest trough back to its prior peak. NaN if still unrecovered.",
    ),
    _s(
        "max_drawdown_recovery_days",
        "drawdown",
        "Calendar days from the deepest trough back to its prior peak. NaN if still unrecovered.",
    ),
    _s("max_drawdown_recovered", "drawdown", "1 if the deepest drawdown recovered inside the sample, otherwise 0."),
    _s("underwater_ratio", "drawdown", "Fraction of NAV dates strictly below the running peak."),
    _s("pain_index", "drawdown", "Mean drawdown depth across all NAV dates, including zeros."),
    _s("ulcer_index", "drawdown", "Root-mean-square drawdown depth across all NAV dates."),
    _s("recovery_factor", "drawdown", "Cumulative return divided by maximum drawdown."),
    _s("calmar", "drawdown", "CAGR divided by maximum drawdown."),
    _s(
        "calmar_36m",
        "drawdown",
        "Calmar ratio on the trailing 36 months. NaN when the sample is shorter than 36 months.",
    ),
    _s("mar_ratio", "drawdown", "Full-sample CAGR divided by maximum drawdown. Same value as calmar."),
    _s(
        "sterling",
        "drawdown",
        "CAGR divided by the mean depth of the N worst drawdown episodes.",
    ),
    _s(
        "burke",
        "drawdown",
        "Annualized arithmetic excess over r_f, divided by the square root of the sum of squared episode depths.",
    ),
    _s("martin_ratio", "drawdown", "Annualized arithmetic excess over r_f, divided by the Ulcer index."),
    _s("pain_ratio", "drawdown", "Annualized arithmetic excess over r_f, divided by the Pain index."),
    _s(
        "cdar",
        "drawdown",
        "Conditional drawdown at risk: mean of drawdown depths at or above the alpha quantile of the drawdown series.",
        kind="scalar_per_confidence",
    ),
    _s(
        "cdar_ratio",
        "drawdown",
        "Annualized arithmetic excess over r_f, divided by CDaR.",
        kind="scalar_per_confidence",
    ),
    _s("sharpe", "risk_adjusted", "Mean excess over r_f divided by its sample std, multiplied by sqrt(q)."),
    _s(
        "sharpe_autocorrelation_adjusted",
        "risk_adjusted",
        "Lo (2002) Sharpe ratio. The annualization term uses autocorrelations through lag min(q - 1, n - 2).",
    ),
    _s(
        "adjusted_sharpe",
        "risk_adjusted",
        "Pézier-White Sharpe ratio adjusted with sample skewness and excess kurtosis.",
    ),
    _s(
        "probabilistic_sharpe",
        "risk_adjusted",
        "Bailey and López de Prado probabilistic Sharpe ratio. The hurdle is an annualized Sharpe.",
        kind="scalar_per_confidence",
    ),
    _s(
        "minimum_track_record_length",
        "risk_adjusted",
        "Minimum number of return observations for the probabilistic Sharpe ratio to reach this confidence level.",
        kind="scalar_per_confidence",
    ),
    _s("sortino", "risk_adjusted", "Sortino ratio using the MAR series as the minimum acceptable return."),
    _s("sortino_zero", "risk_adjusted", "Sortino ratio using a minimum acceptable return of zero."),
    _s("sortino_risk_free", "risk_adjusted", "Sortino ratio using the risk-free series as the minimum acceptable return."),
    _s("omega_zero", "risk_adjusted", "Omega ratio with threshold 0."),
    _s("omega_mar", "risk_adjusted", "Omega ratio with the MAR series as the threshold."),
    _s("omega_risk_free", "risk_adjusted", "Omega ratio with the risk-free series as the threshold."),
    _s(
        "kappa",
        "risk_adjusted",
        "Kappa ratio of the requested order. Annualized by q ** (1 - 1 / order) so that order 2 matches Sortino.",
        kind="scalar_per_order",
    ),
    _s(
        "farinelli_tibiletti",
        "risk_adjusted",
        "Farinelli-Tibiletti ratio UPM_p^(1/p) / LPM_q^(1/q) around MAR.",
        kind="scalar_per_order",
    ),
    _s(
        "upside_potential_ratio",
        "risk_adjusted",
        "Mean of max(r - MAR, 0) divided by the per-period downside deviation. Not annualized.",
    ),
    _s(
        "roy_safety_first",
        "risk_adjusted",
        "Roy safety-first ratio, mean(r - MAR) / sample std(r) * sqrt(q).",
    ),
    _s(
        "starr",
        "risk_adjusted",
        "STARR ratio, mean(r - r_f) / historical CVaR * sqrt(q).",
        kind="scalar_per_confidence",
    ),
    _s(
        "rachev",
        "risk_adjusted",
        "Rachev ratio of the mean upper-tail return to the absolute mean lower-tail return, both tails at alpha.",
        kind="scalar_per_confidence",
    ),
    _s("common_sense_ratio", "risk_adjusted", "Profit factor multiplied by the tail ratio."),
    _s("cpc_index", "risk_adjusted", "Profit factor multiplied by the win rate and the tail ratio."),
    _s(
        "kelly_fraction",
        "risk_adjusted",
        "Continuous Kelly fraction mean(r - r_f) / sample variance(r - r_f). A leverage, not an annualized return.",
    ),
    _s("tracking_error", "relative", "Sample standard deviation of r_p - r_b, multiplied by sqrt(q).", benchmark=True),
    _s(
        "information_ratio",
        "relative",
        "Mean active return divided by its sample std, multiplied by sqrt(q).",
        benchmark=True,
    ),
    _s(
        "downside_information_ratio",
        "relative",
        "Mean active return divided by the downside deviation of active returns below zero, times sqrt(q).",
        benchmark=True,
    ),
    _s(
        "beta",
        "relative",
        "Slope of excess strategy returns on excess benchmark returns, with an intercept.",
        benchmark=True,
    ),
    _s("alpha", "relative", "Jensen alpha: regression intercept multiplied by q.", benchmark=True),
    _s("alpha_tstat", "relative", "t statistic of the per-period Jensen alpha.", benchmark=True),
    _s("alpha_pvalue", "relative", "Two-sided p-value of the Jensen alpha.", benchmark=True),
    _s("beta_tstat", "relative", "t statistic of beta.", benchmark=True),
    _s("beta_pvalue", "relative", "Two-sided p-value of beta.", benchmark=True),
    _s(
        "residual_volatility",
        "relative",
        "Residual standard deviation of the Jensen regression, multiplied by sqrt(q).",
        benchmark=True,
    ),
    _s(
        "appraisal_ratio",
        "relative",
        "Per-period alpha divided by per-period residual standard deviation, multiplied by sqrt(q).",
        benchmark=True,
    ),
    _s("r_squared", "relative", "R-squared of the Jensen excess-return regression.", benchmark=True),
    _s("correlation", "relative", "Pearson correlation of simple strategy and benchmark returns.", benchmark=True),
    _s(
        "annualized_covariance",
        "relative",
        "Sample covariance of simple returns multiplied by q.",
        benchmark=True,
    ),
    _s("treynor", "relative", "Annualized arithmetic excess over r_f, divided by Jensen beta.", benchmark=True),
    _s(
        "m2",
        "relative",
        "M-squared: annualized r_f plus the strategy Sharpe times benchmark volatility.",
        benchmark=True,
    ),
    _s(
        "m2_excess",
        "relative",
        "M-squared minus the benchmark annualized arithmetic return.",
        benchmark=True,
    ),
    _s(
        "gh2",
        "relative",
        "Graham-Harvey GH2: strategy annualized arithmetic return minus the benchmark levered to the strategy volatility.",
        benchmark=True,
    ),
    _s(
        "up_capture",
        "relative",
        "Compounded strategy return divided by compounded benchmark return in periods with r_b above the regime threshold.",
        benchmark=True,
    ),
    _s(
        "down_capture",
        "relative",
        "Compounded strategy return divided by compounded benchmark return in periods with r_b below the regime threshold.",
        benchmark=True,
    ),
    _s("capture_ratio", "relative", "Up capture divided by down capture.", benchmark=True),
    _s("bull_beta", "relative", "Covariance beta in periods with r_b above the regime threshold.", benchmark=True),
    _s("bear_beta", "relative", "Covariance beta in periods with r_b below the regime threshold.", benchmark=True),
    _s("downside_beta", "relative", "Covariance beta in periods with r_b below MAR.", benchmark=True),
    _s("treynor_mazuy_gamma", "relative", "Coefficient on squared excess benchmark return.", benchmark=True),
    _s("treynor_mazuy_gamma_tstat", "relative", "t statistic of the Treynor-Mazuy timing coefficient.", benchmark=True),
    _s("treynor_mazuy_gamma_pvalue", "relative", "Two-sided p-value of the Treynor-Mazuy timing coefficient.", benchmark=True),
    _s(
        "henriksson_merton_gamma",
        "relative",
        "Coefficient on max(r_b - r_f, 0) in the Henriksson-Merton regression.",
        benchmark=True,
    ),
    _s("henriksson_merton_gamma_tstat", "relative", "t statistic of the Henriksson-Merton timing coefficient.", benchmark=True),
    _s("henriksson_merton_gamma_pvalue", "relative", "Two-sided p-value of the Henriksson-Merton timing coefficient.", benchmark=True),
    _s("coskewness", "relative", "Standardized co-skewness of strategy and benchmark simple returns.", benchmark=True),
    _s("cokurtosis", "relative", "Standardized co-kurtosis of strategy and benchmark simple returns.", benchmark=True),
    _s("batting_average", "relative", "Fraction of periods with r_p > r_b.", benchmark=True),
    _s("batting_average_monthly", "relative", "Fraction of months whose compounded strategy return exceeds the benchmark.", benchmark=True),
    _s("batting_average_yearly", "relative", "Fraction of years whose compounded strategy return exceeds the benchmark.", benchmark=True),
    _s("batting_average_up", "relative", "Batting average inside up-market periods.", benchmark=True),
    _s("batting_average_down", "relative", "Batting average inside down-market periods.", benchmark=True),
    _s("average_excess_when_ahead", "relative", "Mean of r_p - r_b in periods where r_p > r_b.", benchmark=True),
    _s("average_excess_when_behind", "relative", "Mean of r_p - r_b in periods where r_p < r_b.", benchmark=True),
    _s("active_omega", "relative", "Omega ratio of active returns r_p - r_b with threshold 0.", benchmark=True),
    _s("active_skewness", "relative", "Unbiased skewness of active returns.", benchmark=True),
    _s("active_excess_kurtosis", "relative", "Unbiased excess kurtosis of active returns.", benchmark=True),
    _s(
        "stress_mean_return",
        "relative",
        "Mean strategy return in periods where the benchmark is at or below its 1 - alpha quantile.",
        benchmark=True,
        kind="scalar_per_confidence",
    ),
    _s(
        "stress_cumulative_return",
        "relative",
        "Compounded strategy return over those same stress periods.",
        benchmark=True,
        kind="scalar_per_confidence",
    ),
    _s(
        "active_historical_var",
        "relative",
        "Historical VaR of active returns, as a positive loss.",
        benchmark=True,
        kind="scalar_per_confidence",
    ),
    _s(
        "active_cvar",
        "relative",
        "Historical expected shortfall of active returns, as a positive loss.",
        benchmark=True,
        kind="scalar_per_confidence",
    ),
    _s(
        "relative_cumulative_return",
        "relative",
        "Total return of the relative wealth curve (V_p / V_p0) / (V_b / V_b0).",
        benchmark=True,
    ),
    _s("relative_cagr", "relative", "CAGR of the relative wealth curve.", benchmark=True),
    _s("relative_volatility", "relative", "Annualized volatility of relative-wealth returns.", benchmark=True),
    _s(
        "relative_sharpe",
        "relative",
        "Sharpe ratio of relative-wealth returns using a risk-free rate of zero.",
        benchmark=True,
    ),
    _s("relative_max_drawdown", "relative", "Maximum drawdown of the relative wealth curve.", benchmark=True),
    _s("relative_ulcer_index", "relative", "Ulcer index of the relative wealth curve.", benchmark=True),
    _s("relative_pain_index", "relative", "Pain index of the relative wealth curve.", benchmark=True),
    _s("relative_underwater_ratio", "relative", "Underwater ratio of the relative wealth curve.", benchmark=True),
    _s("stability", "path", "R-squared of log NAV regressed on a linear time index."),
    _s("k_ratio", "path", "t statistic of the slope of log NAV on a linear time index."),
    _s(
        "rolling",
        "path",
        "Rolling CAGR, volatility, Sharpe, and maximum drawdown. Summaries are rolling_{metric}_{min|median|max}_{window}.",
        kind="frame",
    ),
    _s(
        "calendar_year_returns",
        "return",
        "Compounded return of each calendar year. Benchmark and geometric excess columns are included when a benchmark is supplied.",
        kind="frame",
    ),
    _s("calendar_quarter_returns", "return", "Compounded return of each calendar quarter.", kind="frame"),
    _s("calendar_month_returns", "return", "Compounded return of each calendar month.", kind="frame"),
    _s("drawdown_episodes", "drawdown", "One row per underwater episode.", kind="frame"),
    _s("drawdown_summary", "drawdown", "Peak, trough, and recovery of the deepest drawdown.", kind="frame"),
)


GROUPS: tuple[str, ...] = (
    "return",
    "risk",
    "tail",
    "drawdown",
    "risk_adjusted",
    "relative",
    "path",
)

_BY_ID = {metric.id: metric for metric in METRICS}


def get_metric(metric_id: str) -> Metric | None:
    return _BY_ID.get(metric_id)


def stem_of(metric_id: str) -> str | None:
    """Longest catalog id that is a prefix of an expanded result id."""
    matches = [
        metric.id
        for metric in METRICS
        if metric_id == metric.id or metric_id.startswith(metric.id + "_")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def list_metrics() -> pd.DataFrame:
    """Return the metric catalog as a dataframe.

    Expanded ids such as ``historical_var_95`` are not listed separately.
    Request the stem ``historical_var`` and the report contains one row per
    confidence level.
    """
    rows = [
        {
            "id": metric.id,
            "group": metric.group,
            "kind": metric.kind,
            "requires_benchmark": metric.requires_benchmark,
            "description": metric.description,
        }
        for metric in METRICS
    ]
    return pd.DataFrame(rows)
