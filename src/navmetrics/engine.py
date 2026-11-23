"""Compute the performance catalog from prepared NAV curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from navmetrics.inputs import Prepared, confidence_tag


def compute(data: Prepared) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    ctx = _Context(data)
    scalars: dict[str, float] = {}
    frames: dict[str, pd.DataFrame] = {}
    _returns(ctx, scalars, frames)
    _risk(ctx, scalars)
    _tail(ctx, scalars)
    _drawdowns(ctx, scalars, frames)
    _ratios(ctx, scalars)
    if ctx.rb is not None:
        _relative(ctx, scalars)
    _path(ctx, scalars, frames)
    return scalars, frames


@dataclass
class _Context:
    data: Prepared
    r: np.ndarray
    rf: np.ndarray
    mar: np.ndarray
    n: int
    q: float
    nav_values: np.ndarray
    dd: np.ndarray
    episodes: list[dict]
    rb: np.ndarray | None
    active: np.ndarray | None

    def __init__(self, data: Prepared) -> None:
        self.data = data
        self.r = data.returns.to_numpy(dtype=float)
        self.rf = data.risk_free.to_numpy(dtype=float)
        self.mar = data.mar.to_numpy(dtype=float)
        self.n = int(self.r.shape[0])
        self.q = float(data.periods_per_year)
        self.nav_values = data.nav.to_numpy(dtype=float)
        self.dd = _drawdown_array(self.nav_values)
        self.episodes = _episodes(data.nav, self.dd)
        self.rb = None if data.benchmark_returns is None else data.benchmark_returns.to_numpy(dtype=float)
        self.active = None if self.rb is None else self.r - self.rb


def _returns(ctx: _Context, scalars: dict[str, float], frames: dict[str, pd.DataFrame]) -> None:
    r = ctx.r
    q = ctx.q
    total = _total_return(ctx.nav_values)
    cagr = _cagr_from_values(ctx.nav_values, q)
    scalars["cumulative_return"] = total
    scalars["cagr"] = cagr
    scalars["annualized_arithmetic_return"] = float(np.mean(r) * q)
    scalars["annualized_excess_return"] = float(np.mean(r - ctx.rf) * q)
    scalars["mean_return"] = float(np.mean(r))
    scalars["median_return"] = float(np.median(r))
    positive = r[r > 0]
    negative = r[r < 0]
    scalars["mean_positive_return"] = _mean_or_nan(positive)
    scalars["mean_negative_return"] = _mean_or_nan(negative)
    scalars["win_rate"] = float(np.mean(r > 0))
    monthly = _compound(ctx.data.returns, "ME")
    quarterly = _compound(ctx.data.returns, "QE")
    yearly = _compound(ctx.data.returns, "YE")
    scalars["win_rate_monthly"] = _positive_fraction(monthly)
    scalars["win_rate_quarterly"] = _positive_fraction(quarterly)
    scalars["win_rate_yearly"] = _positive_fraction(yearly)
    payoff_den = abs(_mean_or_nan(negative)) if len(negative) else 0.0
    scalars["payoff_ratio"] = _ratio(_mean_or_nan(positive) if len(positive) else 0.0, payoff_den)
    scalars["profit_factor"] = _ratio(float(np.sum(positive)), abs(float(np.sum(negative))) if len(negative) else 0.0)
    scalars["gain_pain_ratio"] = _ratio(float(np.sum(r)), abs(float(np.sum(negative))) if len(negative) else 0.0)
    scalars["expectancy"] = float(np.mean(r))
    scalars["sqn"] = _ratio(np.sqrt(ctx.n) * float(np.mean(r)), _std(r))
    scalars["longest_win_streak"] = float(_longest(r > 0))
    scalars["longest_loss_streak"] = float(_longest(r < 0))
    scalars["best_period"] = float(np.max(r))
    scalars["worst_period"] = float(np.min(r))
    scalars["best_month"] = _extreme(monthly, np.nanmax)
    scalars["worst_month"] = _extreme(monthly, np.nanmin)
    scalars["best_quarter"] = _extreme(quarterly, np.nanmax)
    scalars["worst_quarter"] = _extreme(quarterly, np.nanmin)
    scalars["best_year"] = _extreme(yearly, np.nanmax)
    scalars["worst_year"] = _extreme(yearly, np.nanmin)
    scalars["mtd"] = _slice_return(ctx.data.nav, ctx.data.nav.index[-1].replace(day=1))
    quarter_start = _quarter_start(ctx.data.nav.index[-1])
    scalars["qtd"] = _slice_return(ctx.data.nav, quarter_start)
    scalars["ytd"] = _slice_return(ctx.data.nav, pd.Timestamp(ctx.data.nav.index[-1].year, 1, 1))
    for name, offset in (
        ("trailing_6m", pd.DateOffset(months=6)),
        ("trailing_1y", pd.DateOffset(years=1)),
        ("trailing_3y", pd.DateOffset(years=3)),
        ("trailing_5y", pd.DateOffset(years=5)),
        ("trailing_10y", pd.DateOffset(years=10)),
    ):
        scalars[name] = _trailing_cagr(ctx.data.nav, offset, q)

    frames["calendar_year_returns"] = _calendar_frame(ctx, yearly, "YE")
    frames["calendar_quarter_returns"] = _calendar_frame(ctx, quarterly, "QE")
    frames["calendar_month_returns"] = _calendar_frame(ctx, monthly, "ME")

    if ctx.rb is not None:
        strategy_growth = ctx.nav_values[-1] / ctx.nav_values[0]
        benchmark_values = ctx.data.benchmark.to_numpy(dtype=float)
        benchmark_growth = benchmark_values[-1] / benchmark_values[0]
        geometric = strategy_growth / benchmark_growth - 1.0
        scalars["geometric_excess_return"] = float(geometric)
        scalars["annualized_geometric_excess"] = float((1.0 + geometric) ** (q / ctx.n) - 1.0)
        scalars["cumulative_arithmetic_excess"] = float(np.sum(ctx.active))
        scalars["annualized_arithmetic_excess"] = float(np.mean(ctx.active) * q)


def _risk(ctx: _Context, scalars: dict[str, float]) -> None:
    r = ctx.r
    q = ctx.q
    centered = r - np.mean(r)
    scalars["annualized_volatility"] = _std(r) * np.sqrt(q)
    scalars["annualized_mean_absolute_deviation"] = float(np.mean(np.abs(centered)) * np.sqrt(q))
    scalars["downside_deviation"] = _downside_deviation(r, ctx.mar, q)
    scalars["upside_deviation"] = _upside_deviation(r, ctx.mar, q)
    scalars["semivariance"] = _partial_moment(ctx.mar - r, 2, ctx.n)
    scalars["lpm_0"] = float(np.mean(r < ctx.mar))
    scalars["lpm_1"] = _partial_moment(ctx.mar - r, 1, ctx.n)
    scalars["lpm_2"] = scalars["semivariance"]
    scalars["lpm_3"] = _partial_moment(ctx.mar - r, 3, ctx.n)
    scalars["upm_0"] = float(np.mean(r > ctx.mar))
    scalars["upm_1"] = _partial_moment(r - ctx.mar, 1, ctx.n)
    scalars["upm_2"] = _partial_moment(r - ctx.mar, 2, ctx.n)
    scalars["upm_3"] = _partial_moment(r - ctx.mar, 3, ctx.n)
    scalars["skewness"] = _skew(r)
    scalars["excess_kurtosis"] = _kurtosis(r)
    jb_stat, jb_p = _jarque_bera(r)
    scalars["jarque_bera"] = jb_stat
    scalars["jarque_bera_pvalue"] = jb_p
    scalars["autocorrelation_lag1"] = _autocorr(r, 1)
    lb_stat, lb_p = _ljung_box(r)
    scalars["ljung_box"] = lb_stat
    scalars["ljung_box_pvalue"] = lb_p
    scalars["hurst"] = _hurst(r)


def _tail(ctx: _Context, scalars: dict[str, float]) -> None:
    for level in ctx.data.confidence:
        tag = confidence_tag(level)
        var_return = _historical_var_return(ctx.r, level)
        scalars[f"historical_var_{tag}"] = -var_return
        scalars[f"gaussian_var_{tag}"] = _gaussian_var(ctx.r, level)
        scalars[f"modified_var_{tag}"] = _modified_var(ctx.r, level)
        scalars[f"cvar_{tag}"] = _cvar(ctx.r, level)
    scalars["tail_ratio"] = _tail_ratio(ctx.r)
    scalars["outlier_win_ratio"] = _outlier_ratio(ctx.r, positive=True)
    scalars["outlier_loss_ratio"] = _outlier_ratio(ctx.r, positive=False)


def _drawdowns(ctx: _Context, scalars: dict[str, float], frames: dict[str, pd.DataFrame]) -> None:
    episodes = ctx.episodes
    depths = np.array([episode["depth"] for episode in episodes], dtype=float)
    scalars["max_drawdown"] = float(np.max(ctx.dd))
    scalars["average_drawdown"] = _mean_or_nan(depths)
    scalars["median_drawdown"] = float(np.median(depths)) if len(depths) else np.nan
    durations = np.array([episode["duration_periods"] for episode in episodes], dtype=float)
    duration_days = np.array([episode["duration_days"] for episode in episodes], dtype=float)
    scalars["max_drawdown_duration_periods"] = float(np.max(durations)) if len(durations) else 0.0
    scalars["average_drawdown_duration_periods"] = _mean_or_nan(durations)
    scalars["max_drawdown_duration_days"] = float(np.max(duration_days)) if len(duration_days) else 0.0
    scalars["average_drawdown_duration_days"] = _mean_or_nan(duration_days)
    scalars["underwater_ratio"] = float(np.mean(ctx.dd > 0))
    scalars["pain_index"] = float(np.mean(ctx.dd))
    scalars["ulcer_index"] = float(np.sqrt(np.mean(ctx.dd**2)))
    deepest = _deepest(episodes)
    if deepest is None:
        scalars["max_drawdown_recovery_periods"] = 0.0
        scalars["max_drawdown_recovery_days"] = 0.0
        scalars["max_drawdown_recovered"] = 1.0
    else:
        scalars["max_drawdown_recovery_periods"] = deepest["recovery_periods"]
        scalars["max_drawdown_recovery_days"] = deepest["recovery_days"]
        scalars["max_drawdown_recovered"] = 1.0 if deepest["recovered"] else 0.0
    cagr = scalars_cagr(ctx)
    excess_ann = float(np.mean(ctx.r - ctx.rf) * ctx.q)
    scalars["recovery_factor"] = _ratio(float(ctx.nav_values[-1] / ctx.nav_values[0] - 1.0), scalars["max_drawdown"])
    scalars["calmar"] = _ratio(cagr, scalars["max_drawdown"])
    scalars["mar_ratio"] = scalars["calmar"]
    scalars["calmar_36m"] = _calmar_36m(ctx)
    worst = np.sort(depths)[-ctx.data.sterling_drawdowns :] if len(depths) else np.array([])
    scalars["sterling"] = _ratio(cagr, float(np.mean(worst)) if len(worst) else 0.0)
    burke_den = float(np.sqrt(np.sum(depths**2))) if len(depths) else 0.0
    scalars["burke"] = _ratio(excess_ann, burke_den)
    scalars["martin_ratio"] = _ratio(excess_ann, scalars["ulcer_index"])
    scalars["pain_ratio"] = _ratio(excess_ann, scalars["pain_index"])
    for level in ctx.data.confidence:
        tag = confidence_tag(level)
        cdar = _cdar(ctx.dd, level)
        scalars[f"cdar_{tag}"] = cdar
        scalars[f"cdar_ratio_{tag}"] = _ratio(excess_ann, cdar)
    frames["drawdown_episodes"] = _episode_frame(episodes)
    frames["drawdown_summary"] = _summary_frame(ctx, deepest)


def _ratios(ctx: _Context, scalars: dict[str, float]) -> None:
    q = ctx.q
    excess = ctx.r - ctx.rf
    scalars["sharpe"] = _sharpe(excess, q)
    scalars["sharpe_autocorrelation_adjusted"] = _lo_sharpe(excess, q)
    scalars["adjusted_sharpe"] = _pezier_sharpe(excess, scalars["sharpe"])
    sr_period = _sharpe(excess, q=1.0)
    sr_star = ctx.data.psr_threshold / np.sqrt(q)
    for level in ctx.data.confidence:
        tag = confidence_tag(level)
        psr, mintrl = _psr_and_mintrl(excess, sr_period, sr_star, level)
        scalars[f"probabilistic_sharpe_{tag}"] = psr
        scalars[f"minimum_track_record_length_{tag}"] = mintrl
    scalars["sortino"] = _sortino(ctx.r, ctx.mar, q)
    zeros = np.zeros_like(ctx.r)
    scalars["sortino_zero"] = _sortino(ctx.r, zeros, q)
    scalars["sortino_risk_free"] = _sortino(ctx.r, ctx.rf, q)
    scalars["omega_zero"] = _omega(ctx.r, zeros)
    scalars["omega_mar"] = _omega(ctx.r, ctx.mar)
    scalars["omega_risk_free"] = _omega(ctx.r, ctx.rf)
    scalars[f"kappa_{ctx.data.kappa_order}"] = _kappa(ctx.r, ctx.mar, ctx.data.kappa_order, q)
    for p_order, q_order in ctx.data.ft_orders:
        scalars[f"farinelli_tibiletti_{p_order}_{q_order}"] = _farinelli(ctx.r, ctx.mar, p_order, q_order)
    down_period = _downside_deviation(ctx.r, ctx.mar, q=1.0)
    scalars["upside_potential_ratio"] = _ratio(float(np.mean(np.maximum(ctx.r - ctx.mar, 0.0))), down_period)
    scalars["roy_safety_first"] = _ratio(float(np.mean(ctx.r - ctx.mar)), _std(ctx.r)) * np.sqrt(q)
    for level in ctx.data.confidence:
        tag = confidence_tag(level)
        cvar = scalars[f"cvar_{tag}"]
        scalars[f"starr_{tag}"] = _ratio(float(np.mean(excess)), cvar) * np.sqrt(q)
        scalars[f"rachev_{tag}"] = _rachev(ctx.r, level)
    scalars["common_sense_ratio"] = _product(scalars["profit_factor"], scalars["tail_ratio"])
    scalars["cpc_index"] = _product(scalars["common_sense_ratio"], scalars["win_rate"])
    variance = _std(excess) ** 2
    scalars["kelly_fraction"] = _ratio(float(np.mean(excess)), variance)


def _relative(ctx: _Context, scalars: dict[str, float]) -> None:
    assert ctx.rb is not None and ctx.active is not None
    q = ctx.q
    active = ctx.active
    scalars["tracking_error"] = _std(active) * np.sqrt(q)
    scalars["information_ratio"] = _sharpe(active, q)
    scalars["downside_information_ratio"] = _sortino(active, np.zeros_like(active), q)
    fit = _ols(ctx.r - ctx.rf, [ctx.rb - ctx.rf])
    if fit is None:
        for key in (
            "beta",
            "alpha",
            "alpha_tstat",
            "alpha_pvalue",
            "beta_tstat",
            "beta_pvalue",
            "residual_volatility",
            "appraisal_ratio",
            "r_squared",
            "treynor",
        ):
            scalars[key] = np.nan
    else:
        alpha, beta = fit.beta[0], fit.beta[1]
        scalars["beta"] = float(beta)
        scalars["alpha"] = float(alpha * q)
        scalars["alpha_tstat"] = float(fit.tstats[0])
        scalars["alpha_pvalue"] = _t_pvalue(fit.tstats[0], fit.dof)
        scalars["beta_tstat"] = float(fit.tstats[1])
        scalars["beta_pvalue"] = _t_pvalue(fit.tstats[1], fit.dof)
        resid_std = float(np.sqrt(np.sum(fit.resid**2) / fit.dof))
        scalars["residual_volatility"] = resid_std * np.sqrt(q)
        scalars["appraisal_ratio"] = _ratio(float(alpha), resid_std) * np.sqrt(q)
        scalars["r_squared"] = fit.r2
        scalars["treynor"] = _ratio(float(np.mean(ctx.r - ctx.rf) * q), float(beta))
    scalars["correlation"] = _corr(ctx.r, ctx.rb)
    scalars["annualized_covariance"] = _cov(ctx.r, ctx.rb) * q
    sharpe = scalars["sharpe"] if "sharpe" in scalars else _sharpe(ctx.r - ctx.rf, q)
    rf_ann = float(np.mean(ctx.rf) * q)
    vol_b = _std(ctx.rb) * np.sqrt(q)
    vol_p = _std(ctx.r) * np.sqrt(q)
    scalars["m2"] = rf_ann + sharpe * vol_b if np.isfinite(sharpe) and np.isfinite(vol_b) else np.nan
    scalars["m2_excess"] = scalars["m2"] - float(np.mean(ctx.rb) * q)
    levered = rf_ann + (float(np.mean(ctx.rb) * q) - rf_ann) * _ratio(vol_p, vol_b)
    scalars["gh2"] = float(np.mean(ctx.r) * q) - levered if np.isfinite(levered) else np.nan
    up = ctx.rb > ctx.data.regime_threshold
    down = ctx.rb < ctx.data.regime_threshold
    scalars["up_capture"] = _capture(ctx.r, ctx.rb, up)
    scalars["down_capture"] = _capture(ctx.r, ctx.rb, down)
    scalars["capture_ratio"] = _ratio(scalars["up_capture"], scalars["down_capture"])
    scalars["bull_beta"] = _cov_beta(ctx.r, ctx.rb, up)
    scalars["bear_beta"] = _cov_beta(ctx.r, ctx.rb, down)
    scalars["downside_beta"] = _cov_beta(ctx.r, ctx.rb, ctx.rb < ctx.mar)
    tm = _ols(ctx.r - ctx.rf, [ctx.rb - ctx.rf, (ctx.rb - ctx.rf) ** 2])
    _store_gamma(scalars, "treynor_mazuy_gamma", tm)
    hm = _ols(ctx.r - ctx.rf, [ctx.rb - ctx.rf, np.maximum(ctx.rb - ctx.rf, 0.0)])
    _store_gamma(scalars, "henriksson_merton_gamma", hm)
    scalars["coskewness"] = _comoment(ctx.r, ctx.rb, 2)
    scalars["cokurtosis"] = _comoment(ctx.r, ctx.rb, 3)
    scalars["batting_average"] = float(np.mean(ctx.r > ctx.rb))
    scalars["batting_average_up"] = _masked_mean(ctx.r > ctx.rb, up)
    scalars["batting_average_down"] = _masked_mean(ctx.r > ctx.rb, down)
    strategy_m = _compound(ctx.data.returns, "ME")
    benchmark_m = _compound(ctx.data.benchmark_returns, "ME")
    strategy_y = _compound(ctx.data.returns, "YE")
    benchmark_y = _compound(ctx.data.benchmark_returns, "YE")
    scalars["batting_average_monthly"] = _paired_batting(strategy_m, benchmark_m)
    scalars["batting_average_yearly"] = _paired_batting(strategy_y, benchmark_y)
    ahead = ctx.r > ctx.rb
    behind = ctx.r < ctx.rb
    scalars["average_excess_when_ahead"] = _mean_or_nan(active[ahead])
    scalars["average_excess_when_behind"] = _mean_or_nan(active[behind])
    scalars["active_omega"] = _omega(active, np.zeros_like(active))
    scalars["active_skewness"] = _skew(active)
    scalars["active_excess_kurtosis"] = _kurtosis(active)
    for level in ctx.data.confidence:
        tag = confidence_tag(level)
        cutoff = float(np.quantile(ctx.rb, 1.0 - level))
        mask = ctx.rb <= cutoff
        scalars[f"stress_mean_return_{tag}"] = _mean_or_nan(ctx.r[mask])
        scalars[f"stress_cumulative_return_{tag}"] = _compounded(ctx.r[mask])
        scalars[f"active_historical_var_{tag}"] = -_historical_var_return(active, level)
        scalars[f"active_cvar_{tag}"] = _cvar(active, level)
    relative_nav = _relative_nav(ctx)
    relative_values = relative_nav.to_numpy(dtype=float)
    relative_returns = relative_nav.pct_change().iloc[1:].to_numpy(dtype=float)
    relative_dd = _drawdown_array(relative_values)
    scalars["relative_cumulative_return"] = _total_return(relative_values)
    scalars["relative_cagr"] = _cagr_from_values(relative_values, q)
    scalars["relative_volatility"] = _std(relative_returns) * np.sqrt(q)
    scalars["relative_sharpe"] = _sharpe(relative_returns, q)
    scalars["relative_max_drawdown"] = float(np.max(relative_dd))
    scalars["relative_ulcer_index"] = float(np.sqrt(np.mean(relative_dd**2)))
    scalars["relative_pain_index"] = float(np.mean(relative_dd))
    scalars["relative_underwater_ratio"] = float(np.mean(relative_dd > 0))


def _path(ctx: _Context, scalars: dict[str, float], frames: dict[str, pd.DataFrame]) -> None:
    log_nav = np.log(ctx.nav_values)
    if ctx.n >= 2 and np.isfinite(log_nav).all() and np.std(log_nav) > 0:
        time = np.arange(len(log_nav), dtype=float)
        slope, _intercept, r_value, _p, se = stats.linregress(time, log_nav)
        scalars["stability"] = float(r_value**2)
        scalars["k_ratio"] = _ratio(float(slope), float(se))
    else:
        scalars["stability"] = np.nan
        scalars["k_ratio"] = np.nan
    rolling = _rolling(ctx)
    frames["rolling"] = rolling
    for window in ctx.data.windows:
        subset = rolling[rolling["window"] == window.name] if len(rolling) else rolling
        for metric in ("cagr", "volatility", "sharpe", "max_drawdown"):
            values = subset[metric].to_numpy(dtype=float) if len(subset) else np.array([])
            finite = values[np.isfinite(values)]
            scalars[f"rolling_{metric}_min_{window.name}"] = float(np.min(finite)) if len(finite) else np.nan
            scalars[f"rolling_{metric}_median_{window.name}"] = (
                float(np.median(finite)) if len(finite) else np.nan
            )
            scalars[f"rolling_{metric}_max_{window.name}"] = float(np.max(finite)) if len(finite) else np.nan


def scalars_cagr(ctx: _Context) -> float:
    return _cagr_from_values(ctx.nav_values, ctx.q)


def _drawdown_array(values: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(values)
    return 1.0 - values / peak


def _episodes(nav: pd.Series, dd: np.ndarray) -> list[dict]:
    index = nav.index
    found: list[dict] = []
    i = 0
    size = len(dd)
    while i < size:
        if dd[i] <= 0:
            i += 1
            continue
        peak_i = i - 1
        j = i
        while j < size and dd[j] > 0:
            j += 1
        segment = dd[i:j]
        trough_i = i + int(np.argmax(segment))
        recovered = j < size
        end_i = j if recovered else size - 1
        found.append(
            {
                "peak_index": peak_i,
                "trough_index": trough_i,
                "peak_date": index[peak_i],
                "trough_date": index[trough_i],
                "recovery_date": index[j] if recovered else pd.NaT,
                "depth": float(segment.max()),
                "duration_periods": int(end_i - peak_i),
                "duration_days": int((index[end_i] - index[peak_i]).days),
                "recovery_periods": float(j - trough_i) if recovered else np.nan,
                "recovery_days": float((index[j] - index[trough_i]).days) if recovered else np.nan,
                "recovered": recovered,
            }
        )
        i = j
    return found


def _deepest(episodes: list[dict]) -> dict | None:
    if not episodes:
        return None
    return max(episodes, key=lambda episode: (episode["depth"], -episode["peak_index"]))


def _episode_frame(episodes: list[dict]) -> pd.DataFrame:
    columns = [
        "peak_date",
        "trough_date",
        "recovery_date",
        "depth",
        "duration_periods",
        "duration_days",
        "recovery_periods",
        "recovery_days",
        "recovered",
    ]
    if not episodes:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([{column: episode[column] for column in columns} for episode in episodes])


def _summary_frame(ctx: _Context, deepest: dict | None) -> pd.DataFrame:
    if deepest is None:
        row = {
            "peak_date": ctx.data.nav.index[0],
            "trough_date": ctx.data.nav.index[0],
            "recovery_date": ctx.data.nav.index[0],
            "depth": 0.0,
            "recovered": True,
        }
    else:
        row = {
            "peak_date": deepest["peak_date"],
            "trough_date": deepest["trough_date"],
            "recovery_date": deepest["recovery_date"],
            "depth": deepest["depth"],
            "recovered": deepest["recovered"],
        }
    return pd.DataFrame([row])


def _total_return(values: np.ndarray) -> float:
    return float(values[-1] / values[0] - 1.0)


def _cagr_from_values(values: np.ndarray, q: float) -> float:
    n_returns = len(values) - 1
    growth = values[-1] / values[0]
    if n_returns <= 0 or growth <= 0:
        return np.nan
    return float(growth ** (q / n_returns) - 1.0)


def _std(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def _mean_or_nan(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def _ratio(numerator: float, denominator: float) -> float:
    if numerator is None or denominator is None:
        return np.nan
    num = float(numerator)
    den = float(denominator)
    if not np.isfinite(num) or not np.isfinite(den):
        return np.nan
    if den == 0.0:
        if num > 0:
            return np.inf
        if num < 0:
            return -np.inf
        return np.nan
    return num / den


def _product(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return np.nan
    return float(left) * float(right)


def _longest(mask: np.ndarray) -> int:
    longest = current = 0
    for flag in mask:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _compound(returns: pd.Series, freq: str) -> pd.Series:
    return (1.0 + returns).resample(freq).prod() - 1.0


def _positive_fraction(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return float(np.mean(clean.to_numpy(dtype=float) > 0))


def _extreme(returns: pd.Series, reducer) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(reducer(clean.to_numpy(dtype=float)))


def _slice_return(nav: pd.Series, start: pd.Timestamp) -> float:
    prior = nav.loc[nav.index < start]
    if prior.empty:
        inside = nav.loc[nav.index >= start]
        if len(inside) < 2:
            return np.nan
        return float(inside.iloc[-1] / inside.iloc[0] - 1.0)
    return float(nav.iloc[-1] / prior.iloc[-1] - 1.0)


def _quarter_start(stamp: pd.Timestamp) -> pd.Timestamp:
    month = ((stamp.month - 1) // 3) * 3 + 1
    return pd.Timestamp(stamp.year, month, 1)


def _trailing_cagr(nav: pd.Series, offset: pd.DateOffset, q: float) -> float:
    boundary = nav.index[-1] - offset
    prior = nav.loc[nav.index <= boundary]
    if prior.empty:
        return np.nan
    base = float(prior.iloc[-1])
    n_returns = int((nav.index > prior.index[-1]).sum())
    if n_returns < 1 or base <= 0:
        return np.nan
    return float((nav.iloc[-1] / base) ** (q / n_returns) - 1.0)


def _calendar_frame(ctx: _Context, strategy: pd.Series, freq: str) -> pd.DataFrame:
    frame = pd.DataFrame({"strategy": strategy})
    if ctx.data.benchmark_returns is not None:
        benchmark = _compound(ctx.data.benchmark_returns, freq)
        aligned = frame.join(benchmark.rename("benchmark"), how="left")
        aligned["geometric_excess"] = (1.0 + aligned["strategy"]) / (1.0 + aligned["benchmark"]) - 1.0
        return aligned
    return frame


def _partial_moment(signed_gap: np.ndarray, order: int, n: int) -> float:
    shortfall = np.maximum(signed_gap, 0.0)
    if n < 2:
        return np.nan
    return float(np.sum(shortfall**order) / (n - 1))


def _downside_deviation(returns: np.ndarray, threshold: np.ndarray, q: float) -> float:
    if len(returns) < 2:
        return np.nan
    gap = np.minimum(returns - threshold, 0.0)
    return float(np.sqrt(np.sum(gap**2) / (len(returns) - 1)) * np.sqrt(q))


def _upside_deviation(returns: np.ndarray, threshold: np.ndarray, q: float) -> float:
    if len(returns) < 2:
        return np.nan
    gap = np.maximum(returns - threshold, 0.0)
    return float(np.sqrt(np.sum(gap**2) / (len(returns) - 1)) * np.sqrt(q))


def _skew(values: np.ndarray) -> float:
    if len(values) < 3 or _std(values) == 0:
        return np.nan
    return float(stats.skew(values, bias=False))


def _kurtosis(values: np.ndarray) -> float:
    if len(values) < 4 or _std(values) == 0:
        return np.nan
    return float(stats.kurtosis(values, fisher=True, bias=False))


def _jarque_bera(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 4 or not np.isfinite(_std(values)) or _std(values) == 0:
        return np.nan, np.nan
    stat, pvalue = stats.jarque_bera(values)
    return float(stat), float(pvalue)


def _autocorr(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag + 1:
        return np.nan
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return np.nan
    return float(np.dot(centered[:-lag], centered[lag:]) / denominator)


def _ljung_box(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    lags = min(10, n // 2 - 1)
    if lags < 1:
        return np.nan, np.nan
    rhos = np.array([_autocorr(values, lag) for lag in range(1, lags + 1)], dtype=float)
    if not np.isfinite(rhos).all():
        return np.nan, np.nan
    weights = n - np.arange(1, lags + 1)
    statistic = float(n * (n + 2) * np.sum(rhos**2 / weights))
    return statistic, float(stats.chi2.sf(statistic, lags))


def _hurst(values: np.ndarray) -> float:
    n = len(values)
    if n < 40:
        return np.nan
    max_size = n // 2
    sizes = np.unique(np.floor(np.logspace(np.log10(8), np.log10(max_size), num=12)).astype(int))
    sizes = sizes[(sizes >= 8) & (sizes <= max_size)]
    used: list[int] = []
    rs_values: list[float] = []
    for size in sizes:
        n_blocks = n // int(size)
        block_rs: list[float] = []
        for block_index in range(n_blocks):
            block = values[block_index * size : (block_index + 1) * size]
            block = block - np.mean(block)
            scale = float(np.std(block, ddof=1))
            if scale <= 0:
                continue
            cumulative = np.cumsum(block)
            block_rs.append(float((np.max(cumulative) - np.min(cumulative)) / scale))
        if block_rs:
            used.append(int(size))
            rs_values.append(float(np.mean(block_rs)))
    if len(used) < 3:
        return np.nan
    slope = np.polyfit(np.log(used), np.log(rs_values), 1)[0]
    return float(slope)


def _historical_var_return(values: np.ndarray, level: float) -> float:
    return float(np.quantile(values, 1.0 - level))


def _gaussian_var(values: np.ndarray, level: float) -> float:
    scale = _std(values)
    if not np.isfinite(scale):
        return np.nan
    z = float(stats.norm.ppf(level))
    return float(-np.mean(values) + z * scale)


def _modified_var(values: np.ndarray, level: float) -> float:
    if len(values) < 8:
        return np.nan
    scale = _std(values)
    skew = _skew(values)
    kurt = _kurtosis(values)
    if not np.isfinite(scale) or scale == 0 or not np.isfinite(skew) or not np.isfinite(kurt):
        return np.nan
    z = float(stats.norm.ppf(1.0 - level))
    z_cf = (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * kurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )
    return float(-(np.mean(values) + scale * z_cf))


def _cvar(values: np.ndarray, level: float) -> float:
    cutoff = np.quantile(values, 1.0 - level)
    tail = values[values <= cutoff]
    if len(tail) == 0:
        return np.nan
    return float(-np.mean(tail))


def _tail_ratio(values: np.ndarray) -> float:
    upper = float(np.quantile(values, 0.95))
    lower = float(np.quantile(values, 0.05))
    return _ratio(abs(upper), abs(lower))


def _outlier_ratio(values: np.ndarray, positive: bool) -> float:
    scale = _std(values)
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    center = float(np.mean(values))
    subset = values[values > 0] if positive else values[values < 0]
    if len(subset) == 0:
        return np.nan
    if positive:
        return float(np.mean(subset > center + 2.0 * scale))
    return float(np.mean(subset < center - 2.0 * scale))


def _cdar(drawdown: np.ndarray, level: float) -> float:
    cutoff = float(np.quantile(drawdown, level))
    tail = drawdown[drawdown >= cutoff]
    if len(tail) == 0:
        return np.nan
    return float(np.mean(tail))


def _calmar_36m(ctx: _Context) -> float:
    end = ctx.data.nav.index[-1]
    start = end - pd.DateOffset(months=36)
    if ctx.data.nav.index[0] > start:
        return np.nan
    window = ctx.data.nav.loc[ctx.data.nav.index >= start]
    if len(window) < 2:
        return np.nan
    values = window.to_numpy(dtype=float)
    return _ratio(_cagr_from_values(values, ctx.q), float(np.max(_drawdown_array(values))))


def _sharpe(excess: np.ndarray, q: float) -> float:
    return _ratio(float(np.mean(excess)), _std(excess)) * np.sqrt(q)


def _lo_sharpe(excess: np.ndarray, q: float) -> float:
    base = _ratio(float(np.mean(excess)), _std(excess))
    if not np.isfinite(base):
        return np.nan
    max_lag = min(int(np.floor(q)) - 1, len(excess) - 2)
    if max_lag < 1:
        return float(base * np.sqrt(q))
    adjustment = 0.0
    for lag in range(1, max_lag + 1):
        rho = _autocorr(excess, lag)
        if not np.isfinite(rho):
            return np.nan
        adjustment += (1.0 - lag / q) * rho
    denominator = 1.0 + 2.0 * adjustment
    if denominator <= 0:
        return np.nan
    return float(base * np.sqrt(q / denominator))


def _pezier_sharpe(excess: np.ndarray, sharpe: float) -> float:
    if len(excess) < 8 or not np.isfinite(sharpe):
        return np.nan
    skew = _skew(excess)
    kurt = _kurtosis(excess)
    if not np.isfinite(skew) or not np.isfinite(kurt):
        return np.nan
    return float(sharpe * (1.0 + skew / 6.0 * sharpe - kurt / 24.0 * sharpe**2))


def _psr_and_mintrl(
    excess: np.ndarray,
    sr_period: float,
    sr_star: float,
    level: float,
) -> tuple[float, float]:
    if len(excess) < 8 or not np.isfinite(sr_period):
        return np.nan, np.nan
    skew = float(stats.skew(excess, bias=True))
    kurt = float(stats.kurtosis(excess, fisher=False, bias=True))
    inside = 1.0 - skew * sr_period + (kurt - 1.0) / 4.0 * sr_period**2
    if inside <= 0 or not np.isfinite(inside):
        return np.nan, np.nan
    n = len(excess)
    psr = float(stats.norm.cdf((sr_period - sr_star) * np.sqrt(n - 1) / np.sqrt(inside)))
    gap = sr_period - sr_star
    if gap <= 0:
        return psr, np.inf
    z = float(stats.norm.ppf(level))
    mintrl = 1.0 + inside * (z / gap) ** 2
    return psr, float(mintrl)


def _sortino(returns: np.ndarray, threshold: np.ndarray, q: float) -> float:
    return _ratio(float(np.mean(returns - threshold)), _downside_deviation(returns, threshold, q=1.0)) * np.sqrt(q)


def _omega(returns: np.ndarray, threshold: np.ndarray) -> float:
    gap = returns - threshold
    gains = float(np.sum(np.maximum(gap, 0.0)))
    losses = float(np.sum(np.maximum(-gap, 0.0)))
    return _ratio(gains, losses)


def _kappa(returns: np.ndarray, threshold: np.ndarray, order: int, q: float) -> float:
    moment = _partial_moment(threshold - returns, order, len(returns))
    if not np.isfinite(moment) or moment < 0:
        return np.nan
    scale = moment ** (1.0 / order)
    return _ratio(float(np.mean(returns - threshold)), scale) * q ** (1.0 - 1.0 / order)


def _farinelli(returns: np.ndarray, threshold: np.ndarray, p_order: int, q_order: int) -> float:
    upper = _partial_moment(returns - threshold, p_order, len(returns))
    lower = _partial_moment(threshold - returns, q_order, len(returns))
    if not np.isfinite(upper) or not np.isfinite(lower):
        return np.nan
    return _ratio(upper ** (1.0 / p_order), lower ** (1.0 / q_order))


def _rachev(values: np.ndarray, level: float) -> float:
    upper_cut = float(np.quantile(values, level))
    lower_cut = float(np.quantile(values, 1.0 - level))
    gains = values[values >= upper_cut]
    losses = values[values <= lower_cut]
    if len(gains) == 0 or len(losses) == 0:
        return np.nan
    return _ratio(float(np.mean(gains)), abs(float(np.mean(losses))))


@dataclass
class _Fit:
    beta: np.ndarray
    tstats: np.ndarray
    resid: np.ndarray
    dof: int
    r2: float


def _ols(y: np.ndarray, columns: list[np.ndarray]) -> _Fit | None:
    n = len(y)
    x = np.column_stack([np.ones(n), *columns])
    k = x.shape[1]
    if n <= k + 1:
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.linalg.matrix_rank(x) < k:
        return None
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    dof = n - k
    sse = float(np.dot(resid, resid))
    centered = y - np.mean(y)
    sst = float(np.dot(centered, centered))
    s2 = sse / dof
    try:
        xtx_inv = np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * s2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = np.where(se > 0, beta / se, np.where(np.abs(beta) <= 1e-12, np.nan, np.inf))
    r2 = np.nan if sst == 0 else float(1.0 - sse / sst)
    return _Fit(beta=beta, tstats=tstats, resid=resid, dof=dof, r2=r2)


def _t_pvalue(t_stat: float, dof: int) -> float:
    if not np.isfinite(t_stat) or dof <= 0:
        return np.nan
    return float(2.0 * stats.t.sf(abs(t_stat), dof))


def _store_gamma(scalars: dict[str, float], name: str, fit: _Fit | None) -> None:
    if fit is None:
        scalars[name] = np.nan
        scalars[f"{name}_tstat"] = np.nan
        scalars[f"{name}_pvalue"] = np.nan
        return
    scalars[name] = float(fit.beta[-1])
    scalars[f"{name}_tstat"] = float(fit.tstats[-1])
    scalars[f"{name}_pvalue"] = _t_pvalue(fit.tstats[-1], fit.dof)


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if _std(left) == 0 or _std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _cov(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return np.nan
    return float(np.cov(left, right, ddof=1)[0, 1])


def _capture(strategy: np.ndarray, benchmark: np.ndarray, mask: np.ndarray) -> float:
    if int(np.sum(mask)) < 1:
        return np.nan
    strategy_growth = float(np.prod(1.0 + strategy[mask]) - 1.0)
    benchmark_growth = float(np.prod(1.0 + benchmark[mask]) - 1.0)
    return _ratio(strategy_growth, benchmark_growth)


def _cov_beta(strategy: np.ndarray, benchmark: np.ndarray, mask: np.ndarray) -> float:
    if int(np.sum(mask)) < 3:
        return np.nan
    variance = _std(benchmark[mask]) ** 2
    return _ratio(_cov(strategy[mask], benchmark[mask]), variance)


def _comoment(left: np.ndarray, right: np.ndarray, power: int) -> float:
    if len(left) < power + 2:
        return np.nan
    dx = left - np.mean(left)
    dy = right - np.mean(right)
    sx = float(np.sqrt(np.mean(dx**2)))
    sy = float(np.sqrt(np.mean(dy**2)))
    if sx == 0 or sy == 0:
        return np.nan
    moment = float(np.mean(dx * dy**power))
    return moment / (sx * sy**power)


def _masked_mean(flags: np.ndarray, mask: np.ndarray) -> float:
    if int(np.sum(mask)) < 1:
        return np.nan
    return float(np.mean(flags[mask]))


def _paired_batting(strategy: pd.Series, benchmark: pd.Series) -> float:
    joined = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    if joined.empty:
        return np.nan
    return float(np.mean(joined["strategy"].to_numpy() > joined["benchmark"].to_numpy()))


def _compounded(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    return float(np.prod(1.0 + values) - 1.0)


def _relative_nav(ctx: _Context) -> pd.Series:
    strategy = ctx.data.nav / ctx.data.nav.iloc[0]
    benchmark = ctx.data.benchmark / ctx.data.benchmark.iloc[0]
    return strategy / benchmark


def _rolling(ctx: _Context) -> pd.DataFrame:
    dates = ctx.data.nav.index
    values = ctx.nav_values
    rows: list[dict] = []
    date_values = dates.to_numpy()
    for window in ctx.data.windows:
        if window.periods is not None:
            anchors = np.arange(len(values)) - window.periods
        else:
            boundaries = dates - window.offset
            anchors = dates.searchsorted(pd.DatetimeIndex(boundaries), side="right") - 1
        for i, anchor in enumerate(anchors):
            anchor = int(anchor)
            if anchor < 0 or anchor >= i:
                continue
            n_returns = i - anchor
            if n_returns < ctx.data.min_rolling_periods:
                continue
            segment = values[anchor : i + 1]
            window_returns = ctx.r[anchor:i]
            window_rf = ctx.rf[anchor:i]
            excess = window_returns - window_rf
            peak = np.maximum.accumulate(segment)
            drawdown = float(np.max(1.0 - segment / peak))
            rows.append(
                {
                    "date": date_values[i],
                    "window": window.name,
                    "cagr": _cagr_from_values(segment, ctx.q),
                    "volatility": _std(window_returns) * np.sqrt(ctx.q),
                    "sharpe": _sharpe(excess, ctx.q),
                    "max_drawdown": drawdown,
                }
            )
    columns = ["date", "window", "cagr", "volatility", "sharpe", "max_drawdown"]
    frame = pd.DataFrame(rows, columns=columns)
    if len(frame):
        frame["date"] = pd.to_datetime(frame["date"])
    return frame
