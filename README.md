# navmetrics

Performance statistics computed from a strategy net-asset-value (NAV) curve and an optional benchmark NAV curve.

Python 3.9 or newer. Dependencies are NumPy, pandas 2.2 or newer, and SciPy.

```bash
pip install git+https://github.com/fyzhang2026/navmetrics.git
```

## Tutorial

`nav` and `benchmark` are NAV levels, not returns. Each series has a `DatetimeIndex` and strictly positive values. `evaluate` turns them into simple returns with `V_t / V_{t-1} - 1`.

```python
import pandas as pd
from navmetrics import evaluate, list_metrics

nav = pd.Series(
    [100.0, 102.0, 101.0, 105.0],
    index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    name="strategy",
)
benchmark = pd.Series([100.0, 101.0, 100.0, 103.0], index=nav.index, name="benchmark")

report = evaluate(
    nav,
    benchmark,
    risk_free=0.02,
    mar=0.0,
    confidence=(0.95, 0.99),
    periods_per_year=252,
    rate_convention="annualized",
)
print(report.scalars["sharpe"])
print(report.scalars["historical_var_95"])
print(report.frames["calendar_year_returns"])
```

`report.scalars` is a float `Series`. `report.frames` holds tables. `report.meta` records the parameters used. `report["sharpe"]` reads one scalar.

Omit `metrics` to return every statistic the inputs can support. Pass names to keep a subset:

```python
subset = evaluate(
    nav,
    benchmark,
    metrics=["sharpe", "max_drawdown", "historical_var", "relative"],
)
catalog = list_metrics()  # id, group, kind, requires_benchmark, description
```

A one-column `DataFrame` is accepted anywhere a `Series` is accepted. Time zones are converted to UTC and then dropped. Dates must be unique. At least two NAV dates are required.

Both NAV series must be performance indexes in the same currency, with distributions reinvested. External cash flows such as subscriptions and redemptions must already have been removed. Otherwise the percentage changes mix funding with investment results.

## Inputs

### Time series

These four inputs share one set of dates. A series must have a `DatetimeIndex`, numeric values, no duplicates, and no missing values.

| Argument | Form | Meaning |
| --- | --- | --- |
| `nav` | required `Series` | Strategy NAV level. Every value must be finite and strictly positive. |
| `benchmark` | `Series` or omitted | Benchmark NAV level, same form as `nav`. Relative statistics are skipped when it is omitted. Requesting one explicitly raises an error. |
| `risk_free` | float or `Series`, default `0` | Risk-free rate. A float is written onto every NAV date. A series must cover every NAV date. |
| `mar` | float or `Series`, default `0` | Minimum acceptable return. Same forms as `risk_free`. Returns below this path are the only ones that count as downside. |

`rate_convention` sets the units of `risk_free` and `mar`:

- `"period"` (default). The value on date `t` is the simple return over the interval that ends on `t`. The first NAV date has no return, so its rate is ignored.
- `"annualized"`. The value on date `t` is an annualized quote. It is converted on the interval ending at `t` with `(1 + y) ** (days / day_count) - 1`. `days` is the actual number of calendar days since the previous NAV date. Annualized quotes must be greater than `-1`.

`align` controls dates that do not match:

- `"strict"` (default). `benchmark`, `risk_free`, and `mar` must contain every `nav` date. Extra dates on those inputs are ignored.
- `"inner"`. Only dates present on every supplied series are kept.

### Scalars and short settings

These are not time series.

| Argument | Default | Form and meaning |
| --- | --- | --- |
| `confidence` | `0.95` | One tail probability, or several, each strictly between 0.5 and 1. Example: `(0.95, 0.99)`. This chooses the tail used by VaR, CVaR, and the other confidence-dependent statistics. It is not a curve. |
| `periods_per_year` | `252` | Positive annualization factor `q`. Use 252 for daily returns, 52 for weekly, and 12 for monthly. CAGR is `(V_T / V_0) ** (q / n) - 1`. Volatility and Sharpe multiply by `sqrt(q)`. Arithmetic means multiply by `q`. |
| `rate_convention` | `"period"` | `"period"` or `"annualized"`, as defined above. |
| `day_count` | `365.25` | Positive denominator used only when `rate_convention` is `"annualized"`. |
| `regime_threshold` | `0` | A benchmark simple return above this number is an up-market period. A return below it is a down-market period. Equals are in neither group. Capture, bull beta, and bear beta use this split. |
| `psr_threshold` | `0` | Annualized Sharpe hurdle for the probabilistic Sharpe ratio. It is divided by `sqrt(q)` before the Bailey and López de Prado formula. The minimum track-record length is a count of return observations. |
| `sterling_drawdowns` | `5` | Integer of 1 or more. Sterling uses the mean depth of this many worst drawdown episodes. Fewer episodes are all used. |
| `kappa_order` | `3` | Integer of 1 or more. Selects the Kappa ratio. Order 2 matches Sortino. The result id is `kappa_{order}`. |
| `min_rolling_periods` | `10` | Integer of 2 or more. A rolling window with fewer returns is skipped. |
| `farinelli_tibiletti_orders` | `((1, 1), (2, 1))` | Pairs of integers `(p, q)`, each at least 1. Result ids are `farinelli_tibiletti_{p}_{q}`. |
| `rolling_windows` | `("6M", "1Y", "3Y")` | Calendar windows such as `"6M"` or `"3Y"`, or an integer number of return periods such as `126`. Duplicate names are ignored. |
| `metrics` | `None` | `None` returns the full report. A string or a sequence selects ids, stems, or groups. `"all"` is the full report. An empty sequence raises an error. |
| `align` | `"strict"` | `"strict"` or `"inner"`. |

## How to choose outputs

Three ways to name a selection:

- An exact id, such as `"sharpe"` or `"historical_var_95"`.
- A stem. `"historical_var"` expands to one row per confidence level, for example `historical_var_95` and `historical_var_99`. `"kappa"` expands to `kappa_3` when `kappa_order` is 3. `"rolling"` returns the rolling table and the min, median, and max summaries.
- A group: `return`, `risk`, `tail`, `drawdown`, `risk_adjusted`, `relative`, `path`.

`list_metrics()` is the machine-readable catalog. The tables below match that catalog.

Confidence-dependent ids use the percent tag of the level: `0.95` becomes `_95`, and `0.975` becomes `_97.5`. Losses for VaR and CVaR are positive numbers when the tail return is negative.

Sample standard deviation divides by `n - 1`. Drawdown depth is positive: `1 - V_t / running_peak`. `calmar` and `mar_ratio` are the same full-sample number. `calmar_36m` is missing when the sample is shorter than 36 months. `sortino` uses `mar`. `sortino_zero` uses 0. `sortino_risk_free` uses the risk-free path.

### `return`

| Id | Result |
| --- | --- |
| `cumulative_return` | Simple total return, `V_T / V_0 - 1`. |
| `cagr` | Compound annual growth, `(V_T / V_0) ** (q / n) - 1`. |
| `annualized_arithmetic_return` | Mean simple return multiplied by `q`. |
| `annualized_excess_return` | Mean of `r - r_f`, multiplied by `q`. |
| `mean_return` | Mean per-period simple return. |
| `median_return` | Median per-period simple return. |
| `mean_positive_return` | Mean of strictly positive simple returns. |
| `mean_negative_return` | Mean of strictly negative simple returns. |
| `win_rate` | Fraction of periods with a positive simple return. |
| `win_rate_monthly` | Fraction of calendar months with a positive compounded return. |
| `win_rate_quarterly` | Fraction of calendar quarters with a positive compounded return. |
| `win_rate_yearly` | Fraction of calendar years with a positive compounded return. |
| `payoff_ratio` | Mean positive return divided by the absolute mean negative return. |
| `profit_factor` | Sum of positive returns divided by the absolute sum of negative returns. |
| `gain_pain_ratio` | Sum of all returns divided by the absolute sum of negative returns. |
| `expectancy` | Per-period expected simple return. Equal to `mean_return`. |
| `sqn` | System quality number, `sqrt(n) * mean(r) / sample std(r)`. |
| `longest_win_streak` | Longest run of strictly positive returns. |
| `longest_loss_streak` | Longest run of strictly negative returns. |
| `best_period`, `worst_period` | Largest and smallest per-period simple return. |
| `best_month`, `worst_month` | Largest and smallest compounded calendar-month return. |
| `best_quarter`, `worst_quarter` | Largest and smallest compounded calendar-quarter return. |
| `best_year`, `worst_year` | Largest and smallest compounded calendar-year return. |
| `mtd`, `qtd`, `ytd` | Total return from the last NAV before the current month, quarter, or year. |
| `trailing_6m`, `trailing_1y`, `trailing_3y`, `trailing_5y`, `trailing_10y` | CAGR from the last NAV on or before that horizon. Missing when the sample does not reach the horizon. |
| `geometric_excess_return` | `(1 + R_p) / (1 + R_b) - 1`. Requires a benchmark. |
| `annualized_geometric_excess` | Geometric excess annualized with exponent `q / n`. Requires a benchmark. |
| `cumulative_arithmetic_excess` | Sum of `r_p - r_b`. Requires a benchmark. |
| `annualized_arithmetic_excess` | Mean of `r_p - r_b`, multiplied by `q`. Requires a benchmark. |
| `calendar_year_returns` | Table of compounded yearly returns. Benchmark and geometric-excess columns are added when a benchmark is supplied. |
| `calendar_quarter_returns` | Table of compounded quarterly returns. |
| `calendar_month_returns` | Table of compounded monthly returns. |

### `risk`

| Id | Result |
| --- | --- |
| `annualized_volatility` | Sample standard deviation of simple returns, multiplied by `sqrt(q)`. |
| `annualized_mean_absolute_deviation` | Mean absolute deviation from the mean return, multiplied by `sqrt(q)`. |
| `downside_deviation` | Root-mean-square of `min(r - MAR, 0)`, using divisor `n - 1`, multiplied by `sqrt(q)`. |
| `upside_deviation` | Same construction for `max(r - MAR, 0)`. |
| `semivariance` | Sum of squared `min(r - MAR, 0)` divided by `n - 1`. Same value as `lpm_2`. |
| `lpm_0` | Fraction of periods with `r < MAR`. Divisor `n`. |
| `lpm_1`, `lpm_2`, `lpm_3` | Lower partial moments around MAR. Divisor `n - 1`. |
| `upm_0` | Fraction of periods with `r > MAR`. Divisor `n`. |
| `upm_1`, `upm_2`, `upm_3` | Upper partial moments around MAR. Divisor `n - 1`. |
| `skewness` | Unbiased sample skewness of simple returns. |
| `excess_kurtosis` | Unbiased sample excess kurtosis of simple returns. |
| `jarque_bera`, `jarque_bera_pvalue` | Jarque-Bera statistic and its p-value. |
| `autocorrelation_lag1` | Lag-1 autocorrelation of simple returns. |
| `ljung_box`, `ljung_box_pvalue` | Ljung-Box Q statistic with `min(10, n // 2 - 1)` lags, and its chi-square p-value. |
| `hurst` | Rescaled-range Hurst exponent of simple returns. Requires at least 40 returns. |

### `tail`

`historical_var`, `gaussian_var`, `modified_var`, and `cvar` are one scalar per confidence level.

| Id | Result |
| --- | --- |
| `historical_var` | Historical Value at Risk as a positive loss: minus the empirical quantile at `1 - alpha`. |
| `gaussian_var` | Gaussian VaR as a positive loss: `-mean + z_alpha * sample std`. |
| `modified_var` | Cornish-Fisher modified VaR as a positive loss. Requires at least 8 returns. |
| `cvar` | Historical expected shortfall as a positive loss: minus the mean of returns at or below the VaR quantile. |
| `tail_ratio` | Absolute ratio of the 95% return quantile to the 5% return quantile. |
| `outlier_win_ratio` | Share of winning periods whose return exceeds `mean + 2` sample standard deviations. |
| `outlier_loss_ratio` | Share of losing periods whose return is below `mean - 2` sample standard deviations. |

### `drawdown`

| Id | Result |
| --- | --- |
| `max_drawdown` | Largest peak-to-trough loss as a positive fraction of the peak NAV. |
| `average_drawdown` | Mean of the maximum depth of each underwater episode. |
| `median_drawdown` | Median of those episode depths. |
| `max_drawdown_duration_periods` | Longest episode from peak to recovery, in return periods. |
| `average_drawdown_duration_periods` | Mean episode length in return periods. An open episode runs to the sample end. |
| `max_drawdown_duration_days` | Longest episode in calendar days. |
| `average_drawdown_duration_days` | Mean episode length in calendar days. |
| `max_drawdown_recovery_periods` | Return periods from the deepest trough back to its prior peak. Missing if still unrecovered. |
| `max_drawdown_recovery_days` | Calendar days for that recovery. Missing if still unrecovered. |
| `max_drawdown_recovered` | `1` if the deepest drawdown recovers inside the sample, otherwise `0`. |
| `underwater_ratio` | Fraction of NAV dates strictly below the running peak. |
| `pain_index` | Mean drawdown depth across all NAV dates, including zeros. |
| `ulcer_index` | Root-mean-square drawdown depth across all NAV dates. |
| `recovery_factor` | Cumulative return divided by maximum drawdown. |
| `calmar` | CAGR divided by maximum drawdown. |
| `calmar_36m` | Calmar ratio on the trailing 36 months. Missing when the sample is shorter. |
| `mar_ratio` | Same value as `calmar`. |
| `sterling` | CAGR divided by the mean depth of the `sterling_drawdowns` worst episodes. |
| `burke` | Annualized arithmetic excess over `r_f`, divided by the square root of the sum of squared episode depths. |
| `martin_ratio` | Annualized arithmetic excess over `r_f`, divided by the Ulcer index. |
| `pain_ratio` | Annualized arithmetic excess over `r_f`, divided by the Pain index. |
| `cdar` | Conditional drawdown at risk: mean of drawdown depths at or above the alpha quantile. One value per confidence level. |
| `cdar_ratio` | Annualized arithmetic excess over `r_f`, divided by CDaR. One value per confidence level. |
| `drawdown_episodes` | Table with one row per underwater episode. |
| `drawdown_summary` | Peak, trough, recovery, and depth of the deepest episode. |

### `risk_adjusted`

| Id | Result |
| --- | --- |
| `sharpe` | `mean(r - r_f) / sample std(r - r_f) * sqrt(q)`. |
| `sharpe_autocorrelation_adjusted` | Lo (2002) Sharpe ratio. Lags run through `min(q - 1, n - 2)`. |
| `adjusted_sharpe` | Pézier-White adjustment of the reported Sharpe using skewness and excess kurtosis. |
| `probabilistic_sharpe` | Probability that the true Sharpe exceeds `psr_threshold`. One value per confidence level. |
| `minimum_track_record_length` | Observations required for that probability to reach the confidence level. |
| `sortino` | Sortino ratio with the MAR path as the threshold. |
| `sortino_zero` | Sortino ratio with threshold 0. |
| `sortino_risk_free` | Sortino ratio with the risk-free path as the threshold. |
| `omega_zero` | Omega ratio with threshold 0. |
| `omega_mar` | Omega ratio with the MAR path as the threshold. |
| `omega_risk_free` | Omega ratio with the risk-free path as the threshold. |
| `kappa` | Kappa ratio of order `kappa_order`. Annualized by `q ** (1 - 1 / order)`. |
| `farinelli_tibiletti` | `UPM_p ** (1/p) / LPM_q ** (1/q)` around MAR, for each requested order pair. |
| `upside_potential_ratio` | Mean of `max(r - MAR, 0)` divided by the per-period downside deviation. |
| `roy_safety_first` | `mean(r - MAR) / sample std(r) * sqrt(q)`. |
| `starr` | `mean(r - r_f) / historical CVaR * sqrt(q)`. One value per confidence level. |
| `rachev` | Mean upper-tail return divided by the absolute mean lower-tail return, both tails at the same confidence level. |
| `common_sense_ratio` | Profit factor multiplied by the tail ratio. |
| `cpc_index` | Profit factor multiplied by the win rate and the tail ratio. |
| `kelly_fraction` | `mean(r - r_f) / sample variance(r - r_f)`. A leverage estimate. |

### `relative`

Every statistic in this group requires `benchmark`.

| Id | Result |
| --- | --- |
| `tracking_error` | Sample standard deviation of `r_p - r_b`, multiplied by `sqrt(q)`. |
| `information_ratio` | Mean active return divided by its sample standard deviation, multiplied by `sqrt(q)`. |
| `downside_information_ratio` | Mean active return divided by the downside deviation of active returns below zero, multiplied by `sqrt(q)`. |
| `beta` | Slope from regressing excess strategy returns on excess benchmark returns, with an intercept. |
| `alpha` | Jensen alpha: that intercept multiplied by `q`. |
| `alpha_tstat`, `alpha_pvalue` | t statistic and two-sided p-value of the per-period alpha. |
| `beta_tstat`, `beta_pvalue` | t statistic and two-sided p-value of beta. |
| `residual_volatility` | Residual standard deviation of the Jensen regression, multiplied by `sqrt(q)`. |
| `appraisal_ratio` | Per-period alpha divided by per-period residual standard deviation, multiplied by `sqrt(q)`. |
| `r_squared` | R-squared of the Jensen regression. |
| `correlation` | Pearson correlation of simple strategy and benchmark returns. |
| `annualized_covariance` | Sample covariance of simple returns, multiplied by `q`. |
| `treynor` | Annualized arithmetic excess over `r_f`, divided by Jensen beta. |
| `m2` | M-squared: annualized `r_f` plus the strategy Sharpe multiplied by benchmark volatility. |
| `m2_excess` | M-squared minus the benchmark annualized arithmetic return. |
| `gh2` | Graham-Harvey GH2. Strategy annualized arithmetic return minus the benchmark levered to the strategy volatility. |
| `up_capture` | Compounded strategy return divided by compounded benchmark return in up-market periods. |
| `down_capture` | Same ratio in down-market periods. |
| `capture_ratio` | Up capture divided by down capture. |
| `bull_beta` | Covariance beta in up-market periods. |
| `bear_beta` | Covariance beta in down-market periods. |
| `downside_beta` | Covariance beta in periods with `r_b` below MAR. |
| `treynor_mazuy_gamma` | Coefficient on squared excess benchmark return, with its t statistic and p-value. |
| `henriksson_merton_gamma` | Coefficient on `max(r_b - r_f, 0)`, with its t statistic and p-value. |
| `coskewness` | Standardized co-skewness of strategy and benchmark simple returns. |
| `cokurtosis` | Standardized co-kurtosis of strategy and benchmark simple returns. |
| `batting_average` | Fraction of periods with `r_p > r_b`. |
| `batting_average_monthly` | Fraction of months whose compounded strategy return exceeds the benchmark. |
| `batting_average_yearly` | Fraction of years whose compounded strategy return exceeds the benchmark. |
| `batting_average_up`, `batting_average_down` | Batting average inside up-market and down-market periods. |
| `average_excess_when_ahead` | Mean of `r_p - r_b` when the strategy is ahead. |
| `average_excess_when_behind` | Mean of `r_p - r_b` when the strategy is behind. |
| `active_omega` | Omega ratio of `r_p - r_b` with threshold 0. |
| `active_skewness` | Unbiased skewness of active returns. |
| `active_excess_kurtosis` | Unbiased excess kurtosis of active returns. |
| `stress_mean_return` | Mean strategy return when the benchmark is at or below its `1 - alpha` quantile. One value per confidence level. |
| `stress_cumulative_return` | Compounded strategy return over those stress periods. |
| `active_historical_var` | Historical VaR of active returns, as a positive loss. |
| `active_cvar` | Historical expected shortfall of active returns, as a positive loss. |
| `relative_cumulative_return` | Total return of `(V_p / V_p0) / (V_b / V_b0)`. |
| `relative_cagr` | CAGR of that relative wealth curve. |
| `relative_volatility` | Annualized volatility of relative-wealth returns. |
| `relative_sharpe` | Sharpe ratio of relative-wealth returns with a risk-free rate of zero. |
| `relative_max_drawdown` | Maximum drawdown of the relative wealth curve. |
| `relative_ulcer_index` | Ulcer index of the relative wealth curve. |
| `relative_pain_index` | Pain index of the relative wealth curve. |
| `relative_underwater_ratio` | Underwater ratio of the relative wealth curve. |

The timing coefficients also appear as `treynor_mazuy_gamma_tstat`, `treynor_mazuy_gamma_pvalue`, `henriksson_merton_gamma_tstat`, and `henriksson_merton_gamma_pvalue`.

### `path`

| Id | Result |
| --- | --- |
| `stability` | R-squared of log NAV regressed on a linear time index. |
| `k_ratio` | t statistic of that slope. |
| `rolling` | Table with date, window, CAGR, volatility, Sharpe, and maximum drawdown. |
| `rolling_{cagr\|volatility\|sharpe\|max_drawdown}_{min\|median\|max}_{window}` | Summary of each rolling series. The default windows produce suffixes `_6M`, `_1Y`, and `_3Y`. |

## What this package does not measure

Turnover, fee attribution, and holdings-based attribution need weights or trades. Put trading costs into the NAV before calling `evaluate` when the curve should be net of costs.

## License

MIT. See [LICENSE](LICENSE).
