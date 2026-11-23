"""Portfolio performance statistics from NAV curves.

Conventions
-----------
Simple return on date ``t`` is ``V_t / V_{t-1} - 1``. ``q`` is
``periods_per_year``. Volatility, Sharpe, and the other annualized ratios use
the sample standard deviation (divide by ``n - 1``) and a factor of ``sqrt(q)``.
CAGR is ``(V_T / V_0) ** (q / n) - 1``.

Drawdown depth is positive: ``1 - V_t / max_{s<=t} V_s``.

The risk-free and minimum-acceptable-return inputs are either per-period simple
returns or annualized rates. Annualized rates are converted on each interval
with ``(1 + y) ** (days / day_count) - 1``.

Tail losses are reported as positive numbers. Confidence levels stay scalars.
``psr_threshold`` is an annualized Sharpe hurdle.
"""

from navmetrics.catalog import list_metrics
from navmetrics.errors import PerformanceEvaluationError
from navmetrics.evaluate import evaluate
from navmetrics.report import PerformanceReport

__all__ = [
    "PerformanceEvaluationError",
    "PerformanceReport",
    "evaluate",
    "list_metrics",
]

__version__ = "0.1.0"
