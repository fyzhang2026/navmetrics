# navmetrics

从策略净值和基准净值计算业绩指标。

## 安装

需要 Python 3.9 或更高版本。

```bash
pip install git+https://github.com/fyzhang2026/navmetrics.git
```

## 使用

净值序列要有日期索引，数值严格为正。它应是业绩净值：与基准同一种货币、含分红复权，并且已经去掉申购赎回。

```python
import pandas as pd
from navmetrics import evaluate, list_metrics

nav = pd.Series(
    [100, 102, 101, 105],
    index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
)
benchmark = pd.Series([100, 101, 100, 103], index=nav.index)

report = evaluate(
    nav,
    benchmark,
    risk_free=0.02,
    rate_convention="annualized",
    periods_per_year=252,
)
print(report.scalars["sharpe"])
print(report.scalars["max_drawdown"])
```

`metrics` 不传就输出全部指标。也可以只选一部分：

```python
subset = evaluate(nav, benchmark, metrics=["sharpe", "historical_var", "relative"])
print(list_metrics())
```

单个指标用名字，例如 `"sharpe"`。`"historical_var"` 会按置信水平展开。分组名有 `return`、`risk`、`tail`、`drawdown`、`risk_adjusted`、`relative`、`path`。

标量在 `report.scalars`。日历收益、回撤过程和滚动路径在 `report.frames`。
