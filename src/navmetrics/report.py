"""Container returned by :func:`navmetrics.evaluate`."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


class PerformanceReport:
    """Selected performance statistics.

    ``scalars`` is a pandas Series of single-number results, in catalog order.
    ``frames`` holds tables such as calendar returns, drawdown episodes, and
    rolling paths. ``meta`` records the parameters that produced the report.
    """

    def __init__(
        self,
        scalars: pd.Series,
        frames: Mapping[str, pd.DataFrame],
        meta: Mapping[str, object],
    ) -> None:
        self.scalars = scalars
        self.frames = dict(frames)
        self.meta = dict(meta)

    def __getitem__(self, key: str):
        if key in self.frames and key not in self.scalars.index:
            return self.frames[key]
        if key in self.scalars.index:
            return self.scalars[key]
        if key in self.frames:
            return self.frames[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.scalars.index or key in self.frames

    def to_dict(self) -> dict[str, float]:
        """Return the scalar statistics as a plain dictionary."""
        return {str(key): float(value) for key, value in self.scalars.items()}

    def __repr__(self) -> str:
        start = self.meta.get("start")
        end = self.meta.get("end")
        return (
            f"PerformanceReport(n_scalars={len(self.scalars)}, "
            f"n_frames={len(self.frames)}, start={start}, end={end})"
        )
