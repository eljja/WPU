from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionPath(str, Enum):
    SPARSE = "sparse"
    HYBRID = "hybrid"
    DENSE = "dense"


@dataclass(frozen=True, slots=True)
class SchedulerMetrics:
    delta_n: int
    fanout: float
    depth: int
    branches: int
    total_n: int
    uncertainty_growth: float = 0.0
    time_budget_ms: float | None = None

    @property
    def rho(self) -> float:
        return (self.delta_n * (self.fanout**self.depth) * self.branches) / max(self.total_n, 1)


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    path: ExecutionPath
    rho: float
    reason: str


class Scheduler:
    def __init__(self, sparse_threshold: float = 0.05, hybrid_threshold: float = 0.25) -> None:
        self.sparse_threshold = sparse_threshold
        self.hybrid_threshold = hybrid_threshold

    def choose_path(self, metrics: SchedulerMetrics) -> SchedulerDecision:
        rho = metrics.rho
        if rho < self.sparse_threshold:
            return SchedulerDecision(ExecutionPath.SPARSE, rho, f"rho={rho:.4f} < {self.sparse_threshold}")
        if rho < self.hybrid_threshold:
            return SchedulerDecision(ExecutionPath.HYBRID, rho, f"rho={rho:.4f} < {self.hybrid_threshold}")
        return SchedulerDecision(ExecutionPath.DENSE, rho, f"rho={rho:.4f} >= {self.hybrid_threshold}")
