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
    objectification_score: float | None = None

    @property
    def rho(self) -> float:
        return (self.delta_n * (self.fanout**self.depth) * self.branches) / max(self.total_n, 1)


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    path: ExecutionPath
    rho: float
    reason: str


class Scheduler:
    def __init__(
        self,
        sparse_threshold: float = 0.05,
        hybrid_threshold: float = 0.25,
        sparse_objectification_threshold: float = 0.75,
        dense_objectification_threshold: float = 0.50,
    ) -> None:
        self.sparse_threshold = sparse_threshold
        self.hybrid_threshold = hybrid_threshold
        self.sparse_objectification_threshold = sparse_objectification_threshold
        self.dense_objectification_threshold = dense_objectification_threshold

    def choose_path(self, metrics: SchedulerMetrics) -> SchedulerDecision:
        rho = metrics.rho
        score = metrics.objectification_score
        if score is not None and score < self.dense_objectification_threshold:
            return SchedulerDecision(
                ExecutionPath.DENSE,
                rho,
                f"objectification={score:.4f} < {self.dense_objectification_threshold}",
            )
        if score is not None and score < self.sparse_objectification_threshold and rho < self.sparse_threshold:
            return SchedulerDecision(
                ExecutionPath.HYBRID,
                rho,
                f"objectification={score:.4f} < {self.sparse_objectification_threshold}",
            )
        if rho < self.sparse_threshold:
            return SchedulerDecision(ExecutionPath.SPARSE, rho, f"rho={rho:.4f} < {self.sparse_threshold}")
        if rho < self.hybrid_threshold:
            return SchedulerDecision(ExecutionPath.HYBRID, rho, f"rho={rho:.4f} < {self.hybrid_threshold}")
        return SchedulerDecision(ExecutionPath.DENSE, rho, f"rho={rho:.4f} >= {self.hybrid_threshold}")
