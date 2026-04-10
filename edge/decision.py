from __future__ import annotations

from edge.types import ContaminationResult, DecisionResult


class DecisionEngine:
    def __init__(self, *, review_threshold: float, reject_threshold: float) -> None:
        self.review_threshold = review_threshold
        self.reject_threshold = reject_threshold

    def evaluate(self, contamination: ContaminationResult | None) -> DecisionResult:
        if contamination is None or not contamination.available:
            return DecisionResult(
                decision="Review",
                contamination_status="UNCERTAIN",
                score=70,
                reason="contamination_unavailable",
            )

        dirty_probability = contamination.dirty_probability or 0.0
        if dirty_probability >= self.reject_threshold:
            return DecisionResult(
                decision="Reject",
                contamination_status="DIRTY",
                score=30,
                reason=f"dirty_probability>={self.reject_threshold:.2f}",
            )
        if dirty_probability >= self.review_threshold:
            return DecisionResult(
                decision="Review",
                contamination_status="UNCERTAIN",
                score=70,
                reason=f"{self.review_threshold:.2f}<=dirty_probability<{self.reject_threshold:.2f}",
            )
        return DecisionResult(
            decision="Accept",
            contamination_status="CLEAN",
            score=95,
            reason=f"dirty_probability<{self.review_threshold:.2f}",
        )
