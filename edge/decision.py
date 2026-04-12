from __future__ import annotations

from edge.types import ContaminationResult, DecisionResult


class DecisionEngine:
    def __init__(
        self,
        *,
        review_threshold: float,
        reject_threshold: float,
        label_accept_confidence: float,
    ) -> None:
        self.review_threshold = review_threshold
        self.reject_threshold = reject_threshold
        self.label_accept_confidence = label_accept_confidence

    def canonicalize_contamination(self, *, label: str, contamination: ContaminationResult | None) -> ContaminationResult:
        if contamination is not None and contamination.available:
            return contamination
        if label.strip().lower() == "metal":
            reason = contamination.reason if contamination is not None and contamination.reason else "metal_contamination_unavailable"
            return ContaminationResult.neutral(reason=reason)
        return ContaminationResult.neutral(reason="supported_non_metal_no_cnn")

    def evaluate(
        self,
        *,
        label: str,
        confidence: float,
        contamination: ContaminationResult,
    ) -> DecisionResult:
        if label.strip().lower() == "metal":
            return self._evaluate_metal(confidence=confidence, contamination=contamination)
        return self._evaluate_supported_non_metal(confidence=confidence)

    def _evaluate_metal(self, *, confidence: float, contamination: ContaminationResult) -> DecisionResult:
        if not contamination.applied or not contamination.available:
            return DecisionResult(
                decision="Review",
                contamination_status="UNCERTAIN",
                score=70,
                reason="metal_contamination_unavailable",
            )
        if confidence < self.label_accept_confidence:
            return DecisionResult(
                decision="Review",
                contamination_status="UNCERTAIN",
                score=70,
                reason=f"metal_label_confidence<{self.label_accept_confidence:.2f}",
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

    def _evaluate_supported_non_metal(self, *, confidence: float) -> DecisionResult:
        if confidence >= self.label_accept_confidence:
            return DecisionResult(
                decision="Accept",
                contamination_status="UNCERTAIN",
                score=85,
                reason=f"supported_non_metal_confidence>={self.label_accept_confidence:.2f}",
            )
        return DecisionResult(
            decision="Review",
            contamination_status="UNCERTAIN",
            score=70,
            reason=f"supported_non_metal_confidence<{self.label_accept_confidence:.2f}",
        )
