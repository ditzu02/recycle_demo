from __future__ import annotations

from edge.types import ContaminationResult, DecisionResult, TrackState


class TrackStabilizer:
    def __init__(self, *, stable_after_frames: int) -> None:
        self.stable_after_frames = stable_after_frames

    def advance(self, track: TrackState) -> str:
        # Keep the lifecycle explicit so later additions like class voting can
        # hook into the same transition points without rewriting the runtime.
        if track.state == "tentative" and track.consecutive_hits >= self.stable_after_frames:
            track.state = "stable"
        return track.state

    def should_evaluate(self, track: TrackState) -> bool:
        return track.state == "stable" and track.decision is None

    def mark_evaluated(
        self,
        track: TrackState,
        *,
        contamination: ContaminationResult | None,
        decision: DecisionResult,
    ) -> None:
        track.contamination = contamination
        track.decision = decision
        track.state = "evaluated"

    def finish(self, track: TrackState) -> bool:
        if track.state == "tentative":
            track.state = "expired"
            return False
        if track.state in {"stable", "evaluated"}:
            track.state = "exited"
            return True
        return False

    def mark_emitted(self, track: TrackState) -> None:
        track.state = "emitted"
