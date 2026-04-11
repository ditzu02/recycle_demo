from __future__ import annotations

from edge.types import ContaminationResult, DecisionResult, TrackState


class TrackStabilizer:
    def __init__(
        self,
        *,
        stable_after_frames: int,
        min_in_zone_frames_for_evaluation: int,
    ) -> None:
        self.stable_after_frames = stable_after_frames
        self.min_in_zone_frames_for_evaluation = min_in_zone_frames_for_evaluation

    def advance(self, track: TrackState) -> str:
        # Keep the lifecycle explicit so later additions like class voting can
        # hook into the same transition points without rewriting the runtime.
        if track.state == "tentative" and track.consecutive_hits >= self.stable_after_frames:
            track.state = "stable"
        return track.state

    def should_evaluate(self, track: TrackState) -> bool:
        return (
            track.state == "stable"
            and track.decision is None
            and not track.event_emitted
            and track.in_evaluation_zone
            and track.in_zone_consecutive_hits >= self.min_in_zone_frames_for_evaluation
            and track.best_in_zone_snapshot is not None
        )

    def mark_evaluated(
        self,
        track: TrackState,
        *,
        contamination: ContaminationResult | None,
        decision: DecisionResult,
    ) -> None:
        track.contamination = contamination
        track.decision = decision
        if track.best_in_zone_snapshot is not None:
            track.evaluation_frame_index = track.best_in_zone_snapshot.frame_index
        track.state = "evaluated"

    def should_emit_on_zone_exit(self, track: TrackState, *, just_left_zone: bool) -> bool:
        return (
            just_left_zone
            and track.state == "evaluated"
            and track.has_entered_evaluation_zone
            and not track.event_emitted
        )

    def should_emit_on_finish(self, track: TrackState) -> bool:
        return track.state == "evaluated" and not track.event_emitted

    def finish(self, track: TrackState) -> bool:
        if track.state == "tentative" or (track.state == "stable" and track.decision is None):
            track.state = "expired"
            return False
        if track.state == "evaluated":
            should_emit = self.should_emit_on_finish(track)
            track.state = "exited"
            return should_emit
        return False

    def mark_event_queued(self, track: TrackState) -> None:
        track.event_emitted = True

    def mark_emitted(self, track: TrackState) -> None:
        track.event_emitted = True
        track.state = "emitted"
