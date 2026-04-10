from __future__ import annotations

from edge.types import Detection, FrameSample, TrackState


class TrackManager:
    def __init__(self, *, iou_threshold: float, max_missed_frames: int) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._active_tracks: dict[int, TrackState] = {}
        self._next_track_number = 1

    @property
    def active_tracks(self) -> list[TrackState]:
        return list(self._active_tracks.values())

    def update(self, frame: FrameSample, detections: list[Detection]) -> tuple[list[TrackState], list[TrackState]]:
        prior_tracks = list(self._active_tracks.values())
        matches = self._greedy_matches(prior_tracks, detections)
        matched_track_numbers = {prior_tracks[track_index].track_number for track_index, _ in matches}
        matched_detection_indexes = {detection_index for _, detection_index in matches}

        for track_index, detection_index in matches:
            track = prior_tracks[track_index]
            detection = detections[detection_index]
            track.observe(frame, detection)

        finished_tracks: list[TrackState] = []
        for track in prior_tracks:
            if track.track_number in matched_track_numbers:
                continue
            track.miss()
            if track.missed_frames >= self.max_missed_frames:
                finished_tracks.append(self._active_tracks.pop(track.track_number))

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indexes:
                continue
            track = TrackState(
                track_number=self._next_track_number,
                object_id=f"track-{self._next_track_number:04d}",
            )
            self._next_track_number += 1
            track.observe(frame, detection)
            self._active_tracks[track.track_number] = track

        return list(self._active_tracks.values()), finished_tracks

    def _greedy_matches(self, tracks: list[TrackState], detections: list[Detection]) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            if track.latest_snapshot is None:
                continue
            for detection_index, detection in enumerate(detections):
                iou = track.latest_snapshot.bbox.intersection_over_union(detection.bbox)
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_index, detection_index))

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches.append((track_index, detection_index))
        return matches
