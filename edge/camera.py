from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import cv2

from edge.config import CameraConfig
from edge.types import FrameSample


class CameraCapture:
    def __init__(self, config: CameraConfig):
        self.config = config
        self._capture = None
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._latest_frame: FrameSample | None = None
        self._frame_index = -1

    def start(self) -> None:
        self._capture = cv2.VideoCapture(self.config.index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.config.index}. Try another index or reconnect the camera."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        self._thread = threading.Thread(target=self._capture_loop, name="edge-camera", daemon=True)
        self._thread.start()

    def get_latest(self, *, after_index: int | None = None, timeout: float = 0.2) -> FrameSample | None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                if self._latest_frame is not None and (
                    after_index is None or self._latest_frame.index > after_index
                ):
                    return self._latest_frame

                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop_event.is_set():
                    return None
                self._condition.wait(timeout=remaining)

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()

    def _capture_loop(self) -> None:
        assert self._capture is not None
        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            self._frame_index += 1
            sample = FrameSample.from_image(
                index=self._frame_index,
                image=frame,
                captured_at=datetime.now(UTC),
            )
            with self._condition:
                self._latest_frame = sample
                self._condition.notify_all()
