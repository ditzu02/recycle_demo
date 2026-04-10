from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error, request


@dataclass(frozen=True)
class TransportResult:
    status_code: int | None
    accepted: bool = False
    duplicate: bool = False
    retryable: bool = False
    detail: str | None = None
    payload: dict[str, Any] | None = None


class BrainTransport:
    def __init__(
        self,
        *,
        event_endpoint_url: str,
        heartbeat_endpoint_url: str,
        timeout_seconds: float = 5.0,
        event_retry_attempts: int = 2,
        event_retry_backoff_seconds: float = 0.5,
        urlopen: Callable[..., Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.event_endpoint_url = event_endpoint_url
        self.heartbeat_endpoint_url = heartbeat_endpoint_url
        self.timeout_seconds = timeout_seconds
        self.event_retry_attempts = event_retry_attempts
        self.event_retry_backoff_seconds = event_retry_backoff_seconds
        self._urlopen = urlopen or request.urlopen
        self._sleep_fn = sleep_fn

    def send_event(self, payload: dict[str, Any]) -> TransportResult:
        attempts = self.event_retry_attempts + 1
        result = TransportResult(status_code=None, retryable=True, detail="send_event_not_attempted")
        for attempt in range(attempts):
            result = self._post_json(self.event_endpoint_url, payload)
            if result.accepted or not result.retryable or attempt == attempts - 1:
                return result
            self._sleep_fn(self.event_retry_backoff_seconds * (attempt + 1))
        return result

    def send_heartbeat(self, payload: dict[str, Any]) -> TransportResult:
        return self._post_json(self.heartbeat_endpoint_url, payload)

    def _post_json(self, url: str, payload: dict[str, Any]) -> TransportResult:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._urlopen(req, timeout=self.timeout_seconds) as response:
                response_payload = _read_json_response(response)
                return _interpret_status(response.status, response_payload)
        except error.HTTPError as exc:
            response_payload = _safe_json(exc.read())
            retryable = exc.code >= 500
            return TransportResult(
                status_code=exc.code,
                retryable=retryable,
                detail=str(exc.reason),
                payload=response_payload,
            )
        except error.URLError as exc:
            return TransportResult(
                status_code=None,
                retryable=True,
                detail=str(exc.reason),
            )


def _read_json_response(response) -> dict[str, Any] | None:
    raw = response.read()
    if not raw:
        return None
    return _safe_json(raw)


def _safe_json(raw: bytes) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _interpret_status(status_code: int, payload: dict[str, Any] | None) -> TransportResult:
    result_text = str((payload or {}).get("result", "")).strip().lower()
    accepted = status_code in {200, 201} and result_text in {"accepted", "duplicate"}
    duplicate = status_code == 200 and result_text == "duplicate"
    retryable = status_code >= 500
    return TransportResult(
        status_code=status_code,
        accepted=accepted,
        duplicate=duplicate,
        retryable=retryable,
        payload=payload,
    )
