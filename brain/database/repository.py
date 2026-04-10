from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from brain.models.schema import NormalizedEvent, NormalizedHeartbeat


HEARTBEAT_FRESH_SECONDS = 30


class EventConflictError(ValueError):
    """Raised when the same event_id is reused by a different device."""


@dataclass(frozen=True)
class EventStoreResult:
    result: str
    row_id: int | None
    received_at: str


class BrainRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS inference_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_uuid TEXT NOT NULL UNIQUE,
                    device_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_index INTEGER,
                    frame_width INTEGER,
                    frame_height INTEGER,
                    frame_index INTEGER,
                    raw_payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS detected_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    object_index INTEGER NOT NULL,
                    object_id TEXT,
                    class_id INTEGER,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    score REAL,
                    decision TEXT,
                    contamination_status TEXT,
                    dirty_probability REAL,
                    clean_probability REAL,
                    bbox_x1 REAL,
                    bbox_y1 REAL,
                    bbox_x2 REAL,
                    bbox_y2 REAL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES inference_events(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS device_status (
                    device_id TEXT PRIMARY KEY,
                    last_contact_received_at TEXT NOT NULL,
                    last_contact_kind TEXT NOT NULL,
                    last_contact_device_timestamp TEXT,
                    last_event_received_at TEXT,
                    last_event_timestamp TEXT,
                    last_event_id TEXT,
                    last_heartbeat_received_at TEXT,
                    last_heartbeat_timestamp TEXT,
                    last_heartbeat_status TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_inference_events_timestamp
                ON inference_events(timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_inference_events_received_at
                ON inference_events(received_at DESC);

                CREATE INDEX IF NOT EXISTS idx_inference_events_device
                ON inference_events(device_id);

                CREATE INDEX IF NOT EXISTS idx_detected_objects_label
                ON detected_objects(label);

                CREATE INDEX IF NOT EXISTS idx_detected_objects_decision
                ON detected_objects(decision);

                CREATE INDEX IF NOT EXISTS idx_detected_objects_event_order
                ON detected_objects(event_id, object_index);

                CREATE INDEX IF NOT EXISTS idx_device_status_last_contact
                ON device_status(last_contact_received_at DESC);
                """
            )

        self.backfill_device_status()

    def backfill_device_status(self) -> None:
        with self._connect() as connection:
            latest_events = connection.execute(
                """
                SELECT
                    event_uuid,
                    device_id,
                    timestamp,
                    received_at
                FROM inference_events AS event_rows
                WHERE id = (
                    SELECT id
                    FROM inference_events AS latest
                    WHERE latest.device_id = event_rows.device_id
                    ORDER BY latest.received_at DESC, latest.id DESC
                    LIMIT 1
                )
                """
            ).fetchall()

            for row in latest_events:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO device_status (
                        device_id,
                        last_contact_received_at,
                        last_contact_kind,
                        last_contact_device_timestamp,
                        last_event_received_at,
                        last_event_timestamp,
                        last_event_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["device_id"],
                        row["received_at"],
                        "event",
                        row["timestamp"],
                        row["received_at"],
                        row["timestamp"],
                        row["event_uuid"],
                    ),
                )

    def count_events(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM inference_events").fetchone()
        return int(row["count"])

    def insert_event(self, event: NormalizedEvent, received_at: str | None = None) -> int:
        stored_at = received_at or self._utc_now()
        with self._connect() as connection:
            row_id = self._insert_event_rows(connection, event, stored_at)
            self._upsert_event_status(connection, event, stored_at)
        return row_id

    def store_event(self, event: NormalizedEvent) -> EventStoreResult:
        received_at = self._utc_now()
        with self._connect() as connection:
            existing = self._find_existing_event(connection, event.event_uuid)
            if existing is not None:
                return self._handle_existing_event(connection, existing, event, received_at)

            try:
                row_id = self._insert_event_rows(connection, event, received_at)
            except sqlite3.IntegrityError:
                existing = self._find_existing_event(connection, event.event_uuid)
                if existing is None:
                    raise
                return self._handle_existing_event(connection, existing, event, received_at)

            self._upsert_event_status(connection, event, received_at)
            return EventStoreResult(result="accepted", row_id=row_id, received_at=received_at)

    def record_heartbeat(self, heartbeat: NormalizedHeartbeat) -> str:
        received_at = self._utc_now()
        with self._connect() as connection:
            self._upsert_heartbeat_status(connection, heartbeat, received_at)
        return received_at

    def get_overview(self, device_limit: int = 8, recent_device_limit: int = 8) -> dict[str, object]:
        with self._connect() as connection:
            totals = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM inference_events) AS total_events,
                    (SELECT COUNT(*) FROM detected_objects) AS total_objects,
                    (SELECT COUNT(*) FROM device_status) AS active_devices
                """
            ).fetchone()

            class_counts = connection.execute(
                """
                SELECT label, COUNT(*) AS count
                FROM detected_objects
                GROUP BY label
                ORDER BY count DESC, label ASC
                """
            ).fetchall()

            decision_counts = connection.execute(
                """
                SELECT COALESCE(decision, 'Unknown') AS decision, COUNT(*) AS count
                FROM detected_objects
                GROUP BY COALESCE(decision, 'Unknown')
                ORDER BY count DESC, decision ASC
                """
            ).fetchall()

            recent_devices = connection.execute(
                """
                SELECT
                    device_id,
                    last_contact_received_at,
                    last_contact_kind,
                    last_contact_device_timestamp,
                    last_event_received_at,
                    last_event_timestamp,
                    last_event_id,
                    last_heartbeat_received_at,
                    last_heartbeat_timestamp,
                    last_heartbeat_status
                FROM device_status
                ORDER BY last_contact_received_at DESC, device_id ASC
                LIMIT ?
                """,
                (recent_device_limit,),
            ).fetchall()

            device_ids = [row["device_id"] for row in recent_devices]
            event_counts = []
            object_summaries = []
            device_class_counts = []
            if device_ids:
                placeholders = ", ".join("?" for _ in device_ids)

                event_counts = connection.execute(
                    f"""
                    SELECT device_id, COUNT(*) AS event_count
                    FROM inference_events
                    WHERE device_id IN ({placeholders})
                    GROUP BY device_id
                    """,
                    device_ids,
                ).fetchall()

                object_summaries = connection.execute(
                    f"""
                    SELECT
                        inference_events.device_id,
                        COUNT(detected_objects.id) AS object_count,
                        SUM(CASE WHEN detected_objects.decision = 'Accept' THEN 1 ELSE 0 END) AS accept_count,
                        SUM(CASE WHEN detected_objects.decision = 'Review' THEN 1 ELSE 0 END) AS review_count,
                        SUM(CASE WHEN detected_objects.decision = 'Reject' THEN 1 ELSE 0 END) AS reject_count
                    FROM detected_objects INDEXED BY idx_detected_objects_event_order
                    INNER JOIN inference_events ON inference_events.id = detected_objects.event_id
                    WHERE inference_events.device_id IN ({placeholders})
                    GROUP BY inference_events.device_id
                    """,
                    device_ids,
                ).fetchall()

                device_class_counts = connection.execute(
                    f"""
                    SELECT
                        inference_events.device_id,
                        detected_objects.label,
                        COUNT(*) AS count
                    FROM detected_objects INDEXED BY idx_detected_objects_event_order
                    INNER JOIN inference_events ON inference_events.id = detected_objects.event_id
                    WHERE inference_events.device_id IN ({placeholders})
                    GROUP BY inference_events.device_id, detected_objects.label
                    ORDER BY inference_events.device_id ASC, count DESC, detected_objects.label ASC
                    """,
                    device_ids,
                ).fetchall()

        now = datetime.now(UTC)
        decision_lookup = {row["decision"]: row["count"] for row in decision_counts}
        event_count_lookup = {row["device_id"]: int(row["event_count"]) for row in event_counts}
        class_counts_by_device: dict[str, list[dict[str, object]]] = {}
        for row in device_class_counts:
            class_counts_by_device.setdefault(row["device_id"], []).append(
                {"label": row["label"], "count": int(row["count"])}
            )

        object_summary_lookup = {
            row["device_id"]: {
                "object_count": int(row["object_count"] or 0),
                "accept_count": int(row["accept_count"] or 0),
                "review_count": int(row["review_count"] or 0),
                "reject_count": int(row["reject_count"] or 0),
            }
            for row in object_summaries
        }

        devices = []
        recent_device_rows = []
        for row in recent_devices:
            object_summary = object_summary_lookup.get(
                row["device_id"],
                {
                    "object_count": 0,
                    "accept_count": 0,
                    "review_count": 0,
                    "reject_count": 0,
                },
            )
            device_payload = {
                "device_id": row["device_id"],
                "event_count": event_count_lookup.get(row["device_id"], 0),
                "object_count": int(object_summary["object_count"]),
                "last_seen": row["last_contact_received_at"],
                "last_contact_kind": row["last_contact_kind"],
                "last_contact_device_timestamp": row["last_contact_device_timestamp"],
                "last_event_received_at": row["last_event_received_at"],
                "last_event_timestamp": row["last_event_timestamp"],
                "last_event_id": row["last_event_id"],
                "last_heartbeat_received_at": row["last_heartbeat_received_at"],
                "last_heartbeat_timestamp": row["last_heartbeat_timestamp"],
                "last_heartbeat_status": row["last_heartbeat_status"],
                "heartbeat_freshness": self._heartbeat_freshness(row["last_heartbeat_received_at"], now),
                "accept_count": int(object_summary["accept_count"]),
                "review_count": int(object_summary["review_count"]),
                "reject_count": int(object_summary["reject_count"]),
                "class_counts": class_counts_by_device.get(row["device_id"], []),
            }
            recent_device_rows.append(dict(device_payload))
            if len(devices) < device_limit:
                devices.append(device_payload)

        return {
            "total_events": int(totals["total_events"]),
            "total_objects": int(totals["total_objects"]),
            "active_devices": int(totals["active_devices"]),
            "accept_count": int(decision_lookup.get("Accept", 0)),
            "review_count": int(decision_lookup.get("Review", 0)),
            "reject_count": int(decision_lookup.get("Reject", 0)),
            "class_counts": [dict(row) for row in class_counts],
            "decision_counts": [dict(row) for row in decision_counts],
            "recent_devices": recent_device_rows,
            "devices": devices,
        }

    def get_recent_objects_for_event_page(
        self,
        event_limit: int,
        event_offset: int = 0,
        object_limit: int | None = None,
    ) -> list[dict[str, object]]:
        with self._connect() as connection:
            query = """
                WITH paged_events AS (
                    SELECT
                        id,
                        event_uuid,
                        device_id,
                        timestamp,
                        received_at
                    FROM inference_events
                    ORDER BY received_at DESC, id DESC
                    LIMIT ?
                    OFFSET ?
                )
                SELECT
                    paged_events.event_uuid,
                    paged_events.device_id,
                    paged_events.timestamp,
                    paged_events.received_at,
                    detected_objects.object_index,
                    detected_objects.object_id,
                    detected_objects.label,
                    detected_objects.confidence,
                    detected_objects.score,
                    detected_objects.decision,
                    detected_objects.contamination_status,
                    detected_objects.dirty_probability,
                    detected_objects.clean_probability,
                    detected_objects.bbox_x1,
                    detected_objects.bbox_y1,
                    detected_objects.bbox_x2,
                    detected_objects.bbox_y2
                FROM paged_events
                INNER JOIN detected_objects INDEXED BY idx_detected_objects_event_order
                    ON detected_objects.event_id = paged_events.id
                ORDER BY paged_events.received_at DESC, paged_events.id DESC, detected_objects.object_index ASC
            """
            params: tuple[int, ...]
            if object_limit is not None:
                query += "\nLIMIT ?"
                params = (event_limit, event_offset, object_limit)
            else:
                params = (event_limit, event_offset)

            rows = connection.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_recent_objects(self, limit: int = 50) -> list[dict[str, object]]:
        return self.get_recent_objects_for_event_page(event_limit=limit, event_offset=0, object_limit=limit)

    def get_recent_events(self, limit: int = 20, offset: int = 0) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                WITH paged_events AS (
                    SELECT
                        id,
                        event_uuid,
                        device_id,
                        timestamp,
                        received_at,
                        source_type,
                        source_index,
                        frame_width,
                        frame_height,
                        frame_index
                    FROM inference_events
                    ORDER BY received_at DESC, id DESC
                    LIMIT ?
                    OFFSET ?
                )
                SELECT
                    paged_events.event_uuid,
                    paged_events.device_id,
                    paged_events.timestamp,
                    paged_events.received_at,
                    paged_events.source_type,
                    paged_events.source_index,
                    paged_events.frame_width,
                    paged_events.frame_height,
                    paged_events.frame_index,
                    COUNT(detected_objects.id) AS object_count
                FROM paged_events
                LEFT JOIN detected_objects INDEXED BY idx_detected_objects_event_order
                    ON detected_objects.event_id = paged_events.id
                GROUP BY paged_events.id
                ORDER BY paged_events.received_at DESC, paged_events.id DESC
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _find_existing_event(self, connection: sqlite3.Connection, event_uuid: str) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT id, device_id
            FROM inference_events
            WHERE event_uuid = ?
            """,
            (event_uuid,),
        ).fetchone()

    def _handle_existing_event(
        self,
        connection: sqlite3.Connection,
        existing: sqlite3.Row,
        event: NormalizedEvent,
        received_at: str,
    ) -> EventStoreResult:
        if existing["device_id"] != event.device_id:
            raise EventConflictError("event_id already exists for a different device_id.")

        self._touch_device_contact(
            connection,
            device_id=event.device_id,
            received_at=received_at,
            kind="event",
            device_timestamp=event.timestamp,
        )
        return EventStoreResult(result="duplicate", row_id=int(existing["id"]), received_at=received_at)

    def _insert_event_rows(
        self,
        connection: sqlite3.Connection,
        event: NormalizedEvent,
        received_at: str,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO inference_events (
                event_uuid, device_id, timestamp, received_at, source_type,
                source_index, frame_width, frame_height, frame_index, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_uuid,
                event.device_id,
                event.timestamp,
                received_at,
                event.source_type,
                event.source_index,
                event.frame_width,
                event.frame_height,
                event.frame_index,
                json.dumps(event.raw_payload, sort_keys=True),
            ),
        )
        event_id = int(cursor.lastrowid)
        for index, obj in enumerate(event.objects):
            bbox = obj.bbox or (None, None, None, None)
            connection.execute(
                """
                INSERT INTO detected_objects (
                    event_id, object_index, object_id, class_id, label, confidence,
                    score, decision, contamination_status, dirty_probability,
                    clean_probability, bbox_x1, bbox_y1, bbox_x2, bbox_y2, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    index,
                    obj.object_id,
                    obj.class_id,
                    obj.label,
                    obj.confidence,
                    obj.score,
                    obj.decision,
                    obj.contamination_status,
                    obj.dirty_probability,
                    obj.clean_probability,
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    json.dumps(obj.metadata, sort_keys=True),
                ),
            )
        return event_id

    def _upsert_event_status(
        self,
        connection: sqlite3.Connection,
        event: NormalizedEvent,
        received_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO device_status (
                device_id,
                last_contact_received_at,
                last_contact_kind,
                last_contact_device_timestamp,
                last_event_received_at,
                last_event_timestamp,
                last_event_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                last_contact_received_at = excluded.last_contact_received_at,
                last_contact_kind = excluded.last_contact_kind,
                last_contact_device_timestamp = excluded.last_contact_device_timestamp,
                last_event_received_at = excluded.last_event_received_at,
                last_event_timestamp = excluded.last_event_timestamp,
                last_event_id = excluded.last_event_id
            """,
            (
                event.device_id,
                received_at,
                "event",
                event.timestamp,
                received_at,
                event.timestamp,
                event.event_uuid,
            ),
        )

    def _upsert_heartbeat_status(
        self,
        connection: sqlite3.Connection,
        heartbeat: NormalizedHeartbeat,
        received_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO device_status (
                device_id,
                last_contact_received_at,
                last_contact_kind,
                last_contact_device_timestamp,
                last_heartbeat_received_at,
                last_heartbeat_timestamp,
                last_heartbeat_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                last_contact_received_at = excluded.last_contact_received_at,
                last_contact_kind = excluded.last_contact_kind,
                last_contact_device_timestamp = excluded.last_contact_device_timestamp,
                last_heartbeat_received_at = excluded.last_heartbeat_received_at,
                last_heartbeat_timestamp = excluded.last_heartbeat_timestamp,
                last_heartbeat_status = excluded.last_heartbeat_status
            """,
            (
                heartbeat.device_id,
                received_at,
                "heartbeat",
                heartbeat.timestamp,
                received_at,
                heartbeat.timestamp,
                heartbeat.status,
            ),
        )

    def _touch_device_contact(
        self,
        connection: sqlite3.Connection,
        device_id: str,
        received_at: str,
        kind: str,
        device_timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO device_status (
                device_id,
                last_contact_received_at,
                last_contact_kind,
                last_contact_device_timestamp
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                last_contact_received_at = excluded.last_contact_received_at,
                last_contact_kind = excluded.last_contact_kind,
                last_contact_device_timestamp = excluded.last_contact_device_timestamp
            """,
            (device_id, received_at, kind, device_timestamp),
        )

    @staticmethod
    def _heartbeat_freshness(value: str | None, now: datetime) -> str:
        if not value:
            return "never"
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return "stale"
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        age_seconds = (now - parsed.astimezone(UTC)).total_seconds()
        return "fresh" if age_seconds <= HEARTBEAT_FRESH_SECONDS else "stale"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()
