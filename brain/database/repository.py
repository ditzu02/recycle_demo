from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from brain.models.schema import NormalizedEvent


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

                CREATE INDEX IF NOT EXISTS idx_inference_events_timestamp
                ON inference_events(timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_inference_events_device
                ON inference_events(device_id);

                CREATE INDEX IF NOT EXISTS idx_detected_objects_label
                ON detected_objects(label);

                CREATE INDEX IF NOT EXISTS idx_detected_objects_decision
                ON detected_objects(decision);

                CREATE INDEX IF NOT EXISTS idx_detected_objects_event_order
                ON detected_objects(event_id, object_index);
                """
            )

    def count_events(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM inference_events").fetchone()
        return int(row["count"])

    def insert_event(self, event: NormalizedEvent) -> int:
        received_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
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

    def get_overview(self, device_limit: int = 8, recent_device_limit: int = 8) -> dict[str, object]:
        with self._connect() as connection:
            totals = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM inference_events) AS total_events,
                    (SELECT COUNT(*) FROM detected_objects) AS total_objects,
                    (SELECT COUNT(DISTINCT device_id) FROM inference_events) AS active_devices
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
                SELECT device_id, MAX(timestamp) AS last_seen, COUNT(*) AS event_count
                FROM inference_events
                GROUP BY device_id
                ORDER BY last_seen DESC
                LIMIT ?
                """
                ,
                (recent_device_limit,),
            ).fetchall()

            device_ids = [row["device_id"] for row in recent_devices[:device_limit]]
            event_summaries = []
            object_summaries = []
            device_class_counts = []
            if device_ids:
                placeholders = ", ".join("?" for _ in device_ids)

                event_summaries = connection.execute(
                    f"""
                    SELECT device_id, COUNT(*) AS event_count, MAX(timestamp) AS last_seen
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

        decision_lookup = {row["decision"]: row["count"] for row in decision_counts}
        class_counts_by_device: dict[str, list[dict[str, object]]] = {}
        for row in device_class_counts:
            class_counts_by_device.setdefault(row["device_id"], []).append(
                {"label": row["label"], "count": int(row["count"])}
            )

        event_summary_lookup = {
            row["device_id"]: {
                "event_count": int(row["event_count"]),
                "last_seen": row["last_seen"],
            }
            for row in event_summaries
        }
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
        for row in recent_devices[:device_limit]:
            event_summary = event_summary_lookup.get(
                row["device_id"],
                {"event_count": int(row["event_count"]), "last_seen": row["last_seen"]},
            )
            object_summary = object_summary_lookup.get(
                row["device_id"],
                {
                    "object_count": 0,
                    "accept_count": 0,
                    "review_count": 0,
                    "reject_count": 0,
                },
            )
            devices.append(
                {
                    "device_id": row["device_id"],
                    "event_count": int(event_summary["event_count"]),
                    "object_count": int(object_summary["object_count"]),
                    "last_seen": event_summary["last_seen"],
                    "accept_count": int(object_summary["accept_count"]),
                    "review_count": int(object_summary["review_count"]),
                    "reject_count": int(object_summary["reject_count"]),
                    "class_counts": class_counts_by_device.get(row["device_id"], []),
                }
            )

        return {
            "total_events": int(totals["total_events"]),
            "total_objects": int(totals["total_objects"]),
            "active_devices": int(totals["active_devices"]),
            "accept_count": int(decision_lookup.get("Accept", 0)),
            "review_count": int(decision_lookup.get("Review", 0)),
            "reject_count": int(decision_lookup.get("Reject", 0)),
            "class_counts": [dict(row) for row in class_counts],
            "decision_counts": [dict(row) for row in decision_counts],
            "recent_devices": [dict(row) for row in recent_devices[:recent_device_limit]],
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
                    ORDER BY timestamp DESC
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
                ORDER BY paged_events.timestamp DESC, detected_objects.object_index ASC
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
                    ORDER BY timestamp DESC
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
                ORDER BY paged_events.timestamp DESC
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection
