from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime, timedelta
from urllib import request

from brain.mock.seed import generate_mock_event


def main() -> None:
    parser = argparse.ArgumentParser(description="Send mock finalized inspection events to the brain server.")
    parser.add_argument("--server", default="http://127.0.0.1:8000/api/inference", help="Finalized inspection API endpoint.")
    parser.add_argument("--count", type=int, default=10, help="Number of events to send.")
    parser.add_argument("--devices", type=int, default=3, help="Number of simulated devices.")
    parser.add_argument("--interval", type=float, default=0.15, help="Delay between events in seconds.")
    parser.add_argument("--seed", type=int, default=20260322, help="Random seed for deterministic payloads.")
    args = parser.parse_args()

    randomizer = random.Random(args.seed)
    start_time = datetime.now(UTC)
    for sequence in range(args.count):
        device_id = f"pi_{(sequence % args.devices) + 1:02d}"
        event_time = start_time + timedelta(seconds=sequence)
        payload = generate_mock_event(
            randomizer=randomizer,
            device_id=device_id,
            timestamp=event_time,
            sequence=sequence + 10_000,
        )
        response = _post_json(args.server, payload)
        print(
            f"sent seq={sequence:02d} device={device_id} "
            f"objects={len(payload['objects'])} status={response['status']} result={response.get('result', '-')}"
        )
        time.sleep(args.interval)


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
