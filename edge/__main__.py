from __future__ import annotations

import logging

from edge.config import build_config
from edge.runtime import EdgeRuntime


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = build_config(argv)
    runtime = EdgeRuntime(config)
    runtime.run()


if __name__ == "__main__":
    main()
