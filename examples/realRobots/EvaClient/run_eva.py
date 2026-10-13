"""Launch the unchanged EVA app with the additive StarVLA/LIBERO plugins loaded."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eva-root", type=Path, default=ROOT / "EVA-CLIENT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.eva_root.resolve()))
    sys.path.insert(0, str(args.eva_root.resolve() / "src"))

    # Import the standard backends first, then register this package's parallel
    # additions. No file under EVA-CLIENT/src is patched.
    import policy_client  # noqa: F401
    import robots  # noqa: F401
    import transport  # noqa: F401
    from examples.realRobots.EvaClient import eva_extension  # noqa: F401
    from core.app.run import run
    from core.config import load_config

    config_path = args.config.resolve()
    config = load_config(config_path)
    run(
        config,
        web_port=args.web_port,
        config_path=str(config_path),
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
