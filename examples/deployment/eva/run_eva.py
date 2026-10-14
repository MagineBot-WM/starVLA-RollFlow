"""Launch the unchanged EVA app with the additive StarVLA/LIBERO plugins loaded."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from .runtime import load_config

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eva-root", type=Path, default=ROOT / "EVA-CLIENT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--trace-dir", type=Path, help="Capture bounded observation/action NPZ traces")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config_path = args.config.resolve()
    config = load_config(config_path, args.eva_root)
    if args.trace_dir:
        config.policy.backend_options.trace_dir = str(args.trace_dir.resolve())
        config.transport.trace_dir = str(args.trace_dir.resolve())
    from core.app.run import run
    run(
        config,
        web_port=args.web_port,
        config_path=str(config_path),
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
