"""Bounded JSONL diagnostics for commands issued by EVA, never execution ACKs."""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ExecutionTrace:
    def __init__(self, directory=None, limit=10000):
        self.path = Path(directory) / "execution.jsonl" if directory else None
        self.limit = int(limit)
        self.count = 0

    def write(self, event, **fields):
        if self.path is None or self.count >= self.limit:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(event=event, monotonic_s=time.monotonic(),
                                             **fields), allow_nan=False) + "\n")
            self.count += 1
        except Exception:
            # Logging must never discard an already advanced model chunk or
            # make the caller repeat a command that was already transmitted.
            logger.warning("Could not save execution trace", exc_info=True)
