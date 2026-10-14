"""Compatibility import for examples.deployment.eva.legacy.evaluate_replay."""

from examples.deployment.eva.legacy.evaluate_replay import *  # noqa: F401,F403

import importlib as _importlib
_implementation = _importlib.import_module("examples.deployment.eva.legacy.evaluate_replay")

def __getattr__(name):
    return getattr(_implementation, name)

if __name__ == "__main__":
    main()
