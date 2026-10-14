"""Compatibility import for examples.deployment.eva.legacy.replay_transport."""

from examples.deployment.eva.legacy.replay_transport import *  # noqa: F401,F403

import importlib as _importlib
_implementation = _importlib.import_module("examples.deployment.eva.legacy.replay_transport")

def __getattr__(name):
    return getattr(_implementation, name)
