"""Compatibility import for examples.deployment.eva.legacy.observation."""

from examples.deployment.eva.legacy.observation import *  # noqa: F401,F403

import importlib as _importlib
_implementation = _importlib.import_module("examples.deployment.eva.legacy.observation")

def __getattr__(name):
    return getattr(_implementation, name)
