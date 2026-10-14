"""Compatibility import for examples.deployment.eva.eva_extension.libero_sim."""

from examples.deployment.eva.eva_extension.libero_sim import *  # noqa: F401,F403

import importlib as _importlib
_implementation = _importlib.import_module("examples.deployment.eva.eva_extension.libero_sim")

def __getattr__(name):
    return getattr(_implementation, name)
