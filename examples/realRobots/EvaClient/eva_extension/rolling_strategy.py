"""Compatibility import for examples.deployment.eva.eva_extension.rolling_strategy."""

from examples.deployment.eva.eva_extension.rolling_strategy import *  # noqa: F401,F403

import importlib as _importlib
_implementation = _importlib.import_module("examples.deployment.eva.eva_extension.rolling_strategy")

def __getattr__(name):
    return getattr(_implementation, name)
