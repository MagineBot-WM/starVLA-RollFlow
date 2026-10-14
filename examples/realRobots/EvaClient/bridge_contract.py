"""Compatibility import for examples.deployment.eva.bridge_contract."""

from examples.deployment.eva.bridge_contract import *  # noqa: F401,F403

import importlib as _importlib
_implementation = _importlib.import_module("examples.deployment.eva.bridge_contract")

def __getattr__(name):
    return getattr(_implementation, name)
