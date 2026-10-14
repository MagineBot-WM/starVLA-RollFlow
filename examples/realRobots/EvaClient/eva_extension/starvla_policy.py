"""Compatibility import for examples.deployment.eva.eva_extension.starvla_policy."""

from examples.deployment.eva.eva_extension.starvla_policy import *  # noqa: F401,F403

import importlib as _importlib
_implementation = _importlib.import_module("examples.deployment.eva.eva_extension.starvla_policy")

def __getattr__(name):
    return getattr(_implementation, name)
