"""Compatibility import for examples.deployment.eva.verify_libero_reference."""

from examples.deployment.eva.verify_libero_reference import *  # noqa: F401,F403

import importlib as _importlib
_implementation = _importlib.import_module("examples.deployment.eva.verify_libero_reference")

def __getattr__(name):
    return getattr(_implementation, name)

if __name__ == "__main__":
    main()
