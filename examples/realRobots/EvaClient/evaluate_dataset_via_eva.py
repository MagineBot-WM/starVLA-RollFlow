"""Compatibility import for examples.deployment.eva.evaluate_dataset_via_eva."""

from examples.deployment.eva.evaluate_dataset_via_eva import *  # noqa: F401,F403

import importlib as _importlib
_implementation = _importlib.import_module("examples.deployment.eva.evaluate_dataset_via_eva")

def __getattr__(name):
    return getattr(_implementation, name)

if __name__ == "__main__":
    main()
