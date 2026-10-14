"""Compatibility import for examples.deployment.eva.contract_server."""

from examples.deployment.eva.contract_server import *  # noqa: F401,F403

import importlib as _importlib
_implementation = _importlib.import_module("examples.deployment.eva.contract_server")

def __getattr__(name):
    return getattr(_implementation, name)

if __name__ == "__main__":
    main()
