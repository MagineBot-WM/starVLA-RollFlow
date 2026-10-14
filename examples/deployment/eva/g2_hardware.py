"""G2 execution entry using native GDK logic and the deployment socket roles."""

import argparse
import importlib.util
import logging
import signal
import tempfile
from pathlib import Path

from .runtime import DEFAULT_EVA_ROOT, bootstrap
from .bridge_contract import load_contract
from .adapters.hardware import configure_action_subscriber, inspect_hardware_joint_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eva-root", type=Path, default=DEFAULT_EVA_ROOT)
    parser.add_argument("--contract", type=Path, default=Path(__file__).parent / "contracts/agibot_g2_rollflow_h64_c16_async.json")
    parser.add_argument("--obs-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5556")
    args = parser.parse_args()
    bootstrap(args.eva_root)
    contract = load_contract(args.contract)
    if contract["eva"].get("action_socket_role") != "bind":
        parser.error("G2 deployment requires EVA command publisher bind / hardware subscriber connect")
    audit = inspect_hardware_joint_order(args.eva_root, contract)
    spec = importlib.util.spec_from_file_location("starvla_native_g2_hardware", audit["path"])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)  # Real GDK is imported only by this explicit entry.
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory(prefix="eva-g2-node-") as directory:
        # Native construction owns an unused private action socket until the
        # external adapter installs the SUB connection, avoiding a bind collision.
        node = native.AgibotHardwareNode(args.obs_endpoint, f"ipc://{directory}/unused.sock")
        configure_action_subscriber(node, args.action_endpoint)
        def stop(*_):
            node._is_running = False
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        try:
            node.serve_forever()
        finally:
            node.close()


if __name__ == "__main__":
    main()
