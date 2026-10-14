"""One deployment entry point; each command imports only its own dependencies."""

import importlib
import sys

COMMANDS = {
    "server": "contract_server", "client": "run_eva", "doctor": "doctor",
    "replay": "evaluate_dataset_via_eva", "wire-replay": "evaluate_g2_wire",
    "hardware": "g2_hardware", "async-replay": "evaluate_g2_async",
    "capture": "capture_observation", "timing": "evaluate_prefetch_timing",
    "libero": "evaluate_libero_via_eva", "test": "self_test",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python -m examples.deployment.eva {server|client|doctor|capture|timing|replay|wire-replay|async-replay|hardware|libero|test} [options]")
        return
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown deployment command: {command}")
    importlib.import_module(f"{__package__}.{COMMANDS[command]}").main()


if __name__ == "__main__":
    main()
