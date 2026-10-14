"""Run CPU deployment checks in the EVA environment."""

from pathlib import Path
import sys


def main():
    try:
        import pytest
    except ImportError as error:
        raise SystemExit("Install pytest in the EVA environment to run deployment checks") from error
    raise SystemExit(pytest.main([str(Path(__file__).parent / "tests"), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
