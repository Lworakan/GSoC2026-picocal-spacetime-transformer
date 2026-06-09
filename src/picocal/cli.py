"""Minimal command-line entry point."""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="picocal")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    args = parser.parse_args(argv)
    print(f"Would run experiment from config: {args.config}")
    # TODO: load config -> build data -> model -> train -> evaluate
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
