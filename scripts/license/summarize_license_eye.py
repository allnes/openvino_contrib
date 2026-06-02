#!/usr/bin/env python3
"""Write a GitHub summary for License-Eye checks and fail on check failures."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from license_config import ConfigError, load_config, require_list, require_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="License-compliance helper config YAML.")
    return parser.parse_args()


def outcome(name: str) -> str:
    return os.environ.get(name, "missing")


def configured_checks(config: dict[str, object]) -> dict[str, str]:
    checks = require_mapping(config, "license_eye_summary", "checks")
    if not all(isinstance(name, str) and isinstance(label, str) for name, label in checks.items()):
        raise ConfigError("license_eye_summary.checks must be a mapping of environment variable names to labels")
    return checks


def build_summary(config: dict[str, object], checks: dict[str, str]) -> str:
    lines = [
        "## License-Eye checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for variable, label in checks.items():
        lines.append(f"| {label} | {outcome(variable)} |")
    lines.append("")
    lines.extend(require_list(config, "license_eye_summary", "failure_guidance"))
    lines.append("")
    return "\n".join(lines)


def write_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(markdown)
            handle.write("\n")
    else:
        print(markdown)


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        checks = configured_checks(config)
        write_summary(build_summary(config, checks))
        return 0 if all(outcome(variable) == "success" for variable in checks) else 1
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
