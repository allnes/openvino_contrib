#!/usr/bin/env python3
"""Write a GitHub summary for License-Eye checks and fail on check failures."""

from __future__ import annotations

import os
import sys
from pathlib import Path


CHECKS = (
    ("OpenVINO header policy mirror", "OPENVINO_POLICY_OUTCOME"),
    ("OpenVINO copyright headers", "OPENVINO_HEADERS_OUTCOME"),
    ("SkyWalking Eyes header", "LICENSE_EYE_HEADER_OUTCOME"),
    ("SkyWalking Eyes dependency", "LICENSE_EYE_DEPENDENCY_OUTCOME"),
)


def outcome(name: str) -> str:
    return os.environ.get(name, "missing")


def build_summary() -> str:
    lines = [
        "## License-Eye checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for label, variable in CHECKS:
        lines.append(f"| {label} | {outcome(variable)} |")
    lines.extend(
        [
            "",
            "If this job fails, update `.licenserc.yaml`, source headers, or dependency licensing metadata.",
            "Do not exclude source or vendored directories to make the job green.",
            "",
        ]
    )
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
    write_summary(build_summary())
    return 0 if all(outcome(variable) == "success" for _, variable in CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
