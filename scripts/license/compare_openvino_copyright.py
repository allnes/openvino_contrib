#!/usr/bin/env python3
"""Compare the local OpenVINO copyright checker mirror with upstream."""

from __future__ import annotations

import argparse
import difflib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/openvinotoolkit/openvino/HEAD/"
    ".github/scripts/check_copyright.py"
)


class CompareError(RuntimeError):
    """A user-facing comparison error."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        type=Path,
        default=Path(os.environ.get("OPENVINO_COPYRIGHT_SCRIPT", "scripts/license/openvino_check_copyright.py")),
        help="Local mirrored checker path.",
    )
    parser.add_argument(
        "--source-url",
        default=os.environ.get("OPENVINO_COPYRIGHT_SCRIPT_URL", DEFAULT_SOURCE_URL),
        help="OpenVINO upstream checker URL.",
    )
    return parser.parse_args()


def fetch_lines(url: str) -> list[str]:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8").splitlines()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise CompareError(f"failed to fetch {url}: {exc}") from exc


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise CompareError(f"local checker does not exist: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def normalized(lines: list[str]) -> list[str]:
    return [line.rstrip() + "\n" for line in lines]


def compare(local_path: Path, source_url: str) -> int:
    local = normalized(read_lines(local_path))
    upstream = normalized(fetch_lines(source_url))
    if local == upstream:
        print(f"{local_path} matches {source_url}")
        return 0

    print(
        f"::error::OpenVINO copyright checker changed upstream. "
        f"Update {local_path} from {source_url} and review header policy changes."
    )
    sys.stdout.writelines(
        difflib.unified_diff(
            local,
            upstream,
            fromfile=str(local_path),
            tofile=source_url,
        )
    )
    return 1


def main() -> int:
    args = parse_args()
    try:
        return compare(args.local, args.source_url)
    except CompareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
