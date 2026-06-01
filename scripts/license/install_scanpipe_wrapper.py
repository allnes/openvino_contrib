#!/usr/bin/env python3
"""Install a hardened Python scanpipe wrapper for ScanCode Action."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path


WRAPPER = """#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"error: required environment variable {name} is not set", file=sys.stderr)
        sys.exit(2)
    return value


workspace = require_env("GITHUB_WORKSPACE")
image = require_env("SCANCODEIO_IMAGE")

cmd = [
    "docker",
    "run",
    "--rm",
    "--network",
    "host",
    "--user",
    f"{os.getuid()}:{os.getgid()}",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "-e",
    "SECRET_KEY",
    "-e",
    "SCANCODEIO_WORKSPACE_LOCATION",
    "-e",
    "VULNERABLECODE_URL",
    "-e",
    "SCANCODEIO_POLICIES_FILE=/workspace/.github/scancode/policies.yml",
    "-e",
    "HOME=/workspace/.home",
    "-v",
    f"{workspace}:/workspace",
    image,
    "scanpipe",
    *sys.argv[1:],
]

os.execvp("docker", cmd)
"""


class InstallError(RuntimeError):
    """A user-facing wrapper installation error."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=Path(os.environ.get("RUNNER_TEMP", "")) if os.environ.get("RUNNER_TEMP") else None,
        help="Directory where the scanpipe executable should be installed. Defaults to RUNNER_TEMP.",
    )
    return parser.parse_args()


def append_github_path(path: Path) -> None:
    github_path = os.environ.get("GITHUB_PATH")
    if not github_path:
        raise InstallError("GITHUB_PATH is not set; cannot expose scanpipe to subsequent action steps.")
    with Path(github_path).open("a", encoding="utf-8") as handle:
        handle.write(f"{path}\n")


def install_wrapper(install_dir: Path) -> Path:
    install_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = install_dir / "scanpipe"
    wrapper_path.write_text(WRAPPER, encoding="utf-8")
    current_mode = wrapper_path.stat().st_mode
    wrapper_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    append_github_path(install_dir)
    return wrapper_path


def main() -> int:
    args = parse_args()
    try:
        if args.install_dir is None:
            raise InstallError("RUNNER_TEMP is not set; pass --install-dir explicitly.")
        wrapper_path = install_wrapper(args.install_dir)
        print(f"Installed ScanCode scanpipe wrapper at {wrapper_path}")
        return 0
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
