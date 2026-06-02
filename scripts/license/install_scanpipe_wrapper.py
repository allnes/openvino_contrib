#!/usr/bin/env python3
"""Install a hardened scanpipe wrapper using YAML-provided settings."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from license_config import ConfigError, load_config, require_list, require_mapping, require_str


WRAPPER = """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"error: required environment variable {name} is not set", file=sys.stderr)
        sys.exit(2)
    return value


config_path = Path(__file__).with_name("scanpipe-wrapper.json")
config = json.loads(config_path.read_text(encoding="utf-8"))

workspace = require_env(config["workspace_env"])
image = require_env(config["image_env"])
workspace_mount = config["workspace_mount"]

cmd = [
    "docker",
    "run",
    "--rm",
    "--network",
    config["network"],
    "--user",
    f"{os.getuid()}:{os.getgid()}",
    "--cap-drop",
    config["cap_drop"],
    "--security-opt",
    config["security_opt"],
]

for name in config["env_passthrough"]:
    cmd.extend(["-e", name])
for name, value in config["env_values"].items():
    cmd.extend(["-e", f"{name}={value}"])

cmd.extend([
    "-v",
    f"{workspace}:{workspace_mount}",
    image,
    "scanpipe",
    *sys.argv[1:],
])

os.execvp("docker", cmd)
"""


class InstallError(RuntimeError):
    """A user-facing wrapper installation error."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="License-compliance helper config YAML.")
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


def wrapper_config(config: dict[str, Any]) -> dict[str, Any]:
    env_values = require_mapping(config, "scanpipe", "env_values")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in env_values.items()):
        raise InstallError("scanpipe.env_values must be a mapping of strings")

    return {
        "workspace_env": require_str(config, "scanpipe", "workspace_env"),
        "image_env": require_str(config, "scanpipe", "image_env"),
        "workspace_mount": require_str(config, "scanpipe", "workspace_mount"),
        "network": require_str(config, "scanpipe", "network"),
        "cap_drop": require_str(config, "scanpipe", "cap_drop"),
        "security_opt": require_str(config, "scanpipe", "security_opt"),
        "env_passthrough": require_list(config, "scanpipe", "env_passthrough"),
        "env_values": env_values,
    }


def install_wrapper(install_dir: Path, config: dict[str, Any]) -> Path:
    install_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = install_dir / "scanpipe"
    config_path = install_dir / "scanpipe-wrapper.json"
    wrapper_path.write_text(WRAPPER, encoding="utf-8")
    config_path.write_text(json.dumps(wrapper_config(config), indent=2, sort_keys=True), encoding="utf-8")
    current_mode = wrapper_path.stat().st_mode
    wrapper_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    append_github_path(install_dir)
    return wrapper_path


def main() -> int:
    args = parse_args()
    try:
        if args.install_dir is None:
            raise InstallError("RUNNER_TEMP is not set; pass --install-dir explicitly.")
        config = load_config(args.config)
        wrapper_path = install_wrapper(args.install_dir, config)
        print(f"Installed ScanCode scanpipe wrapper at {wrapper_path}")
        return 0
    except (ConfigError, InstallError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
