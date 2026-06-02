#!/usr/bin/env python3
"""Prepare license-compliance scan inputs from repository YAML config."""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from license_config import ConfigError, load_config, require_list, require_str


class LicenseInputError(RuntimeError):
    """A user-facing license input preparation error."""


def run_git(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as exc:
        details = ""
        if capture:
            details = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
        raise LicenseInputError(f"git {' '.join(args)} failed.{chr(10) + details if details else ''}") from exc


def find_repo_root() -> Path:
    result = run_git(["rev-parse", "--show-toplevel"], capture=True)
    return Path(result.stdout.strip()).resolve()


def repo_path(repo_root: Path, config: dict[str, Any], *keys: str) -> Path:
    return repo_root / require_str(config, *keys)


def ensure_policy(repo_root: Path, config: dict[str, Any]) -> Path:
    policy = repo_path(repo_root, config, "paths", "scancode_policy_file")
    if not policy.is_file():
        raise LicenseInputError(f"Missing ScanCode policy file: {policy.relative_to(repo_root)}")
    return policy


def excluded_dir_names(config: dict[str, Any]) -> set[str]:
    return set(require_list(config, "copy_excludes", "directory_names"))


def excluded_dir_patterns(config: dict[str, Any]) -> list[str]:
    return require_list(config, "copy_excludes", "directory_patterns")


def is_excluded_directory(name: str, config: dict[str, Any]) -> bool:
    if name in excluded_dir_names(config):
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in excluded_dir_patterns(config))


def make_copytree_ignore(config: dict[str, Any]) -> Callable[[str, list[str]], set[str]]:
    def ignore(_: str, names: list[str]) -> set[str]:
        return {name for name in names if is_excluded_directory(name, config)}

    return ignore


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def copy_path(source: Path, destination: Path, config: dict[str, Any]) -> None:
    if destination.exists() or destination.is_symlink():
        remove_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_symlink():
        os.symlink(os.readlink(source), destination)
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=True, ignore=make_copytree_ignore(config))
    elif source.is_file():
        shutil.copy2(source, destination)


def copy_relative_path(repo_root: Path, relative_path: Path, destination_root: Path, config: dict[str, Any]) -> None:
    normalized = Path(str(relative_path).removeprefix("./"))
    source = repo_root / normalized
    if not source.exists() and not source.is_symlink():
        return
    copy_path(source, destination_root / normalized, config)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def validate_submodules(repo_root: Path) -> None:
    if not (repo_root / ".gitmodules").is_file():
        print("No .gitmodules file found; nothing to initialize.")
        return

    run_git(["submodule", "sync", "--recursive"])
    run_git(["submodule", "update", "--init", "--recursive"])
    status = run_git(["submodule", "status", "--recursive"], capture=True).stdout
    missing = [line for line in status.splitlines() if line.startswith("-")]
    if missing:
        joined = "\n".join(missing)
        raise LicenseInputError(f"One or more submodules are not initialized:\n{joined}")


def input_root(repo_root: Path, config: dict[str, Any], key: str) -> Path:
    return repo_path(repo_root, config, "paths", key)


def repository_input(input_directory: Path, config: dict[str, Any]) -> Path:
    return input_directory / require_str(config, "paths", "repository_input_dir")


def prepare_full_repo(repo_root: Path, config: dict[str, Any]) -> None:
    policy = ensure_policy(repo_root, config)
    input_directory = input_root(repo_root, config, "full_repo_input_dir")
    destination = repository_input(input_directory, config)
    if input_directory.exists():
        remove_path(input_directory)
    input_directory.mkdir(parents=True, exist_ok=True)

    shutil.copytree(repo_root, destination, symlinks=True, ignore=make_copytree_ignore(config))
    shutil.copy2(policy, input_directory / "policies.yml")
    print(f"Prepared full repository ScanCode input at {input_directory.relative_to(repo_root)}/")


def should_skip_walk_dir(path: Path, repo_root: Path, config: dict[str, Any]) -> bool:
    if is_excluded_directory(path.name, config):
        return True
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return True
    generated_inputs = {
        Path(require_str(config, "paths", "full_repo_input_dir")).parts[:1],
        Path(require_str(config, "paths", "thirdparty_input_dir")).parts[:1],
    }
    return relative.parts[:1] in generated_inputs


def find_vendored_roots(repo_root: Path, config: dict[str, Any]) -> list[Path]:
    vendored_names = set(require_list(config, "vendored", "directory_names"))
    roots: list[Path] = []
    for current, dirnames, _ in os.walk(repo_root):
        current_path = Path(current)
        kept_dirnames = []
        for dirname in sorted(dirnames):
            child = current_path / dirname
            if should_skip_walk_dir(child, repo_root, config):
                continue
            if dirname in vendored_names:
                roots.append(child.relative_to(repo_root))
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

    for root in require_list(config, "vendored", "repo_specific_roots"):
        root_path = Path(root)
        if (repo_root / root_path).is_dir():
            roots.append(root_path)

    return sorted(set(roots), key=lambda value: value.as_posix())


def prepare_thirdparty_focused(repo_root: Path, config: dict[str, Any]) -> None:
    policy = ensure_policy(repo_root, config)
    input_directory = input_root(repo_root, config, "thirdparty_input_dir")
    destination = repository_input(input_directory, config)
    if input_directory.exists():
        remove_path(input_directory)
    destination.mkdir(parents=True, exist_ok=True)

    shutil.copy2(policy, input_directory / "policies.yml")
    for root_file in require_list(config, "vendored", "root_license_files"):
        copy_relative_path(repo_root, Path(root_file), destination, config)

    vendored_roots = find_vendored_roots(repo_root, config)
    if not vendored_roots:
        print("No vendored roots found; focused input contains repository-level license documents only.")
    else:
        vendored_roots_file = input_directory / require_str(config, "paths", "vendored_roots_file")
        write_lines(vendored_roots_file, (path.as_posix() for path in vendored_roots))
        for root in vendored_roots:
            copy_relative_path(repo_root, root, destination, config)

    print(f"Prepared third-party ScanCode input at {input_directory.relative_to(repo_root)}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="License-compliance helper config YAML.")
    parser.add_argument(
        "mode",
        choices=("validate-submodules", "full-repo", "thirdparty-focused"),
        help="Preparation mode to run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo_root = find_repo_root()
        config = load_config(repo_root / args.config if not args.config.is_absolute() else args.config)
        if args.mode == "validate-submodules":
            validate_submodules(repo_root)
        elif args.mode == "full-repo":
            prepare_full_repo(repo_root, config)
        elif args.mode == "thirdparty-focused":
            prepare_thirdparty_focused(repo_root, config)
        else:
            raise LicenseInputError(f"Unsupported mode: {args.mode}")
        return 0
    except (ConfigError, LicenseInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
