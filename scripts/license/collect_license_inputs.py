#!/usr/bin/env python3
"""Prepare license-compliance scan inputs.

This script keeps repository selection logic out of GitHub Actions YAML. It
intentionally does not exclude vendored source roots from ScanCode inputs.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT_LICENSE_FILES = ("LICENSE", "NOTICE", "third-party-programs.txt")
VENDORED_DIR_NAMES = {
    "thirdparty",
    "3rdparty",
    "external",
    "vendor",
    "vendors",
    "deps",
    "submodules",
}
REPO_SPECIFIC_VENDORED_ROOTS = (
    Path("modules/ollama_openvino/llama/llama.cpp"),
    Path("modules/ollama_openvino/ml/backend/ggml/ggml"),
)

EXCLUDED_DIR_NAMES = {
    ".git",
    ".scancodeio",
    "license-inputs",
    "scancode-inputs",
    "scancode-thirdparty-inputs",
    "build",
    ".cache",
    ".ccache",
    "__pycache__",
    "node_modules",
}
EXCLUDED_DIR_PATTERNS = ("cmake-build-*",)


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


def ensure_policy(repo_root: Path) -> None:
    policy = repo_root / ".github/scancode/policies.yml"
    if not policy.is_file():
        raise LicenseInputError("Missing .github/scancode/policies.yml")


def is_excluded_directory(name: str) -> bool:
    return name in EXCLUDED_DIR_NAMES or any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_DIR_PATTERNS)


def copytree_ignore(_: str, names: list[str]) -> set[str]:
    return {name for name in names if is_excluded_directory(name)}


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def copy_path(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        remove_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_symlink():
        os.symlink(os.readlink(source), destination)
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=True, ignore=copytree_ignore)
    elif source.is_file():
        shutil.copy2(source, destination)


def copy_relative_path(repo_root: Path, relative_path: Path, destination_root: Path) -> None:
    normalized = Path(str(relative_path).removeprefix("./"))
    source = repo_root / normalized
    if not source.exists() and not source.is_symlink():
        return
    copy_path(source, destination_root / normalized)


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


def should_skip_openvino_header_file(path: str) -> bool:
    parts = Path(path).parts
    if any(part in VENDORED_DIR_NAMES for part in parts):
        return True
    return (
        path.startswith("modules/ollama_openvino/llama/llama.cpp/")
        or path.startswith("modules/ollama_openvino/ml/backend/ggml/ggml/")
    )


def write_openvino_header_files(repo_root: Path) -> None:
    result = run_git(["ls-files"], capture=True)
    files = [
        path
        for path in result.stdout.splitlines()
        if path and not should_skip_openvino_header_file(path)
    ]
    destination = repo_root / "license-inputs/openvino-header-files.txt"
    write_lines(destination, files)
    print(f"Wrote {destination.relative_to(repo_root)}")


def prepare_full_repo(repo_root: Path) -> None:
    ensure_policy(repo_root)
    input_root = repo_root / "scancode-inputs"
    repository_input = input_root / "repository"
    remove_path(input_root) if input_root.exists() else None
    input_root.mkdir(parents=True, exist_ok=True)

    shutil.copytree(repo_root, repository_input, symlinks=True, ignore=copytree_ignore)
    shutil.copy2(repo_root / ".github/scancode/policies.yml", input_root / "policies.yml")
    print(f"Prepared full repository ScanCode input at {input_root.relative_to(repo_root)}/")


def should_skip_walk_dir(path: Path, repo_root: Path) -> bool:
    if is_excluded_directory(path.name):
        return True
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return True
    return relative.parts[:1] in {
        ("scancode-inputs",),
        ("scancode-thirdparty-inputs",),
        ("license-inputs",),
    }


def find_vendored_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    for current, dirnames, _ in os.walk(repo_root):
        current_path = Path(current)
        kept_dirnames = []
        for dirname in sorted(dirnames):
            child = current_path / dirname
            if should_skip_walk_dir(child, repo_root):
                continue
            if dirname in VENDORED_DIR_NAMES:
                roots.append(child.relative_to(repo_root))
                continue
            kept_dirnames.append(dirname)
        dirnames[:] = kept_dirnames

    for root in REPO_SPECIFIC_VENDORED_ROOTS:
        if (repo_root / root).is_dir():
            roots.append(root)

    return sorted(set(roots), key=lambda value: value.as_posix())


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def prepare_thirdparty_focused(repo_root: Path) -> None:
    ensure_policy(repo_root)
    input_root = repo_root / "scancode-thirdparty-inputs"
    repository_input = input_root / "repository"
    remove_path(input_root) if input_root.exists() else None
    repository_input.mkdir(parents=True, exist_ok=True)

    shutil.copy2(repo_root / ".github/scancode/policies.yml", input_root / "policies.yml")
    for root_file in ROOT_LICENSE_FILES:
        copy_relative_path(repo_root, Path(root_file), repository_input)

    vendored_roots = find_vendored_roots(repo_root)
    if not vendored_roots:
        print("No vendored roots found; focused input contains repository-level license documents only.")
    else:
        write_lines(input_root / "vendored-roots.txt", (path.as_posix() for path in vendored_roots))
        for root in vendored_roots:
            copy_relative_path(repo_root, root, repository_input)

    print(f"Prepared third-party ScanCode input at {input_root.relative_to(repo_root)}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("validate-submodules", "openvino-header-files", "full-repo", "thirdparty-focused"),
        help="Preparation mode to run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo_root = find_repo_root()
        if args.mode == "validate-submodules":
            validate_submodules(repo_root)
        elif args.mode == "openvino-header-files":
            write_openvino_header_files(repo_root)
        elif args.mode == "full-repo":
            prepare_full_repo(repo_root)
        elif args.mode == "thirdparty-focused":
            prepare_thirdparty_focused(repo_root)
        else:
            raise LicenseInputError(f"Unsupported mode: {args.mode}")
        return 0
    except LicenseInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
