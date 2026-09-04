# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ov_contrib_license.model import Inventory, InventoryBuilder, RepositoryInfo

from .cmake import discover_cmake
from .common import owning_scope
from .downloads import discover_downloads
from .github_actions import discover_github_actions
from .manifests import discover_manifests
from .submodules import discover_submodules
from .vendored import discover_vendored_trees

DEFAULT_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)


class DiscoveryError(RuntimeError):
    """Raised when deterministic repository discovery cannot be completed."""


@dataclass(frozen=True)
class DiscoveryOptions:
    base_ref: str | None = None
    head_ref: str | None = None
    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    offline: bool = False

    def validate(self) -> None:
        if bool(self.base_ref) != bool(self.head_ref):
            raise ValueError("--base-ref and --head-ref must be provided together")


def _run_git(root: Path, arguments: list[str], *, required: bool) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as error:
        if required:
            raise DiscoveryError(f"Unable to execute Git: {error}") from error
        return None
    if result.returncode:
        if required:
            reason = result.stderr.strip() or f"exit status {result.returncode}"
            raise DiscoveryError(f"Git {' '.join(arguments)} failed: {reason}")
        return None
    return result.stdout


def git_revision(root: Path) -> str | None:
    output = _run_git(root, ["rev-parse", "HEAD"], required=False)
    return output.strip() if output else None


def changed_scopes(root: Path, base_ref: str, head_ref: str) -> tuple[str, ...]:
    output = _run_git(
        root,
        ["diff", "--name-only", "--diff-filter=ACMRTUXB", base_ref, head_ref, "--"],
        required=True,
    )
    assert output is not None
    return tuple(sorted({owning_scope(line.strip()) for line in output.splitlines() if line.strip()}))


def _is_ignored_directory(path: str) -> bool:
    return any(part in DEFAULT_IGNORED_DIRECTORY_NAMES for part in PurePosixPath(path).parts[:-1])


def _matches(path: str, patterns: Iterable[str]) -> bool:
    pure = PurePosixPath(path)
    return any(fnmatch.fnmatchcase(path, pattern) or pure.match(pattern) for pattern in patterns)


def _selected(path: str, options: DiscoveryOptions, scopes: tuple[str, ...] | None) -> bool:
    if _is_ignored_directory(path):
        return False
    if scopes is not None and owning_scope(path) not in scopes:
        return False
    if options.includes and not _matches(path, options.includes):
        return False
    return not _matches(path, options.excludes)


def repository_files(
    root: Path, options: DiscoveryOptions, scopes: tuple[str, ...] | None
) -> tuple[str, ...]:
    output = _run_git(root, ["ls-files", "-z", "--cached", "--others", "--exclude-standard"], required=False)
    if output is not None:
        candidates = output.split("\0")
    else:
        candidates = [item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()]
    return tuple(sorted(path for path in candidates if path and _selected(path, options, scopes)))


def read_text(root: Path, relative_path: str, *, size_limit: int = 4 * 1024 * 1024) -> str | None:
    path = root / relative_path
    try:
        if not path.is_file() or path.stat().st_size > size_limit:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


class RepositoryDiscovery:
    def __init__(self, root: Path, options: DiscoveryOptions | None = None) -> None:
        self.root = root.resolve()
        self.options = options or DiscoveryOptions()

    def run(self) -> Inventory:
        self.options.validate()
        if not self.root.is_dir():
            raise ValueError(f"Repository path is not a directory: {self.root}")

        scopes: tuple[str, ...] | None = None
        if self.options.base_ref and self.options.head_ref:
            scopes = changed_scopes(self.root, self.options.base_ref, self.options.head_ref)
        files = repository_files(self.root, self.options, scopes)
        repository = RepositoryInfo(
            path=".",
            revision=git_revision(self.root),
            base_ref=self.options.base_ref,
            head_ref=self.options.head_ref,
            impacted_scopes=scopes or (),
        )
        builder = InventoryBuilder(repository)

        discover_manifests(builder, files)
        discover_cmake(builder, self.root, files)
        discover_downloads(builder, self.root, files)
        discover_submodules(builder, self.root, files)
        discover_vendored_trees(builder, self.root, files)
        discover_github_actions(builder, self.root, files)
        return builder.build()
