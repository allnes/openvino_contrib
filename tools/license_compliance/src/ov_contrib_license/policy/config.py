# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ov_contrib_license.model import (
    Decision,
    DistributionStatus,
    Obligation,
    Relationship,
)


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Unable to read policy file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in policy file {path}: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Policy file {path} must contain a mapping")  # noqa: TRY004
    return data


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected a string or list of strings, got {value!r}")
    return tuple(value)


@dataclass(frozen=True)
class PolicyRule:
    id: str
    decision: Decision
    licenses: tuple[str, ...] = ()
    license_classes: tuple[str, ...] = ()
    relationships: tuple[Relationship, ...] = ()
    distributions: tuple[DistributionStatus, ...] = ()
    modules: tuple[str, ...] = ()
    artifact_scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExceptionRule:
    id: str
    component: str
    module: str
    relationships: tuple[Relationship, ...]
    distributions: tuple[DistributionStatus, ...]
    rationale: str
    approved_by: tuple[str, ...]
    expires: dt.date
    decision: Decision = Decision.PASS


@dataclass(frozen=True)
class BaselineEntry:
    finding_fingerprint: str
    reason: str


@dataclass(frozen=True)
class ToolchainPolicy:
    allowed_license_classes: tuple[str, ...] = ("permissive",)
    allowed_explicit_licenses: tuple[str, ...] = ()
    require_full_sha_for_github_actions: bool = True
    action_licenses: tuple[tuple[str, str], ...] = ()
    dependency_licenses: tuple[tuple[str, str], ...] = ()
    forbidden_services: tuple[tuple[str, str], ...] = ()

    @property
    def actions(self) -> dict[str, str]:
        return dict(self.action_licenses)

    @property
    def dependencies(self) -> dict[str, str]:
        return dict(self.dependency_licenses)


@dataclass(frozen=True)
class PolicyConfig:
    directory: Path
    license_classes: dict[str, frozenset[str]]
    rules: tuple[PolicyRule, ...]
    obligations_by_license: dict[str, tuple[Obligation, ...]]
    obligations_by_class: dict[str, tuple[Obligation, ...]]
    exceptions: tuple[ExceptionRule, ...]
    baseline: tuple[BaselineEntry, ...]
    toolchain: ToolchainPolicy
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def baseline_by_id(self) -> dict[str, BaselineEntry]:
        return {item.finding_fingerprint: item for item in self.baseline}


def _load_rules(data: dict[str, Any]) -> tuple[PolicyRule, ...]:
    rules: list[PolicyRule] = []
    for item in data.get("rules", ()):
        if not isinstance(item, dict) or not isinstance(item.get("when", {}), dict):
            raise ValueError(  # noqa: TRY004
                "Each policy rule must be a mapping with a 'when' mapping"
            )
        when = item.get("when", {})
        rules.append(
            PolicyRule(
                id=str(item["id"]),
                decision=Decision(str(item["decision"])),
                licenses=_strings(when.get("license")),
                license_classes=_strings(when.get("license_class")),
                relationships=tuple(
                    Relationship(value) for value in _strings(when.get("relationship"))
                ),
                distributions=tuple(
                    DistributionStatus(value)
                    for value in _strings(when.get("distribution"))
                ),
                modules=_strings(when.get("module")),
                artifact_scopes=_strings(when.get("artifact_scope")),
            )
        )
    identifiers = [item.id for item in rules]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Policy rule IDs must be unique")
    return tuple(rules)


def _load_exceptions(data: dict[str, Any]) -> tuple[ExceptionRule, ...]:
    result: list[ExceptionRule] = []
    for item in data.get("exceptions", ()):
        if not isinstance(item, dict):
            raise ValueError("Each exception must be a mapping")  # noqa: TRY004
        component = str(item.get("component", ""))
        if not component or "@" not in component:
            raise ValueError(
                "Exception component must identify an exact version/revision"
            )
        approved_by = _strings(item.get("approved_by"))
        if not approved_by:
            raise ValueError(
                f"Exception {item.get('id')} must contain approval metadata"
            )
        expires_value = item.get("expires")
        try:
            expires = (
                expires_value
                if isinstance(expires_value, dt.date)
                else dt.date.fromisoformat(str(expires_value))
            )
        except ValueError as error:
            raise ValueError(
                f"Exception {item.get('id')} has an invalid expiration date"
            ) from error
        relationships = tuple(
            Relationship(value) for value in _strings(item.get("allowed_relationships"))
        )
        distributions = tuple(
            DistributionStatus(value)
            for value in _strings(item.get("allowed_distribution"))
        )
        if not relationships or not distributions:
            raise ValueError(
                f"Exception {item.get('id')} must constrain relationship and distribution"
            )
        result.append(
            ExceptionRule(
                id=str(item["id"]),
                component=component,
                module=str(item.get("module", "")),
                relationships=relationships,
                distributions=distributions,
                rationale=str(item.get("rationale", "")),
                approved_by=approved_by,
                expires=expires,
                decision=Decision(str(item.get("decision", "PASS"))),
            )
        )
    return tuple(result)


def _load_obligations(
    data: dict[str, Any],
) -> tuple[dict[str, tuple[Obligation, ...]], dict[str, tuple[Obligation, ...]]]:
    def convert(values: Any) -> dict[str, tuple[Obligation, ...]]:
        if values is None:
            return {}
        if not isinstance(values, dict):
            raise ValueError("Obligation mappings must be mappings")  # noqa: TRY004
        return {
            str(key): tuple(Obligation(value) for value in _strings(item))
            for key, item in values.items()
        }

    return convert(data.get("licenses")), convert(data.get("classes"))


def _load_toolchain(data: dict[str, Any]) -> tuple[ToolchainPolicy, dict[str, Any]]:
    actions = data.get("actions", {})
    dependencies = data.get("dependencies", {})
    if not isinstance(actions, dict) or not isinstance(dependencies, dict):
        raise ValueError(  # noqa: TRY004
            "Toolchain action/dependency registries must be mappings"
        )
    forbidden: list[tuple[str, str]] = []
    for item in data.get("forbidden_services", ()):
        if not isinstance(item, dict):
            raise ValueError("Each forbidden service must be a mapping")  # noqa: TRY004
        forbidden.append(
            (str(item["id"]), str(item.get("reason", "forbidden by policy")))
        )
    providers = data.get("providers", {})
    return ToolchainPolicy(
        allowed_license_classes=_strings(data.get("allowed_license_classes"))
        or ("permissive",),
        allowed_explicit_licenses=_strings(data.get("allowed_explicit_licenses")),
        require_full_sha_for_github_actions=bool(
            data.get("require_full_sha_for_github_actions", True)
        ),
        action_licenses=tuple(
            sorted((str(key).lower(), str(value)) for key, value in actions.items())
        ),
        dependency_licenses=tuple(
            sorted(
                (str(key).lower(), str(value)) for key, value in dependencies.items()
            )
        ),
        forbidden_services=tuple(sorted(forbidden)),
    ), providers if isinstance(providers, dict) else {}


def load_policy(directory: Path) -> PolicyConfig:
    directory = directory.resolve()
    required = (
        "licenses.yml",
        "rules.yml",
        "obligations.yml",
        "exceptions.yml",
        "baseline.yml",
        "toolchain.yml",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(
            f"Policy directory {directory} is missing: {', '.join(missing)}"
        )

    licenses_data = _load_mapping(directory / "licenses.yml")
    classes_data = licenses_data.get("classes", {})
    if not isinstance(classes_data, dict):
        raise ValueError("licenses.yml 'classes' must be a mapping")  # noqa: TRY004
    classes = {
        str(key): frozenset(_strings(value)) for key, value in classes_data.items()
    }
    duplicate_licenses: dict[str, list[str]] = {}
    for class_name, values in classes.items():
        for license_id in values:
            duplicate_licenses.setdefault(license_id, []).append(class_name)
    duplicates = {
        key: value for key, value in duplicate_licenses.items() if len(value) > 1
    }
    if duplicates:
        raise ValueError(f"Licenses must belong to one class: {duplicates}")

    obligations = _load_obligations(_load_mapping(directory / "obligations.yml"))
    baseline_data = _load_mapping(directory / "baseline.yml")
    baseline = tuple(
        BaselineEntry(
            str(item["finding_fingerprint"]), str(item.get("reason", "baseline"))
        )
        for item in baseline_data.get("baseline", ())
    )
    toolchain, providers = _load_toolchain(_load_mapping(directory / "toolchain.yml"))
    return PolicyConfig(
        directory=directory,
        license_classes=classes,
        rules=_load_rules(_load_mapping(directory / "rules.yml")),
        obligations_by_license=obligations[0],
        obligations_by_class=obligations[1],
        exceptions=_load_exceptions(_load_mapping(directory / "exceptions.yml")),
        baseline=baseline,
        toolchain=toolchain,
        providers=providers,
    )


def load_baseline(path: Path) -> tuple[BaselineEntry, ...]:
    data = _load_mapping(path)
    return tuple(
        BaselineEntry(
            str(item["finding_fingerprint"]), str(item.get("reason", "baseline"))
        )
        for item in data.get("baseline", ())
    )
