#!/usr/bin/env python3
"""Create a GitHub Markdown summary from ScanCode.io JSON outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from license_config import ConfigError, load_config, require_list, require_str


def load_json_reports(search_root: Path) -> list[tuple[Path, Any]]:
    reports = []
    for path in sorted(search_root.rglob("*.json")):
        if path.is_dir():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                reports.append((path, json.load(handle)))
        except (OSError, json.JSONDecodeError):
            continue
    return reports


def iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for item in value:
            yield from iter_dicts(item)


def as_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(as_text_values(item))
        return values
    if isinstance(value, dict):
        values = []
        for key in ("key", "name", "spdx_license_key", "license_expression", "license-expression"):
            values.extend(as_text_values(value.get(key)))
        return values
    return []


def license_texts(item: dict[str, Any]) -> list[str]:
    keys = [
        "detected_license_expression",
        "detected_license_expression_spdx",
        "license_expression",
        "license_expression_spdx",
        "declared_license_expression",
        "declared_license_expression_spdx",
        "other_license_expression",
        "license_expressions",
        "license_detections",
        "licenses",
    ]
    values: list[str] = []
    for key in keys:
        values.extend(as_text_values(item.get(key)))
    return [value.strip() for value in values if value and value.strip()]


def compliance_alerts(item: dict[str, Any]) -> list[str]:
    alerts = []
    for key in ("compliance_alert", "license_clarity_compliance_alert", "scorecard_compliance_alert"):
        value = item.get(key)
        if isinstance(value, str) and value:
            alerts.append(value.lower())
    for license_item in item.get("licenses", []) or []:
        if isinstance(license_item, dict):
            policy = license_item.get("policy") or {}
            alert = policy.get("compliance_alert")
            if isinstance(alert, str) and alert:
                alerts.append(alert.lower())
    return alerts


def is_resource(item: dict[str, Any]) -> bool:
    return isinstance(item.get("path"), str) and item.get("type") != "directory"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def classify_license_text(text: str, config: dict[str, Any]) -> str | None:
    normalized = normalize(text)
    unknown_patterns = require_list(config, "scancode_summary", "unknown_patterns")
    prohibited_patterns = require_list(config, "scancode_summary", "prohibited_patterns")
    restricted_patterns = require_list(config, "scancode_summary", "restricted_patterns")
    exact_prohibited = require_list(config, "scancode_summary", "exact_prohibited")
    ambiguous_marker = require_str(config, "scancode_summary", "ambiguous_marker")

    if any(pattern in normalized for pattern in unknown_patterns):
        return "unknown"
    if ambiguous_marker in normalized:
        return "ambiguous"
    if any(pattern in normalized for pattern in prohibited_patterns):
        return "prohibited"
    if normalized in exact_prohibited:
        return "prohibited"
    if any(pattern in normalized for pattern in restricted_patterns):
        return "restricted"
    return None


def summarize_reports(reports: list[tuple[Path, Any]], strict_thirdparty: bool, config: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "json_reports": [str(path) for path, _ in reports],
        "files": 0,
        "packages": 0,
        "dependencies": 0,
        "alerts": [],
        "vulnerabilities": 0,
    }

    seen_resources = set()
    seen_packages = set()
    seen_dependencies = set()

    for _, data in reports:
        for item in iter_dicts(data):
            path = item.get("path")
            if is_resource(item) and path not in seen_resources:
                seen_resources.add(path)
                summary["files"] += 1
                licenses = license_texts(item)
                if strict_thirdparty and not licenses:
                    summary["alerts"].append(("no license", path, "no detected license"))
                for license_text in licenses:
                    kind = classify_license_text(license_text, config)
                    if kind:
                        summary["alerts"].append((kind, path, license_text))

            purl = item.get("purl") or item.get("package_url")
            if isinstance(purl, str) and purl not in seen_packages:
                seen_packages.add(purl)
                summary["packages"] += 1

            dependency_uid = item.get("dependency_uid") or item.get("extracted_requirement")
            if isinstance(dependency_uid, str) and dependency_uid not in seen_dependencies:
                seen_dependencies.add(dependency_uid)
                summary["dependencies"] += 1

            for alert in compliance_alerts(item):
                if alert in {"error", "warning", "missing"}:
                    label = f"compliance {alert}"
                    location = path or item.get("name") or purl or "<project>"
                    summary["alerts"].append((label, str(location), "ScanCode compliance alert"))

            vulnerabilities = item.get("vulnerabilities") or item.get("affected_by_vulnerabilities")
            if isinstance(vulnerabilities, list):
                summary["vulnerabilities"] += len(vulnerabilities)
            elif vulnerabilities:
                summary["vulnerabilities"] += 1

    unique_alerts = []
    seen_alerts = set()
    for alert in summary["alerts"]:
        if alert in seen_alerts:
            continue
        seen_alerts.add(alert)
        unique_alerts.append(alert)
    summary["alerts"] = unique_alerts
    return summary


def write_markdown(
    args: argparse.Namespace,
    summary: dict[str, Any],
    report_found: bool,
    config: dict[str, Any],
) -> str:
    lines = [
        f"## {args.title}",
        "",
        f"- Artifact: `{args.artifact_name}`",
        f"- JSON reports discovered: `{len(summary['json_reports'])}`",
        f"- Scanned files reported: `{summary['files']}`",
        f"- Packages reported: `{summary['packages']}`",
        f"- Dependencies reported: `{summary['dependencies']}`",
        f"- Vulnerabilities reported: `{summary['vulnerabilities']}`",
        "",
        f"Report formats requested from ScanCode: {require_str(config, 'scancode_summary', 'report_formats')}.",
        "",
    ]

    if summary["json_reports"]:
        lines.append("JSON report paths:")
        for path in summary["json_reports"][:10]:
            lines.append(f"- `{path}`")
        if len(summary["json_reports"]) > 10:
            lines.append(f"- ... {len(summary['json_reports']) - 10} more")
        lines.append("")
    elif not report_found:
        lines.extend(require_list(config, "scancode_summary", "no_report_message"))
        lines.append("")

    alerts = summary["alerts"]
    if alerts:
        lines.append(require_str(config, "scancode_summary", "finding_heading"))
        for kind, location, detail in alerts[:50]:
            lines.append(f"- `{kind}` at `{location}`: {detail}")
        if len(alerts) > 50:
            lines.append(f"- ... {len(alerts) - 50} more findings")
        lines.append("")
    else:
        lines.append(require_str(config, "scancode_summary", "no_findings_message"))
        lines.append("")

    lines.append("Failure handling:")
    lines.extend(f"- {line}" for line in require_list(config, "scancode_summary", "failure_handling"))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path, help="License-compliance helper config YAML.")
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--strict-thirdparty", action="store_true")
    parser.add_argument("--require-report", action="store_true")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        reports = load_json_reports(args.search_root)
        summary = summarize_reports(reports, args.strict_thirdparty, config)
        markdown = write_markdown(args, summary, bool(reports), config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
            handle.write("\n")
    else:
        print(markdown)

    if args.require_report and not reports:
        return 1
    if args.strict_thirdparty and summary["alerts"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
