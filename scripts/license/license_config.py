"""Load the license-compliance helper YAML config without extra dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """A user-facing configuration error."""


Line = tuple[int, str]


def parse_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def prepared_lines(text: str) -> list[Line]:
    lines: list[Line] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent % 2:
            raise ConfigError(f"invalid indentation at line {line_number}: use two-space indents")
        lines.append((indent, raw.strip()))
    return lines


def parse_node(lines: list[Line], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    current_indent, content = lines[index]
    if current_indent != indent:
        raise ConfigError(f"unexpected indentation near: {content}")
    if content.startswith("- "):
        return parse_list(lines, index, indent)
    return parse_mapping(lines, index, indent)


def parse_list(lines: list[Line], index: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent or not content.startswith("- "):
            break
        item = content[2:].strip()
        index += 1
        if item:
            result.append(parse_scalar(item))
        else:
            if index >= len(lines) or lines[index][0] <= indent:
                result.append({})
            else:
                child, index = parse_node(lines, index, lines[index][0])
                result.append(child)
    return result, index


def parse_mapping(lines: list[Line], index: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent or content.startswith("- "):
            break
        key, separator, value = content.partition(":")
        if not separator:
            raise ConfigError(f"expected mapping entry near: {content}")
        key = key.strip()
        if not key:
            raise ConfigError(f"empty mapping key near: {content}")
        value = value.strip()
        index += 1
        if value:
            result[key] = parse_scalar(value)
        elif index >= len(lines) or lines[index][0] <= indent:
            result[key] = {}
        else:
            child, index = parse_node(lines, index, lines[index][0])
            result[key] = child
    return result, index


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    lines = prepared_lines(path.read_text(encoding="utf-8"))
    if not lines:
        raise ConfigError(f"config file is empty: {path}")
    data, index = parse_node(lines, 0, lines[0][0])
    if index != len(lines):
        raise ConfigError(f"could not parse full config file: {path}")
    if not isinstance(data, dict):
        raise ConfigError(f"top-level config must be a mapping: {path}")
    return data


def require_mapping(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f"missing config mapping: {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, dict):
        raise ConfigError(f"config value must be a mapping: {'.'.join(keys)}")
    return value


def require_list(data: dict[str, Any], *keys: str) -> list[str]:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f"missing config list: {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"config value must be a list of strings: {'.'.join(keys)}")
    return value


def require_str(data: dict[str, Any], *keys: str) -> str:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f"missing config string: {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"config value must be a non-empty string: {'.'.join(keys)}")
    return value
