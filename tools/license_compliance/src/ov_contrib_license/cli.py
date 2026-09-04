# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ov_contrib_license.discovery import DiscoveryOptions, RepositoryDiscovery
from ov_contrib_license.discovery.repository import DiscoveryError
from ov_contrib_license.reporters import write_inventory

EXIT_SUCCESS = 0
EXIT_CONFIGURATION_ERROR = 4
EXIT_DISCOVERY_ERROR = 5
EXIT_INTERNAL_ERROR = 6


class ConfigurationError(ValueError):
    pass


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ConfigurationError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(prog="ov-contrib-license", description="OpenVINO contrib license discovery")
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=ArgumentParser)
    inventory = subparsers.add_parser("inventory", help="build a deterministic repository inventory")
    inventory.add_argument(
        "path", nargs="?", default=".", help="repository path (default: current directory)"
    )
    inventory.add_argument("--output", type=Path, help="write the inventory to this file")
    inventory.add_argument("--format", dest="output_format", choices=("json", "yaml"), default="json")
    inventory.add_argument("--base-ref", help="base Git revision for incremental discovery")
    inventory.add_argument("--head-ref", help="head Git revision for incremental discovery")
    inventory.add_argument(
        "--offline", action="store_true", help="forbid network providers (discovery is offline)"
    )
    inventory.add_argument("--include", action="append", default=[], metavar="GLOB")
    inventory.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    return parser


def _run_inventory(arguments: argparse.Namespace) -> int:
    options = DiscoveryOptions(
        base_ref=arguments.base_ref,
        head_ref=arguments.head_ref,
        includes=tuple(arguments.include),
        excludes=tuple(arguments.exclude),
        offline=arguments.offline,
    )
    inventory = RepositoryDiscovery(Path(arguments.path), options).run()
    write_inventory(inventory, arguments.output, arguments.output_format)
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
        if arguments.command == "inventory":
            return _run_inventory(arguments)
        raise ConfigurationError(f"Unsupported command: {arguments.command}")
    except (ConfigurationError, ValueError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIGURATION_ERROR
    except DiscoveryError as error:
        print(f"discovery error: {error}", file=sys.stderr)
        return EXIT_DISCOVERY_ERROR
    except OSError as error:
        print(f"discovery I/O error: {error}", file=sys.stderr)
        return EXIT_DISCOVERY_ERROR
    except Exception as error:  # pragma: no cover - final CLI containment
        print(f"internal error: {error}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
