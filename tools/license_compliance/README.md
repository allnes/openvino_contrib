<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO Contrib License Compliance

`ov-contrib-license` builds a deterministic inventory of third-party software
entry points in an `openvino_contrib` checkout. It is repository tooling and is
deliberately independent of the OpenVINO build and runtime.

This initial implementation is the discovery-only first stage of the compliance
design. It records evidence and uncertainty; it does not make license
compatibility decisions. Policy evaluation, provider adapters, TPP
reconciliation, and artifact verification belong to later reviewable changes.

## Install and run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install ./tools/license_compliance
.venv/bin/ov-contrib-license inventory . --output inventory.json
```

The command has no runtime dependencies outside the Python standard library and
does not access the network. JSON is the canonical output. A human-readable YAML
representation is also available:

```bash
.venv/bin/ov-contrib-license inventory . --format yaml
```

For a pull-request-sized inventory, provide both Git revisions. Discovery is
expanded from changed files to their owning module or repository scope:

```bash
.venv/bin/ov-contrib-license inventory . \
  --base-ref origin/master \
  --head-ref HEAD \
  --output inventory.json
```

Additional `--include` and `--exclude` globs may be repeated. Paths are always
reported relative to the repository root. Canonical JSON contains no timestamps
and has stable ordering.

## What is discovered

- Python, Node.js, Go, Gradle, Cargo, Conan, Docker, and Git manifests;
- balanced, multiline CMake dependency commands without evaluating conditions;
- executable `git clone`, `curl`, `wget`, and remote package-install commands;
- Git submodules, including their index commit when available;
- conventional vendored-source directory names, nested license files, and
  foreign-copyright clusters;
- local, container, and repository-backed GitHub Actions.

Unresolved CMake revisions and mutable GitHub Action refs are retained as
explicit `REVIEW` discovery findings. They are data-quality findings, not legal
policy verdicts.

## Development

The offline test suite uses only `unittest`:

```bash
.venv/bin/python -m unittest discover -s tools/license_compliance/tests -v
```

The project is licensed under Apache-2.0, like the containing repository. See
the repository root `LICENSE` file.
