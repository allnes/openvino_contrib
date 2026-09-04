# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from ov_contrib_license.cli import EXIT_CONFIGURATION_ERROR, EXIT_SUCCESS, main


class CliTests(unittest.TestCase):
    def test_inventory_writes_canonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "requirements.txt").write_text("example==1\n", encoding="utf-8")
            output = root / "result.json"

            exit_code = main(["inventory", str(root), "--offline", "--output", str(output)])
            data = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, EXIT_SUCCESS)
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["discoveries"][0]["path"], "requirements.txt")

    def test_partial_incremental_configuration_has_distinct_exit_code(self) -> None:
        exit_code = main(["inventory", ".", "--base-ref", "HEAD~1"])

        self.assertEqual(exit_code, EXIT_CONFIGURATION_ERROR)


if __name__ == "__main__":
    unittest.main()
