"""End-to-end test for ACT-Backend using QKV ISA."""

import os
import tempfile
import pytest
from conftest import EXAMPLES_DIR, generate_oracle_and_backend, compile_hlo, run_compiled_kernel

QKV_DIR = os.path.join(EXAMPLES_DIR, "QKV")
WORK_DIR = tempfile.mkdtemp()

generate_oracle_and_backend(QKV_DIR, WORK_DIR)


@pytest.mark.incremental
class TestQKV:
    def test_backend_binary_exists(self):
        assert os.path.exists(os.path.join(WORK_DIR, "backends", "QKV"))

    def test_compile_hlo(self):
        output_path = compile_hlo(WORK_DIR, QKV_DIR)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_compiled_kernel_matches_golden(self):
        max_diff = run_compiled_kernel(WORK_DIR, os.path.join(QKV_DIR, "data"))
        assert max_diff == 0, f"Max diff: {max_diff}"
