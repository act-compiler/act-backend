"""End-to-end test for ACT-Backend using QKV_new ISA."""

import os
import tempfile
import pytest
import numpy as np
import jax.numpy as jnp
from conftest import (
    EXAMPLES_DIR,
    generate_oracle_and_backend,
    unfilled_sources,
    compile_hlo,
    run_compiled_kernel,
    load_golden,
    attention_reference,
)

QKV_DIR = os.path.join(EXAMPLES_DIR, 'QKV_new')
DATA_DIR = os.path.join(QKV_DIR, 'data')
WORK_DIR = tempfile.mkdtemp()

generate_oracle_and_backend(QKV_DIR, WORK_DIR)


@pytest.mark.incremental
class TestQKVNew:
    def test_backend_binary_exists(self):
        assert os.path.exists(os.path.join(WORK_DIR, 'backends', 'QKV'))

    def test_backend_sources_filled(self):
        assert unfilled_sources(WORK_DIR) == []

    def test_compile_hlo(self):
        output_path = compile_hlo(WORK_DIR, QKV_DIR)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_compiled_kernel_matches_reference(self):
        output = run_compiled_kernel(WORK_DIR, DATA_DIR)
        reference, tol = attention_reference(DATA_DIR)
        max_diff = float(np.max(np.abs(np.asarray(output, np.float64) - reference)))
        assert max_diff <= tol, f'Max diff from reference: {max_diff} > {tol}'

    def test_compiled_kernel_matches_golden(self):
        output = run_compiled_kernel(WORK_DIR, DATA_DIR)
        golden = load_golden(DATA_DIR)
        max_diff = float(jnp.max(jnp.abs(output - golden)))
        assert max_diff == 0, f'Max diff: {max_diff}'
