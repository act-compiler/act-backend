"""End-to-end test for ACT-Backend using QKV ISA."""

import os
import pytest
import numpy as np
import jax.numpy as jnp
from conftest import EXAMPLES_DIR, unfilled_sources, load_golden, attention_reference

QKV_DIR = os.path.join(EXAMPLES_DIR, 'QKV')
DATA_DIR = os.path.join(QKV_DIR, 'data')


@pytest.mark.incremental
class TestQKV:
    def test_backend_sources_filled(self, work_dir):
        assert unfilled_sources(work_dir) == []

    def test_compile_hlo(self, assembly):
        assert os.path.exists(assembly)
        assert os.path.getsize(assembly) > 0

    def test_compiled_kernel_matches_reference(self, output):
        reference, tol = attention_reference(DATA_DIR)
        max_diff = float(np.max(np.abs(np.asarray(output, np.float64) - reference)))
        assert max_diff <= tol, f'Max diff from reference: {max_diff} > {tol}'

    def test_compiled_kernel_matches_golden(self, output):
        golden, record = load_golden(DATA_DIR)
        max_diff = float(jnp.max(jnp.abs(output - golden)))
        assert max_diff == 0, f'Max diff: {max_diff} (golden from {record})'
