"""End-to-end test for ACT-Backend using QKV_new ISA."""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    EXAMPLES_DIR,
    attention_reference,
    instruction_calls,
    load_golden,
    unfilled_sources,
)

QKV_DIR = os.path.join(EXAMPLES_DIR, 'QKV_new')
DATA_DIR = os.path.join(QKV_DIR, 'data')


@pytest.mark.incremental
class TestQKVNew:
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


# Outside the incremental class: counted even when an output test fails
def test_compiled_kernel_calls(assembly):
    # The cheapest selection the backend finds for the attention kernel
    assert instruction_calls(assembly) == {
        'load_01': 1,
        'load_03': 2,
        'transpose_13': 1,
        'gemm_33': 2,
        'softmax': 1,
        'mov_21': 1,
        'mov_23': 1,
        'store_10': 1,
    }
