"""End-to-end test for ACT-Backend using mini_trn ISA."""

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

EXAMPLE_DIR = os.path.join(EXAMPLES_DIR, 'mini_trn')
DATA_DIR = os.path.join(EXAMPLE_DIR, 'data')


@pytest.mark.incremental
class TestMiniTrn:
    def test_backend_binary_exists(self, work_dir):
        assert os.path.exists(os.path.join(work_dir, 'backends', 'mini_trn'))

    def test_backend_sources_filled(self, work_dir):
        assert unfilled_sources(work_dir, EXAMPLE_DIR) == []

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


# Outside the incremental class: counted even if outputs fail
def test_compiled_kernel_calls(assembly):
    # The cheapest selection the backend finds for the attention kernel
    assert instruction_calls(assembly) == {
        'load_rm': 2,
        'load_cm': 1,
        'gemm': 2,
        'softmax': 1,
        'mov': 2,
        'store_rm': 1,
    }
