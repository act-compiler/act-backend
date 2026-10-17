"""Tests of how a compile ends: what it says and the output it leaves, a spec made for them."""

import os
import subprocess

import pytest
from conftest import add_moves, add_on_d1
from taidl import Accelerator

from act_backend import generate_backend


def endings():
    """A buffer of 100 rows, its moves, and an add of two of its regions of 64 rows"""
    acc = Accelerator('ENDINGS')
    acc.add_data_model('d1', [100], [64], 'bf16')
    add_moves(acc)
    # Its two inputs alone hold more rows than the buffer
    add_on_d1(
        acc,
        'vadd',
        ['64', '64', '64'],
        [
            '%In0 = bf16[64,64] parameter(0)',
            '%In1 = bf16[64,64] parameter(1)',
            'ROOT %Out0 = bf16[64,64] add(%In0, %In1)',
        ],
    )
    return acc


@pytest.fixture(scope='module')
def backend(tmp_path_factory):
    """Generate the spec's backend once and return the binary's path"""
    work_dir = str(tmp_path_factory.mktemp('endings'))
    generate_backend(endings(), work_dir)
    return os.path.join(work_dir, 'backends', 'ENDINGS')


def compiled(backend, tmp_path, kernel):
    """Compile a kernel's text; return the run and the output's path"""
    path = tmp_path / 'kernel.hlo'
    path.write_text(kernel)
    output = tmp_path / 'kernel.py'
    arguments = ['--input', str(path), '--output', str(output), '--log', str(tmp_path / 'log')]
    return subprocess.run(
        [backend, *arguments], capture_output=True, text=True, check=False
    ), output


# A sum only the small buffer's add computes
ADD = """HloModule add, entry_computation_layout={(bf16[64,64]{1,0}, bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[64,64] parameter(0)
  %b = bf16[64,64] parameter(1)
  ROOT %c = bf16[64,64] add(%a, %b)
}
"""


def test_dropped_candidate_names_its_reason(backend, tmp_path):
    run, _ = compiled(backend, tmp_path, ADD)
    assert 'PII #0 dropped: the solver found no placement of its tensors' in run.stdout
