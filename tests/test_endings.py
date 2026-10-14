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


# A sum of an input and an undefined value
UNDEFINED = """HloModule undefined, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[64,64] parameter(0)
  ROOT %c = bf16[64,64] add(%a, %b)
}
"""

# One exponential, which no instruction computes
EXPONENTIAL = """HloModule exponential, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[64,64] parameter(0)
  ROOT %e = bf16[64,64] exponential(%a)
}
"""


def test_dropped_candidate_names_its_reason(backend, tmp_path):
    run, _ = compiled(backend, tmp_path, ADD)
    assert 'PII #0 dropped: the solver found no placement of its tensors' in run.stdout


def test_no_assembly_ends_in_one_line_and_status_1(backend, tmp_path):
    run, output = compiled(backend, tmp_path, ADD)
    assert run.returncode == 1
    assert run.stderr == (
        'Error: no assembly found: the time limit of 10s ran out, and every candidate was '
        'dropped, the first because the solver found no placement of its tensors\n'
    )
    assert not output.exists()


def test_no_assembly_says_no_candidate_was_extracted(backend, tmp_path):
    run, output = compiled(backend, tmp_path, EXPONENTIAL)
    assert run.returncode == 1
    assert run.stderr == (
        'Error: no assembly found: the time limit of 10s ran out, and no candidate was extracted\n'
    )
    assert not output.exists()


def test_failed_compile_leaves_no_earlier_output(backend, tmp_path):
    (tmp_path / 'kernel.py').write_text('# an earlier compile\n')
    run, output = compiled(backend, tmp_path, UNDEFINED)
    assert run.returncode != 0
    assert not output.exists()


def test_usage_error_leaves_the_earlier_output(backend, tmp_path):
    (tmp_path / 'kernel.py').write_text('# an earlier compile\n')
    (tmp_path / 'kernel.hlo').write_text(UNDEFINED)
    # No directory after --log, a usage error
    command = ['--input', str(tmp_path / 'kernel.hlo'), '--output', str(tmp_path / 'kernel.py')]
    run = subprocess.run([backend, *command, '--log'], capture_output=True, text=True, check=False)
    assert run.returncode == 1
    assert (tmp_path / 'kernel.py').read_text() == '# an earlier compile\n'
