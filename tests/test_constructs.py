"""Tests of constructs a backend builds with: QKV's spec with each one added, generated once."""

import os
import re
import subprocess

import pytest

import act_backend
from act_backend import generate_backend
from conftest import EXAMPLES_DIR, generate_oracle_and_backend, instruction_calls, run_backend

QKV_DIR = os.path.join(EXAMPLES_DIR, 'QKV')


def add_concatenate(acc):
    """An instruction that joins two halves of d1 into d2"""
    instr = acc.add_instruction('join', [], ['addr_1', 'addr_2', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_1'], ['32']], ['d1', ['@a.addr_2'], ['32']]])
    instr.set_outputs([['d2', ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY join {
    %In1 = bf16[32,64] parameter(0);
    %In2 = bf16[32,64] parameter(1);
    ROOT %Out0 = bf16[64,64] concatenate(%In1, %In2), dimensions={0};
}
""")


# A kernel that joins its two inputs and takes the softmax of the rows
JOINED = """HloModule joined, entry_computation_layout={(bf16[32,64]{1,0}, bf16[32,64]{1,0})->bf16[64,64]{1,0}}

%reduce_add (x: bf16[], y: bf16[]) -> bf16[] {
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}

ENTRY main {
  %a = bf16[32,64] parameter(0)
  %b = bf16[32,64] parameter(1)
  %joined = bf16[64,64] concatenate(%a, %b), dimensions={0}
  %exp = bf16[64,64] exponential(%joined)
  %zero = bf16[] constant(0)
  %sum = bf16[64] reduce(%exp, %zero), dimensions={1}, to_apply=%reduce_add
  %sums = bf16[64,64] broadcast(%sum), dimensions={0}
  ROOT %out = bf16[64,64] divide(%exp, %sums)
}
"""


def add_instruction_with_no_semantics(acc):
    """An instruction that sets a mode and computes nothing"""
    acc.add_instruction('configure', ['mode'], [])


@pytest.fixture(scope='module')
def work_dir(tmp_path_factory):
    """Generate QKV's backend with every construct above added to its accelerator"""
    work_dir = str(tmp_path_factory.mktemp('constructs'))

    def with_constructs(acc, *args, **kwargs):
        add_concatenate(acc)
        add_instruction_with_no_semantics(acc)
        return generate_backend(acc, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        # The spec calls generate_backend(qkv): add to the accelerator in that call
        patch.setattr(act_backend, 'generate_backend', with_constructs)
        generate_oracle_and_backend(QKV_DIR, work_dir)
    return work_dir


def test_backend_compiles_with_the_constructs(work_dir, tmp_path):
    # Every rule loads, and the kernel compiles as QKV's own does
    run, output_path = run_backend(
        os.path.join(work_dir, 'backends', 'QKV'), QKV_DIR, str(tmp_path)
    )
    assert run.returncode == 0, run.stderr
    assert sum(instruction_calls(output_path).values()) == 9


def test_concatenate_is_selected(work_dir, tmp_path):
    kernel = tmp_path / 'joined.hlo'
    kernel.write_text(JOINED)
    output = tmp_path / 'joined.py'
    backend = os.path.join(work_dir, 'backends', 'QKV')
    log = str(tmp_path / 'log')
    arguments = ['--input', str(kernel), '--output', str(output), '--log', log]
    run = subprocess.run([backend, *arguments], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    calls = [
        (name, dict(re.findall(r'(\w+) = (\d+)', arguments)))
        for name, arguments in re.findall(r'api\.(\w+)\(([^)]*)\)', output.read_text())
    ]
    # Each half is loaded, the two are joined in the kernel's order, and the rows normalised
    names = [name for name, _ in calls]
    assert names == ['load_rm', 'load_rm', 'join', 'softmax', 'mov', 'store_rm']
    loaded = {call['addr_in']: call['addr_out'] for name, call in calls if name == 'load_rm'}
    join = calls[2][1]
    assert (join['addr_1'], join['addr_2']) == (loaded['0'], loaded['4096'])


def test_instruction_with_no_semantics_is_left_out(work_dir):
    crate = os.path.join(work_dir, 'targets', 'QKV', 'backend')
    for source in [('src', 'ir', 'egraph.rs'), ('cpp', 'malloc', 'include', 'instructions.h')]:
        with open(os.path.join(crate, *source)) as f:
            assert 'Configure' not in f.read()
