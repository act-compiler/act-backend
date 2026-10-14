"""Tests of where an instruction is selected: a spec made for them, generated once."""

import os
import re
import subprocess

import pytest
from taidl import Accelerator

from act_backend import generate_backend


def add_moves(acc):
    """A load of bf16 rows into d1 and a store of them from it"""
    instr = acc.add_instruction('load_rm', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['@c.n * 128']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In0);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

    instr = acc.add_instruction('store_rm', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['@c.n']]])
    instr.set_outputs([['d0', ['@a.addr_out'], ['@c.n * 128']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In0);
    ROOT %Out0 = u8[`@c.n * 128`] reshape(%a);
}
""")


def add_on_d1(acc, name, sizes, lines, attributes=()):
    """An instruction from regions of d1 to a region of d1, the output's size last"""
    addresses = [f'addr_{number}' for number in range(1, len(sizes))]
    instr = acc.add_instruction(name, list(attributes), [*addresses, 'addr_out'])
    instr.set_inputs([['d1', [f'@a.{address}'], [size]] for address, size in zip(addresses, sizes)])
    instr.set_outputs([['d1', ['@a.addr_out'], [sizes[-1]]]])
    instr.add_semantics('ENTRY {\n' + ''.join(f'    {line};\n' for line in lines) + '}\n')


def matching():
    """One buffer of 512 rows, its moves, and the instructions the tests select"""
    acc = Accelerator('MATCHING')
    acc.add_data_model('d1', [512], [64], 'bf16')
    add_moves(acc)
    # The sum of each row, spread along it
    add_on_d1(
        acc,
        'rowsum',
        ['64', '64'],
        [
            '%In0 = bf16[64,64] parameter(0)',
            '%s = bf16[64] reduce_add(%In0), dimensions={1}',
            'ROOT %Out0 = bf16[64,64] broadcast(%s), dimensions={0}',
        ],
    )
    return acc


@pytest.fixture(scope='module')
def backend(tmp_path_factory):
    """Generate the spec's backend once and return the binary's path"""
    work_dir = tmp_path_factory.mktemp('matching')
    generate_backend(matching(), str(work_dir))
    return os.path.join(work_dir, 'backends', 'MATCHING')


def selected(backend, tmp_path, kernel):
    """Compile a kernel's text; return each call of its assembly, a name and its arguments"""
    path = tmp_path / 'kernel.hlo'
    path.write_text(kernel)
    output = tmp_path / 'kernel.py'
    arguments = ['--input', str(path), '--output', str(output), '--log', str(tmp_path / 'log')]
    run = subprocess.run([backend, *arguments], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert output.exists(), 'the compile found no assembly'
    return [
        (name, dict(re.findall(r'(\w+) = (\d+)', arguments)))
        for name, arguments in re.findall(r'api\.(\w+)\(([^)]*)\)', output.read_text())
    ]


# The sum of a reduce region
REDUCE_ADD = """%reduce_add (x: bf16[], y: bf16[]) -> bf16[] {
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}"""

# The rows' sums spread, then broadcast onto their shape
SPREAD_TWICE = f"""HloModule spread_twice, entry_computation_layout={{(bf16[64,64]{{1,0}})->bf16[64,64]{{1,0}}}}

{REDUCE_ADD}

ENTRY main {{
  %x = bf16[64,64] parameter(0)
  %zero = bf16[] constant(0)
  %s = bf16[64] reduce(%x, %zero), dimensions={{1}}, to_apply=%reduce_add
  %b = bf16[64,64] broadcast(%s), dimensions={{0}}
  ROOT %c = bf16[64,64] broadcast(%b), dimensions={{0,1}}
}}
"""


def test_identity_broadcast_is_dropped_onto_its_own_value(backend, tmp_path):
    calls = selected(backend, tmp_path, SPREAD_TWICE)
    # The identity broadcast is the spread rows, which rowsum computes
    assert [name for name, _ in calls] == ['load_rm', 'rowsum', 'store_rm']
