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
    # A product through a transpose, both inputs declared first
    add_on_d1(
        acc,
        'turn_dot',
        ['32', '32', '64'],
        [
            '%In0 = bf16[32,64] parameter(0)',
            '%In1 = bf16[32,64] parameter(1)',
            '%t = bf16[64,32] transpose(%In0), dimensions={1,0}',
            (
                'ROOT %Out0 = bf16[64,64] dot(%t, %In1), lhs_contracting_dims={1}, '
                'rhs_contracting_dims={0}'
            ),
        ],
    )
    # Sums of 32 rows, a value named like a copy
    add_on_d1(
        acc,
        'rowsum_32',
        ['32', '32'],
        [
            '%In0 = bf16[32,64] parameter(0)',
            '%s_dup = bf16[32] reduce_add(%In0), dimensions={1}',
            'ROOT %Out0 = bf16[32,64] broadcast(%s_dup), dimensions={0}',
        ],
    )
    return acc


def add_on_hbm(acc, name, sizes, lines, attributes=()):
    """An instruction from regions of the HBM to a region of it, the output's size last"""
    addresses = [f'addr_{number}' for number in range(1, len(sizes))]
    instr = acc.add_instruction(name, list(attributes), [*addresses, 'addr_out'])
    instr.set_inputs([['d0', [f'@a.{address}'], [size]] for address, size in zip(addresses, sizes)])
    instr.set_outputs([['d0', ['@a.addr_out'], [sizes[-1]]]])
    instr.add_semantics('ENTRY {\n' + ''.join(f'    {line};\n' for line in lines) + '}\n')


def hbm_only():
    """No data model but the HBM: a product and a sum, each reading its inputs' bytes"""
    acc = Accelerator('HBM_ONLY')
    add_on_hbm(
        acc,
        'gemm',
        ['8192', '8192', '8192'],
        [
            '%In0 = u8[8192] parameter(0)',
            '%In1 = u8[8192] parameter(1)',
            '%a = u8[64,64,2] reshape(%In0)',
            '%b = u8[64,64,2] reshape(%In1)',
            '%x = bf16[64,64] bitcast_convert(%a)',
            '%y = bf16[64,64] bitcast_convert(%b)',
            '%z = bf16[64,64] dot(%x, %y), lhs_contracting_dims={1}, rhs_contracting_dims={0}',
            '%c = u8[64,64,2] bitcast_convert(%z)',
            'ROOT %Out0 = u8[8192] reshape(%c)',
        ],
    )
    add_on_hbm(
        acc,
        'vadd',
        ['@c.n * 128', '@c.n * 128', '@c.n * 128'],
        [
            '%In0 = u8[`@c.n * 128`] parameter(0)',
            '%In1 = u8[`@c.n * 128`] parameter(1)',
            '%a = u8[`@c.n`,64,2] reshape(%In0)',
            '%b = u8[`@c.n`,64,2] reshape(%In1)',
            '%x = bf16[`@c.n`,64] bitcast_convert(%a)',
            '%y = bf16[`@c.n`,64] bitcast_convert(%b)',
            '%z = bf16[`@c.n`,64] add(%x, %y)',
            '%c = u8[`@c.n`,64,2] bitcast_convert(%z)',
            'ROOT %Out0 = u8[`@c.n * 128`] reshape(%c)',
        ],
        ['n'],
    )
    return acc


@pytest.fixture(scope='module')
def work_dir(tmp_path_factory):
    """Generate the spec's backend once, in a directory of its own"""
    work_dir = str(tmp_path_factory.mktemp('matching'))
    generate_backend(matching(), work_dir)
    return work_dir


@pytest.fixture(scope='module')
def backend(work_dir):
    return os.path.join(work_dir, 'backends', 'MATCHING')


@pytest.fixture(scope='module')
def hbm_backend(tmp_path_factory):
    """Generate the backend of the spec with the HBM alone and return the binary's path"""
    work_dir = str(tmp_path_factory.mktemp('hbm_only'))
    generate_backend(hbm_only(), work_dir)
    return os.path.join(work_dir, 'backends', 'HBM_ONLY')


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


# A library caller running the rewrites twice on fresh kernels
TWICE = """use backend::isel;
use egg::Runner;

fn main() {
  let kernel = std::env::args().nth(1).unwrap();
  for _ in 0..2 {
    let (egraph, ..) = isel::initializer::parse_hlo_module_to_egraph(&kernel).unwrap();
    let read = egraph.total_number_of_nodes();
    let rewrites = isel::rewrites::get_rewrites();
    let runner = Runner::default().with_egraph(egraph).with_iter_limit(3).run(&rewrites);
    println!("{} {}", read, runner.egraph.total_number_of_nodes());
  }
}
"""


def test_library_matches_as_much_in_a_second_run(work_dir, tmp_path):
    crate = os.path.join(work_dir, 'targets', 'MATCHING', 'backend')
    os.makedirs(os.path.join(crate, 'examples'), exist_ok=True)
    with open(os.path.join(crate, 'examples', 'twice.rs'), 'w') as f:
        f.write(TWICE)
    kernel = tmp_path / 'kernel.hlo'
    kernel.write_text(SPREAD_TWICE)
    command = ['cargo', 'run', '--release', '--locked', '--example', 'twice', str(kernel)]
    run = subprocess.run(command, cwd=crate, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    # Both runs grow the kernel to the same node count
    read, first, _, second = (int(count) for count in run.stdout.split())
    assert first > read
    assert second == first


# A signed-byte sum, scalar and matrix constants written alike
SAME_TEXT = """HloModule same_text, entry_computation_layout={(s8[16,16]{1,0})->s8[16,16]{1,0}}

%reduce_add (x: s8[], y: s8[]) -> s8[] {
  %x = s8[] parameter(0)
  %y = s8[] parameter(1)
  ROOT %sum = s8[] add(%x, %y)
}

ENTRY main {
  %x = s8[16,16] parameter(0)
  %one = s8[16,16] constant(1)
  %a = s8[16,16] add(%one, %x)
  %init = s8[] constant(1)
  %r = s8[16] reduce(%a, %init), dimensions={1}, to_apply=%reduce_add
  ROOT %b = s8[16,16] broadcast(%r), dimensions={0}
}
"""


def test_rewrite_that_joins_unlike_values_ends_the_compile(backend, tmp_path):
    path = tmp_path / 'kernel.hlo'
    path.write_text(SAME_TEXT)
    output = tmp_path / 'kernel.py'
    arguments = ['--input', str(path), '--output', str(output), '--log', str(tmp_path / 'log')]
    run = subprocess.run([backend, *arguments], capture_output=True, text=True, check=False)
    # Both constants one scalar e-node, the sum widened over it
    assert run.returncode != 0
    assert 'internal error: a rewrite joins the values s8[16, 16] and s8[1]' in run.stderr
    assert not output.exists()


# One input's transpose times the other
TURNED = """HloModule turned, entry_computation_layout={(bf16[32,64]{1,0}, bf16[32,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[32,64] parameter(0)
  %b = bf16[32,64] parameter(1)
  %t = bf16[64,32] transpose(%a), dimensions={1,0}
  ROOT %c = bf16[64,64] dot(%t, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
"""


def test_semantics_not_in_pattern_order_is_selected(backend, tmp_path):
    calls = selected(backend, tmp_path, TURNED)
    assert [name for name, _ in calls] == ['load_rm', 'load_rm', 'turn_dot', 'store_rm']
    # The transposed input comes first, as in the kernel
    loaded = {call['addr_in']: call['addr_out'] for name, call in calls if name == 'load_rm'}
    turned = calls[2][1]
    assert (turned['addr_1'], turned['addr_2']) == (loaded['0'], loaded['4096'])


# The rows' sums of 32 rows, spread along them
SPREAD_32 = f"""HloModule spread_32, entry_computation_layout={{(bf16[32,64]{{1,0}})->bf16[32,64]{{1,0}}}}

{REDUCE_ADD}

ENTRY main {{
  %x = bf16[32,64] parameter(0)
  %zero = bf16[] constant(0)
  %s = bf16[32] reduce(%x, %zero), dimensions={{1}}, to_apply=%reduce_add
  ROOT %b = bf16[32,64] broadcast(%s), dimensions={{0}}
}}
"""


def test_value_named_as_a_copy_is_selected(backend, tmp_path):
    calls = selected(backend, tmp_path, SPREAD_32)
    assert [name for name, _ in calls] == ['load_rm', 'rowsum_32', 'store_rm']


# A product and its sum, each value in the HBM
PRODUCT_SUM = """HloModule product_sum, entry_computation_layout={(bf16[64,64]{1,0}, bf16[64,64]{1,0}, bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[64,64] parameter(0)
  %b = bf16[64,64] parameter(1)
  %c = bf16[64,64] parameter(2)
  %p = bf16[64,64] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %s = bf16[64,64] add(%p, %c)
}
"""


def test_accelerator_with_no_on_chip_data_model_compiles(hbm_backend, tmp_path):
    calls = selected(hbm_backend, tmp_path, PRODUCT_SUM)
    assert [name for name, _ in calls] == ['gemm', 'vadd']
    # The sum reads the product's output and the third input
    (_, product), (_, total) = calls
    assert (total['addr_1'], total['addr_2']) == (product['addr_out'], '16384')
