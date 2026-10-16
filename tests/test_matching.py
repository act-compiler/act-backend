"""Tests of where an instruction is selected: a spec made for them, generated once."""

import importlib.util
import os
import re
import subprocess
from collections import Counter

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import add_moves, add_on_d1, import_oracle
from taidl import Accelerator
from taidl_to import generate_oracle

from act_backend import generate_backend


def add_typed_moves(acc, element_type, size):
    """A load of rows of one element type into d1 and a store of them, each element size bytes"""
    last = '' if size == 1 else f',{size}'
    instr = acc.add_instruction(f'load_{element_type}', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], [f'@c.n * {64 * size}']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[`@c.n * {64 * size}`] parameter(0);
    %a = u8[`@c.n`,64{last}] reshape(%In0);
    ROOT %Out0 = {element_type}[`@c.n`,64] bitcast_convert(%a);
}}
""")

    instr = acc.add_instruction(f'store_{element_type}', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['@c.n']]])
    instr.set_outputs([['d0', ['@a.addr_out'], [f'@c.n * {64 * size}']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = {element_type}[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64{last}] bitcast_convert(%In0);
    ROOT %Out0 = u8[`@c.n * {64 * size}`] reshape(%a);
}}
""")


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
    # One input read 27 times, by 26 sums
    add_on_d1(
        acc,
        'sums',
        ['64', '64'],
        [
            '%In0 = bf16[64,64] parameter(0)',
            '%a0 = bf16[64,64] add(%In0, %In0)',
            *[f'%a{k} = bf16[64,64] add(%a{k - 1}, %In0)' for k in range(1, 25)],
            'ROOT %Out0 = bf16[64,64] add(%a24, %In0)',
        ],
    )
    # The input added to itself, twice
    add_on_d1(
        acc,
        'four',
        ['64', '64'],
        [
            '%In0 = bf16[64,64] parameter(0)',
            '%a = bf16[64,64] add(%In0, %In0)',
            'ROOT %Out0 = bf16[64,64] add(%a, %a)',
        ],
    )
    # Softmax, and the operations that spell it out
    add_on_d1(
        acc,
        'softmax',
        ['@c.n', '@c.n'],
        [
            '%In0 = bf16[`@c.n`,64] parameter(0)',
            '%a = bf16[`@c.n`,64] exponential(%In0)',
            '%reduced = bf16[`@c.n`] reduce_add(%a), dimensions={1}',
            '%b = bf16[`@c.n`,64] broadcast(%reduced), dimensions={0}',
            'ROOT %Out0 = bf16[`@c.n`,64] divide(%a, %b)',
        ],
        ['n'],
    )
    add_on_d1(
        acc,
        'vexp',
        ['@c.n', '@c.n'],
        [
            '%In0 = bf16[`@c.n`,64] parameter(0)',
            'ROOT %Out0 = bf16[`@c.n`,64] exponential(%In0)',
        ],
        ['n'],
    )
    add_on_d1(
        acc,
        'vdiv',
        ['@c.n', '@c.n', '@c.n'],
        [
            '%In0 = bf16[`@c.n`,64] parameter(0)',
            '%In1 = bf16[`@c.n`,64] parameter(1)',
            'ROOT %Out0 = bf16[`@c.n`,64] divide(%In0, %In1)',
        ],
        ['n'],
    )
    # A load spelling its attribute in its result's second dimension
    instr = acc.add_instruction('load_cm', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['@c.n * 128']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In0);
    %b = bf16[`@c.n`,64] bitcast_convert(%a);
    ROOT %Out0 = bf16[64,`@c.n`] transpose(%b), dimensions={1,0};
}
""")
    # A store of the exponentials of that shape
    instr = acc.add_instruction('store_exp', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['64']]])
    instr.set_outputs([['d0', ['@a.addr_out'], ['@c.n * 128']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,`@c.n`] parameter(0);
    %e = bf16[64,`@c.n`] exponential(%In0);
    %a = u8[64,`@c.n`,2] bitcast_convert(%e);
    ROOT %Out0 = u8[`@c.n * 128`] reshape(%a);
}
""")
    # The first n of m rows
    add_on_d1(
        acc,
        'top',
        ['@c.m', '@c.n'],
        [
            '%In0 = bf16[`@c.m`,64] parameter(0)',
            'ROOT %Out0 = bf16[`@c.n`,64] slice(%In0), slice={[0:`@c.n`], [0:64]}',
        ],
        ['m', 'n'],
    )
    # The last n of m rows
    add_on_d1(
        acc,
        'bottom',
        ['@c.m', '@c.n'],
        [
            '%In0 = bf16[`@c.m`,64] parameter(0)',
            'ROOT %Out0 = bf16[`@c.n`,64] slice(%In0), slice={[`@c.m - @c.n`:`@c.m`], [0:64]}',
        ],
        ['m', 'n'],
    )
    # The 16 rows from row s, which no shape names
    add_on_d1(
        acc,
        'window',
        ['64', '16'],
        [
            '%In0 = bf16[64,64] parameter(0)',
            'ROOT %Out0 = bf16[16,64] slice(%In0), slice={[`@c.s`:`@c.s + 16`], [0:64]}',
        ],
        ['s'],
    )
    # A tile, m rows of n f32: loaded, exponentiated, stored
    instr = acc.add_instruction('load_tile', ['m', 'n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['@c.m * @c.n * 4']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['@c.m']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.m * @c.n * 4`] parameter(0);
    %a = u8[`@c.m`,`@c.n`,4] reshape(%In0);
    ROOT %Out0 = f32[`@c.m`,`@c.n`] bitcast_convert(%a);
}
""")
    add_on_d1(
        acc,
        'exp_tile',
        ['@c.m', '@c.m'],
        [
            '%In0 = f32[`@c.m`,`@c.n`] parameter(0)',
            'ROOT %Out0 = f32[`@c.m`,`@c.n`] exponential(%In0)',
        ],
        ['m', 'n'],
    )
    instr = acc.add_instruction('store_tile', ['m', 'n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['@c.m']]])
    instr.set_outputs([['d0', ['@a.addr_out'], ['@c.m * @c.n * 4']]])
    instr.add_semantics("""
ENTRY {
    %In0 = f32[`@c.m`,`@c.n`] parameter(0);
    %a = u8[`@c.m`,`@c.n`,4] bitcast_convert(%In0);
    ROOT %Out0 = u8[`@c.m * @c.n * 4`] reshape(%a);
}
""")
    # Rows of s8 and of s32, narrowed and widened
    add_typed_moves(acc, 's8', 1)
    add_typed_moves(acc, 's32', 4)
    for name, source, result in [('narrow', 's32', 's8'), ('widen', 's8', 's32')]:
        add_on_d1(
            acc,
            name,
            ['@c.n', '@c.n'],
            [
                f'%In0 = {source}[`@c.n`,64] parameter(0)',
                f'ROOT %Out0 = {result}[`@c.n`,64] convert(%In0)',
            ],
            ['n'],
        )
    return acc


def spreading():
    """The moves and a load of a vector in the HBM spread over 64 rows, for the oracle too"""
    acc = Accelerator('SPREADING')
    acc.add_data_model('d1', [128], [64], 'bf16')
    add_moves(acc)
    instr = acc.add_instruction('load_spread', [], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['128']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[128] parameter(0);
    %a = u8[64,2] reshape(%In0);
    %b = bf16[64] bitcast_convert(%a);
    ROOT %Out0 = bf16[64,64] broadcast(%b), dimensions={1};
}
""")
    return acc


def add_on_hbm(acc, name, sizes, lines, attributes=()):
    """An instruction from regions of the HBM to a region of it, the output's size last"""
    addresses = [f'addr_{number}' for number in range(1, len(sizes))]
    instr = acc.add_instruction(name, list(attributes), [*addresses, 'addr_out'])
    instr.set_inputs([['d0', [f'@a.{address}'], [size]] for address, size in zip(addresses, sizes)])
    instr.set_outputs([['d0', ['@a.addr_out'], [sizes[-1]]]])
    instr.add_semantics('ENTRY {\n' + ''.join(f'    {line};\n' for line in lines) + '}\n')


def hbm_only():
    """No data model but the HBM: products, a sum and a fill, each reading HBM bytes"""
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
    # A product of each pair of 32 by 32 blocks
    add_on_hbm(
        acc,
        'bgemm',
        ['4096', '4096', '4096'],
        [
            '%In0 = u8[4096] parameter(0)',
            '%In1 = u8[4096] parameter(1)',
            '%a = u8[2,32,32,2] reshape(%In0)',
            '%b = u8[2,32,32,2] reshape(%In1)',
            '%x = bf16[2,32,32] bitcast_convert(%a)',
            '%y = bf16[2,32,32] bitcast_convert(%b)',
            (
                '%z = bf16[2,32,32] dot(%x, %y), lhs_batch_dims={0}, rhs_batch_dims={0}, '
                'lhs_contracting_dims={2}, rhs_contracting_dims={1}'
            ),
            '%c = u8[2,32,32,2] bitcast_convert(%z)',
            'ROOT %Out0 = u8[4096] reshape(%c)',
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
    # A product over the first two dimensions of each input
    add_on_hbm(
        acc,
        'cdot',
        ['8192', '8192', '8192'],
        [
            '%In0 = u8[8192] parameter(0)',
            '%In1 = u8[8192] parameter(1)',
            '%a = u8[8,8,64,2] reshape(%In0)',
            '%b = u8[8,8,64,2] reshape(%In1)',
            '%x = bf16[8,8,64] bitcast_convert(%a)',
            '%y = bf16[8,8,64] bitcast_convert(%b)',
            '%z = bf16[64,64] dot(%x, %y), lhs_contracting_dims={1,0}, rhs_contracting_dims={1,0}',
            '%c = u8[64,64,2] bitcast_convert(%z)',
            'ROOT %Out0 = u8[8192] reshape(%c)',
        ],
    )
    # A product whose result shape disagrees with its dimensions
    add_on_hbm(
        acc,
        'skew',
        ['8192', '8192', '2'],
        [
            '%In0 = u8[8192] parameter(0)',
            '%In1 = u8[8192] parameter(1)',
            '%a = u8[64,64,2] reshape(%In0)',
            '%b = u8[64,64,2] reshape(%In1)',
            '%x = bf16[64,64] bitcast_convert(%a)',
            '%y = bf16[64,64] bitcast_convert(%b)',
            (
                '%z = bf16[1] dot(%x, %y), lhs_batch_dims={0}, lhs_contracting_dims={1}, '
                'rhs_batch_dims={0}, rhs_contracting_dims={1}'
            ),
            '%c = u8[1,2] bitcast_convert(%z)',
            'ROOT %Out0 = u8[2] reshape(%c)',
        ],
    )
    # One bf16 value spread over 64 by 64
    add_on_hbm(
        acc,
        'fill',
        ['2', '8192'],
        [
            '%In0 = u8[2] parameter(0)',
            '%a = u8[1,2] reshape(%In0)',
            '%x = bf16[1] bitcast_convert(%a)',
            '%b = bf16[64,64] broadcast(%x), dimensions={0}',
            '%c = u8[64,64,2] bitcast_convert(%b)',
            'ROOT %Out0 = u8[8192] reshape(%c)',
        ],
    )
    return acc


def scalar():
    """No data model but the HBM: an add of two bf16 values of one element each, for the oracle"""
    acc = Accelerator('SCALAR')
    add_on_hbm(
        acc,
        'sadd',
        ['2', '2', '2'],
        [
            '%In0 = u8[2] parameter(0)',
            '%In1 = u8[2] parameter(1)',
            '%a = u8[1,2] reshape(%In0)',
            '%b = u8[1,2] reshape(%In1)',
            '%x = bf16[1] bitcast_convert(%a)',
            '%y = bf16[1] bitcast_convert(%b)',
            '%z = bf16[1] add(%x, %y)',
            '%c = u8[1,2] bitcast_convert(%z)',
            'ROOT %Out0 = u8[2] reshape(%c)',
        ],
    )
    return acc


def through_product():
    """Moves of rows of 16 s8, a 16 by 16 product the only way from d1 to d2, and an add on d2"""
    acc = Accelerator('THROUGH_PRODUCT')
    acc.add_data_model('d1', [64], [16], 's8')
    acc.add_data_model('d2', [64], [16], 's8')
    instr = acc.add_instruction('load_s8', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['@c.n * 16']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 16`] parameter(0);
    %a = u8[`@c.n`,16] reshape(%In0);
    ROOT %Out0 = s8[`@c.n`,16] bitcast_convert(%a);
}
""")
    instr = acc.add_instruction('store_s8', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d2', ['@a.addr_in'], ['@c.n']]])
    instr.set_outputs([['d0', ['@a.addr_out'], ['@c.n * 16']]])
    instr.add_semantics("""
ENTRY {
    %In0 = s8[`@c.n`,16] parameter(0);
    %a = u8[`@c.n`,16] bitcast_convert(%In0);
    ROOT %Out0 = u8[`@c.n * 16`] reshape(%a);
}
""")
    instr = acc.add_instruction('gemm16', [], ['addr_1', 'addr_2', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_1'], ['16']], ['d1', ['@a.addr_2'], ['16']]])
    instr.set_outputs([['d2', ['@a.addr_out'], ['16']]])
    instr.add_semantics("""
ENTRY {
    %In0 = s8[16,16] parameter(0);
    %In1 = s8[16,16] parameter(1);
    ROOT %Out0 = s8[16,16] dot(%In0, %In1), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")
    instr = acc.add_instruction('vadd16', [], ['addr_1', 'addr_2', 'addr_out'])
    instr.set_inputs([['d2', ['@a.addr_1'], ['16']], ['d2', ['@a.addr_2'], ['16']]])
    instr.set_outputs([['d2', ['@a.addr_out'], ['16']]])
    instr.add_semantics("""
ENTRY {
    %In0 = s8[16,16] parameter(0);
    %In1 = s8[16,16] parameter(1);
    ROOT %Out0 = s8[16,16] add(%In0, %In1);
}
""")
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
def spreading_dir(tmp_path_factory):
    """Generate the oracle and the backend of the spec that spreads a vector"""
    work_dir = str(tmp_path_factory.mktemp('spreading'))
    acc = spreading()
    generate_oracle(acc, work_dir)
    generate_backend(acc, work_dir)
    return work_dir


@pytest.fixture(scope='module')
def scalar_dir(tmp_path_factory):
    """Generate the oracle and the backend of the spec that adds scalars"""
    work_dir = str(tmp_path_factory.mktemp('scalar'))
    acc = scalar()
    generate_oracle(acc, work_dir)
    generate_backend(acc, work_dir)
    return work_dir


@pytest.fixture(scope='module')
def product_backend(tmp_path_factory):
    """Generate the backend of the spec whose only way from d1 to d2 is a product; return its path"""
    work_dir = str(tmp_path_factory.mktemp('through_product'))
    generate_backend(through_product(), work_dir)
    return os.path.join(work_dir, 'backends', 'THROUGH_PRODUCT')


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


def simulated(work_dir, name, assembly, *inputs):
    """Run a compiled kernel on the oracle of the accelerator named and return its output"""
    decorator, api = import_oracle(work_dir, os.path.join(work_dir, name))
    spec = importlib.util.spec_from_file_location('kernel', assembly)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (function,) = re.findall(r'^def (\w+)\(kernel, api\):', assembly.read_text(), re.MULTILINE)
    kernel = getattr(module, function)(decorator.kernel, api)
    kernel('fsim-compile')()
    outputs, _ = kernel('fsim')(*inputs)
    return outputs[0]


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


# One bf16 value spread over 64 by 64, from a value of rank 0 and of one element
FILL_FROM_RANK_0 = """HloModule fill_from_rank_0, entry_computation_layout={(bf16[])->bf16[64,64]{1,0}}

ENTRY main {
  %s = bf16[] parameter(0)
  ROOT %b = bf16[64,64] broadcast(%s), dimensions={}
}
"""

FILL_FROM_ONE_ELEMENT = """HloModule fill_from_one_element, entry_computation_layout={(bf16[1]{0})->bf16[64,64]{1,0}}

ENTRY main {
  %s = bf16[1] parameter(0)
  ROOT %b = bf16[64,64] broadcast(%s), dimensions={0}
}
"""


@pytest.mark.parametrize(
    'kernel', [FILL_FROM_RANK_0, FILL_FROM_ONE_ELEMENT], ids=['rank_0', 'one_element']
)
def test_broadcast_of_one_element_is_selected_whatever_its_map(hbm_backend, tmp_path, kernel):
    calls = selected(hbm_backend, tmp_path, kernel)
    # The semantics spreads a bf16[1] with dimensions={0}
    assert [name for name, _ in calls] == ['fill']


def test_broadcast_of_rank_0_with_no_map_compiles_as_with_the_empty_map(hbm_backend, tmp_path):
    # XLA-HLO lets a rank-0 value's broadcast give no dimensions
    unmapped = FILL_FROM_RANK_0.replace(', dimensions={}', '')
    assert 'dimensions' not in unmapped
    assemblies = []
    for name, kernel in [('mapped', FILL_FROM_RANK_0), ('unmapped', unmapped)]:
        (tmp_path / name).mkdir()
        selected(hbm_backend, tmp_path / name, kernel)
        text = (tmp_path / name / 'kernel.py').read_text()
        # The assembly, its header comments aside, which name the input file
        assemblies.append([line for line in text.splitlines() if not line.startswith('#')])
    assert assemblies[0] == assemblies[1]


# A product over the first dimension of each input, each value in the HBM
TURNED_PRODUCT = """HloModule turned_product, entry_computation_layout={(bf16[64,64]{1,0}, bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[64,64] parameter(0)
  %b = bf16[64,64] parameter(1)
  ROOT %p = bf16[64,64] dot(%a, %b), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
"""


def contracted(lhs, rhs):
    """A product of two 8 by 8 by 64 values over their first two dimensions, paired as given"""
    return f"""HloModule contracted, entry_computation_layout={{(bf16[8,8,64]{{2,1,0}}, bf16[8,8,64]{{2,1,0}})->bf16[64,64]{{1,0}}}}

ENTRY main {{
  %x = bf16[8,8,64] parameter(0)
  %y = bf16[8,8,64] parameter(1)
  ROOT %z = bf16[64,64] dot(%x, %y), lhs_contracting_dims={{{lhs}}}, rhs_contracting_dims={{{rhs}}}
}}
"""


def test_dot_is_selected_by_its_contracting_pairs(hbm_backend, tmp_path):
    calls = selected(hbm_backend, tmp_path, contracted('1,0', '1,0'))
    # The semantics pairs them as {1,0} with {1,0}
    assert [name for name, _ in calls] == ['cdot']


# A full contraction whose integers, in order, are skew's, in other lists
LISTS_APART = """HloModule lists_apart, entry_computation_layout={(bf16[64,64]{1,0}, bf16[64,64]{1,0})->bf16[]}

ENTRY main {
  %a = bf16[64,64] parameter(0)
  %b = bf16[64,64] parameter(1)
  ROOT %p = bf16[] dot(%a, %b), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
}
"""


# A product of each pair of blocks, each value in the HBM
BATCHED = """HloModule batched, entry_computation_layout={(bf16[2,32,32]{2,1,0}, bf16[2,32,32]{2,1,0})->bf16[2,32,32]{2,1,0}}

ENTRY main {
  %a = bf16[2,32,32] parameter(0)
  %b = bf16[2,32,32] parameter(1)
  ROOT %p = bf16[2,32,32] dot(%a, %b), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
}
"""


def test_batched_dot_is_selected(hbm_backend, tmp_path):
    calls = selected(hbm_backend, tmp_path, BATCHED)
    assert [name for name, _ in calls] == ['bgemm']


# A batched product over the last dimension of both blocks, each value in the HBM
BATCHED_ROWS = """HloModule batched_rows, entry_computation_layout={(bf16[2,32,32]{2,1,0}, bf16[2,32,32]{2,1,0})->bf16[2,32,32]{2,1,0}}

ENTRY main {
  %a = bf16[2,32,32] parameter(0)
  %b = bf16[2,32,32] parameter(1)
  ROOT %p = bf16[2,32,32] dot(%a, %b), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={2}
}
"""


@pytest.mark.parametrize(
    'kernel',
    [TURNED_PRODUCT, contracted('0,1', '1,0'), LISTS_APART, BATCHED_ROWS],
    ids=['turned', 'crossed_pairs', 'lists_apart', 'batched_rows'],
)
def test_dot_whose_dimensions_no_instruction_spells_finds_no_assembly(
    hbm_backend, tmp_path, kernel
):
    path = tmp_path / 'kernel.hlo'
    path.write_text(kernel)
    output = tmp_path / 'kernel.py'
    arguments = ['--input', str(path), '--output', str(output), '--log', str(tmp_path / 'log')]
    run = subprocess.run([hbm_backend, *arguments], capture_output=True, text=True, check=False)
    assert run.returncode == 1
    assert run.stderr.endswith('and no candidate was extracted\n')
    assert not output.exists()


# A product of two s8 values over dimension 1 of each
TURNED_SQUARE = """HloModule turned_square, entry_computation_layout={(s8[16,16]{1,0}, s8[16,16]{1,0})->s8[16,16]{1,0}}

ENTRY main {
  %a = s8[16,16] parameter(0)
  %b = s8[16,16] parameter(1)
  ROOT %p = s8[16,16] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
"""


def test_dot_is_compared_as_matched_not_as_another_of_its_class(product_backend, tmp_path):
    path = tmp_path / 'kernel.hlo'
    path.write_text(TURNED_SQUARE)
    output = tmp_path / 'kernel.py'
    arguments = ['--input', str(path), '--output', str(output), '--log', str(tmp_path / 'log')]
    run = subprocess.run([product_backend, *arguments], capture_output=True, text=True, check=False)
    # Its e-class also holds its product by the identity, which gemm16 spells
    assert run.returncode == 1
    assert run.stderr.endswith('and no candidate was extracted\n')
    assert not output.exists()


# The input added to itself and each sum: 26 sums
SUMS = (
    """HloModule sums, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  %a0 = bf16[64,64] add(%x, %x)
"""
    + ''.join(f'  %a{k} = bf16[64,64] add(%a{k - 1}, %x)\n' for k in range(1, 25))
    + """  ROOT %out = bf16[64,64] add(%a24, %x)
}
"""
)


def test_semantics_with_27_variables_is_selected(backend, tmp_path):
    calls = selected(backend, tmp_path, SUMS)
    assert [name for name, _ in calls] == ['load_rm', 'sums', 'store_rm']


# Four times the input, by two sums
FOUR = """HloModule four, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  %a = bf16[64,64] add(%x, %x)
  ROOT %b = bf16[64,64] add(%a, %a)
}
"""


def test_copy_of_a_copy_is_selected(backend, tmp_path):
    calls = selected(backend, tmp_path, FOUR)
    assert [name for name, _ in calls] == ['load_rm', 'four', 'store_rm']


# One input's exponentials over the rows' sums of the other's
SKEWED = f"""HloModule skewed, entry_computation_layout={{(bf16[64,64]{{1,0}}, bf16[64,64]{{1,0}})->bf16[64,64]{{1,0}}}}

{REDUCE_ADD}

ENTRY main {{
  %x = bf16[64,64] parameter(0)
  %y = bf16[64,64] parameter(1)
  %ex = bf16[64,64] exponential(%x)
  %ey = bf16[64,64] exponential(%y)
  %zero = bf16[] constant(0)
  %s = bf16[64] reduce(%ey, %zero), dimensions={{1}}, to_apply=%reduce_add
  %b = bf16[64,64] broadcast(%s), dimensions={{0}}
  ROOT %out = bf16[64,64] divide(%ex, %b)
}}
"""

# The rows' softmax: the exponentials, one value, read twice
SOFTMAX = f"""HloModule softmax, entry_computation_layout={{(bf16[64,64]{{1,0}})->bf16[64,64]{{1,0}}}}

{REDUCE_ADD}

ENTRY main {{
  %x = bf16[64,64] parameter(0)
  %ex = bf16[64,64] exponential(%x)
  %zero = bf16[] constant(0)
  %s = bf16[64] reduce(%ex, %zero), dimensions={{1}}, to_apply=%reduce_add
  %b = bf16[64,64] broadcast(%s), dimensions={{0}}
  ROOT %out = bf16[64,64] divide(%ex, %b)
}}
"""


def test_value_used_twice_matches_one_value(backend, tmp_path):
    calls = selected(backend, tmp_path, SKEWED)
    # Two exponentials, not softmax, which reads one value twice
    names = Counter(name for name, _ in calls)
    assert names == {'load_rm': 2, 'vexp': 2, 'rowsum': 1, 'vdiv': 1, 'store_rm': 1}


def test_softmax_is_selected_over_one_value(backend, tmp_path):
    calls = selected(backend, tmp_path, SOFTMAX)
    assert [name for name, _ in calls] == ['load_rm', 'softmax', 'store_rm']


# The exponentials of the transpose of 32 rows
TURNED_EXP = """HloModule turned_exp, entry_computation_layout={(bf16[32,64]{1,0})->bf16[64,32]{1,0}}

ENTRY main {
  %k = bf16[32,64] parameter(0)
  %t = bf16[64,32] transpose(%k), dimensions={1,0}
  ROOT %e = bf16[64,32] exponential(%t)
}
"""


def test_attribute_is_read_from_the_dimension_that_holds_it(backend, tmp_path):
    calls = selected(backend, tmp_path, TURNED_EXP)
    # Each attribute in its result's second dimension, 32 there
    assert calls == [
        ('load_cm', {'n': '32', 'addr_in': '0', 'addr_out': '0'}),
        ('store_exp', {'n': '32', 'addr_in': '0', 'addr_out': '4096'}),
    ]


# The first 32 of 64 rows
TOP = """HloModule top, entry_computation_layout={(bf16[64,64]{1,0})->bf16[32,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  ROOT %y = bf16[32,64] slice(%x), slice={[0:32], [0:64]}
}
"""


def test_two_attributes_are_carried_into_the_assembly(backend, tmp_path):
    calls = selected(backend, tmp_path, TOP)
    assert [name for name, _ in calls] == ['load_rm', 'top', 'store_rm']
    # Each attribute by name, from the dimension holding it
    assert {name: calls[1][1][name] for name in ['m', 'n']} == {'m': '64', 'n': '32'}


# The last 32 of 64 rows
BOTTOM = """HloModule bottom, entry_computation_layout={(bf16[64,64]{1,0})->bf16[32,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  ROOT %y = bf16[32,64] slice(%x), slice={[32:64], [0:64]}
}
"""


def test_slice_is_selected_by_its_ranges(backend, tmp_path):
    calls = selected(backend, tmp_path, BOTTOM)
    assert [name for name, _ in calls] == ['load_rm', 'bottom', 'store_rm']
    assert {name: calls[1][1][name] for name in ['m', 'n']} == {'m': '64', 'n': '32'}


# The 16 rows from row 8 of 64
WINDOW = """HloModule window, entry_computation_layout={(bf16[64,64]{1,0})->bf16[16,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  ROOT %y = bf16[16,64] slice(%x), slice={[8:24], [0:64]}
}
"""


def test_attribute_no_shape_names_is_read_from_its_slice(backend, tmp_path):
    calls = selected(backend, tmp_path, WINDOW)
    assert [name for name, _ in calls] == ['load_rm', 'window', 'store_rm']
    assert calls[1][1]['s'] == '8'


# The middle 32 of 64 rows
MIDDLE = """HloModule middle, entry_computation_layout={(bf16[64,64]{1,0})->bf16[32,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  ROOT %y = bf16[32,64] slice(%x), slice={[16:48], [0:64]}
}
"""


def test_slice_of_ranges_no_instruction_spells_selects_none(backend, tmp_path):
    calls = selected(backend, tmp_path, MIDDLE)
    # Neither top nor bottom, which slice other rows
    assert not {'top', 'bottom'} & {name for name, _ in calls}


# The exponentials of a transpose that keeps each dimension
KEPT_EXP = """HloModule kept_exp, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %k = bf16[64,64] parameter(0)
  %t = bf16[64,64] transpose(%k), dimensions={0,1}
  ROOT %e = bf16[64,64] exponential(%t)
}
"""

# The columns' sums, spread along each row
COLUMN_SUM = f"""HloModule column_sum, entry_computation_layout={{(bf16[64,64]{{1,0}})->bf16[64,64]{{1,0}}}}

{REDUCE_ADD}

ENTRY main {{
  %x = bf16[64,64] parameter(0)
  %zero = bf16[] constant(0)
  %s = bf16[64] reduce(%x, %zero), dimensions={{0}}, to_apply=%reduce_add
  ROOT %b = bf16[64,64] broadcast(%s), dimensions={{0}}
}}
"""

# The rows' sums, spread along each row
ROW_SUM_ACROSS = f"""HloModule row_sum_across, entry_computation_layout={{(bf16[64,64]{{1,0}})->bf16[64,64]{{1,0}}}}

{REDUCE_ADD}

ENTRY main {{
  %x = bf16[64,64] parameter(0)
  %zero = bf16[] constant(0)
  %s = bf16[64] reduce(%x, %zero), dimensions={{1}}, to_apply=%reduce_add
  ROOT %b = bf16[64,64] broadcast(%s), dimensions={{1}}
}}
"""

# The exponentials divided by the rows' sums spread along each row
SOFTMAX_ACROSS = f"""HloModule softmax_across, entry_computation_layout={{(bf16[64,64]{{1,0}})->bf16[64,64]{{1,0}}}}

{REDUCE_ADD}

ENTRY main {{
  %x = bf16[64,64] parameter(0)
  %ex = bf16[64,64] exponential(%x)
  %zero = bf16[] constant(0)
  %s = bf16[64] reduce(%ex, %zero), dimensions={{1}}, to_apply=%reduce_add
  %b = bf16[64,64] broadcast(%s), dimensions={{1}}
  ROOT %out = bf16[64,64] divide(%ex, %b)
}}
"""

# The exponentials of a broadcast that turns its rows into columns
TURNED_SPREAD_EXP = """HloModule turned_spread_exp, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  %t = bf16[64,64] broadcast(%x), dimensions={1,0}
  ROOT %e = bf16[64,64] exponential(%t)
}
"""


@pytest.mark.parametrize(
    'kernel', [KEPT_EXP, COLUMN_SUM, ROW_SUM_ACROSS, SOFTMAX_ACROSS, TURNED_SPREAD_EXP]
)
def test_operation_whose_attributes_no_instruction_spells_finds_no_assembly(
    backend, tmp_path, kernel
):
    path = tmp_path / 'kernel.hlo'
    path.write_text(kernel)
    output = tmp_path / 'kernel.py'
    arguments = ['--input', str(path), '--output', str(output), '--log', str(tmp_path / 'log')]
    run = subprocess.run([backend, *arguments], capture_output=True, text=True, check=False)
    assert run.returncode == 1
    assert run.stderr.endswith('and no candidate was extracted\n')
    assert not output.exists()


# The exponentials of 32 rows of 16 f32 values
TILE = """HloModule tile, entry_computation_layout={(f32[32,16]{1,0})->f32[32,16]{1,0}}

ENTRY main {
  %x = f32[32,16] parameter(0)
  ROOT %y = f32[32,16] exponential(%x)
}
"""


def test_attributes_in_a_product_are_read_where_each_stands_alone(backend, tmp_path):
    calls = selected(backend, tmp_path, TILE)
    assert [name for name, _ in calls] == ['load_tile', 'exp_tile', 'store_tile']
    assert all((call['m'], call['n']) == ('32', '16') for _, call in calls)


# A vector, the one input, spread over 64 rows
SPREAD_ROWS = """HloModule spread_rows, entry_computation_layout={(bf16[64]{0})->bf16[64,64]{1,0}}

ENTRY main {
  %v = bf16[64] parameter(0)
  ROOT %b = bf16[64,64] broadcast(%v), dimensions={1}
}
"""


def test_rank_1_input_is_simulated(spreading_dir, tmp_path):
    backend = os.path.join(spreading_dir, 'backends', 'SPREADING')
    calls = selected(backend, tmp_path, SPREAD_ROWS)
    assert [name for name, _ in calls] == ['load_spread', 'store_rm']
    # The vector's shape, a tuple of one, reaches the oracle
    vector = jnp.arange(64, dtype=jnp.bfloat16) / 8
    output = simulated(spreading_dir, 'SPREADING', tmp_path / 'kernel.py', vector)
    assert np.array_equal(np.asarray(output), np.tile(np.asarray(vector), (64, 1)))


# A bf16 scalar, the one input, added to itself
SCALAR = """HloModule scalar, entry_computation_layout={(bf16[])->bf16[]}

ENTRY main {
  %s = bf16[] parameter(0)
  ROOT %t = bf16[] add(%s, %s)
}
"""


def test_rank_0_value_is_simulated(scalar_dir, tmp_path):
    backend = os.path.join(scalar_dir, 'backends', 'SCALAR')
    assert [name for name, _ in selected(backend, tmp_path, SCALAR)] == ['sadd']
    # The scalar reaches the oracle with the kernel's shape, ()
    scalar = jnp.array(1.5, dtype=jnp.bfloat16)
    output = simulated(scalar_dir, 'SCALAR', tmp_path / 'kernel.py', scalar)
    assert output.shape == ()
    assert float(output) == 3.0


# s32 rows narrowed to s8, widened back: low bytes kept
WRAPPED = """HloModule wrapped, entry_computation_layout={(s32[32,64]{1,0})->s32[32,64]{1,0}}

ENTRY main {
  %x = s32[32,64] parameter(0)
  %n = s8[32,64] convert(%x)
  ROOT %w = s32[32,64] convert(%n)
}
"""


def test_narrowing_convert_is_kept(backend, tmp_path):
    calls = selected(backend, tmp_path, WRAPPED)
    assert [name for name, _ in calls] == ['load_s32', 'narrow', 'widen', 'store_s32']


# Rows of s8 widened, narrowed back and widened again
WIDENED = """HloModule widened, entry_computation_layout={(s8[32,64]{1,0})->s32[32,64]{1,0}}

ENTRY main {
  %x = s8[32,64] parameter(0)
  %w = s32[32,64] convert(%x)
  %n = s8[32,64] convert(%w)
  ROOT %v = s32[32,64] convert(%n)
}
"""


def test_convert_pair_over_a_widened_value_is_dropped(backend, tmp_path):
    calls = selected(backend, tmp_path, WIDENED)
    assert [name for name, _ in calls] == ['load_s8', 'widen', 'store_s32']
