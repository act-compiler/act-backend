"""Tests of constructs a backend builds with: QKV's spec with each one added, generated once."""

import os
import re
import subprocess

import pytest
from conftest import EXAMPLES_DIR, generate_oracle_and_backend, instruction_calls, run_backend

import act_backend
from act_backend import generate_backend

QKV_DIR = os.path.join(EXAMPLES_DIR, 'QKV')


def add_concatenate(acc):
    """An instruction that joins two halves of d1 into d2"""
    instr = acc.add_instruction('join', [], ['addr_1', 'addr_2', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_1'], ['32']], ['d1', ['@a.addr_2'], ['32']]])
    instr.set_outputs([['d2', ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[32,64] parameter(0);
    %In1 = bf16[32,64] parameter(1);
    ROOT %Out0 = bf16[64,64] concatenate(%In0, %In1), dimensions={0};
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


def add_rank_0_result(acc):
    """An instruction that fills d2 with the sum of 64 rows of d1"""
    instr = acc.add_instruction('spread', [], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['64']]])
    instr.set_outputs([['d2', ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %total = bf16[] reduce_add(%In0), dimensions={0,1};
    ROOT %Out0 = bf16[64,64] broadcast(%total), dimensions={};
}
""")


# A kernel filling its output with its input's sum
SPREAD = """HloModule spread, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

%reduce_add (x: bf16[], y: bf16[]) -> bf16[] {
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  %zero = bf16[] constant(0)
  %total = bf16[] reduce(%x, %zero), dimensions={0,1}, to_apply=%reduce_add
  ROOT %out = bf16[64,64] broadcast(%total), dimensions={}
}
"""


def add_slice(acc):
    """An instruction that takes the exponentials of 64 rows of d1 and keeps the first 32"""
    instr = acc.add_instruction('upper', [], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['64']]])
    instr.set_outputs([['d2', ['@a.addr_out'], ['32']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %a = bf16[64,64] exponential(%In0);
    ROOT %Out0 = bf16[32,64] slice(%a), slice={[0:32], [0:64]};
}
""")


# A kernel keeping 32 rows of its input's exponentials
SLICED = """HloModule sliced, entry_computation_layout={(bf16[64,64]{1,0})->bf16[32,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  %y = bf16[64,64] exponential(%x)
  ROOT %out = bf16[32,64] slice(%y), slice={[0:32], [0:64]}
}
"""


def add_moves(acc, element_type, size):
    """A load of rows of one element type into d1 and a store of them from it"""
    # A row is 64 elements of `size` bytes each
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


def add_binary(acc, name, word, element_type):
    """An instruction that joins two values of 32 rows of d1 by one operation word"""
    instr = acc.add_instruction(name, [], ['addr_1', 'addr_2', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_1'], ['32']], ['d1', ['@a.addr_2'], ['32']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['32']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = {element_type}[32,64] parameter(0);
    %In1 = {element_type}[32,64] parameter(1);
    ROOT %Out0 = {element_type}[32,64] {word}(%In0, %In1);
}}
""")


def add_bit_operations(acc):
    """Bytes loaded and stored, and one instruction each for or and the two shifts"""
    add_moves(acc, 'u8', 1)
    add_binary(acc, 'bits_or', 'or', 'u8')
    add_binary(acc, 'bits_shl', 'shift_left', 'u8')
    add_binary(acc, 'bits_shr', 'shift_right_logical', 'u8')


# Two byte inputs joined by or and both shifts
BITS = """HloModule bits, entry_computation_layout={(u8[32,64]{1,0}, u8[32,64]{1,0})->u8[32,64]{1,0}}

ENTRY main {
  %a = u8[32,64] parameter(0)
  %b = u8[32,64] parameter(1)
  %c = u8[32,64] or(%a, %b)
  %d = u8[32,64] shift-left(%c, %b)
  ROOT %e = u8[32,64] shift-right-logical(%d, %a)
}
"""


def add_f32(acc):
    """Rows of f32 loaded and stored, and an instruction that adds two values of them"""
    add_moves(acc, 'f32', 4)
    add_binary(acc, 'add_f32', 'add', 'f32')


# A kernel that adds two inputs of f32
SUMMED = """HloModule summed, entry_computation_layout={(f32[32,64]{1,0}, f32[32,64]{1,0})->f32[32,64]{1,0}}

ENTRY main {
  %a = f32[32,64] parameter(0)
  %b = f32[32,64] parameter(1)
  ROOT %c = f32[32,64] add(%a, %b)
}
"""


def add_data_model_with_no_access_dimension(acc):
    """A buffer that is one unit of 64 columns"""
    acc.add_data_model('d3', [], [64], 'bf16')


def add_instruction_with_no_semantics(acc):
    """An instruction that sets a mode and computes nothing"""
    acc.add_instruction('configure', ['mode'], [])


@pytest.fixture(scope='module')
def work_dir(tmp_path_factory):
    """Generate QKV's backend with every construct above added to its accelerator"""
    work_dir = str(tmp_path_factory.mktemp('constructs'))

    def with_constructs(acc, *args, **kwargs):
        add_concatenate(acc)
        add_rank_0_result(acc)
        add_slice(acc)
        add_bit_operations(acc)
        add_f32(acc)
        add_data_model_with_no_access_dimension(acc)
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


def compiled_calls(work_dir, tmp_path, kernel):
    """Compile a kernel's text; return each call of its assembly, a name and its arguments"""
    path = tmp_path / 'kernel.hlo'
    path.write_text(kernel)
    output = tmp_path / 'kernel.py'
    backend = os.path.join(work_dir, 'backends', 'QKV')
    log = str(tmp_path / 'log')
    arguments = ['--input', str(path), '--output', str(output), '--log', log]
    run = subprocess.run([backend, *arguments], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    return [
        (name, dict(re.findall(r'(\w+) = (\d+)', arguments)))
        for name, arguments in re.findall(r'api\.(\w+)\(([^)]*)\)', output.read_text())
    ]


def test_concatenate_is_selected(work_dir, tmp_path):
    calls = compiled_calls(work_dir, tmp_path, JOINED)
    # Each half is loaded, the two are joined in the kernel's order, and the rows normalised
    names = [name for name, _ in calls]
    assert names == ['load_rm', 'load_rm', 'join', 'softmax', 'mov', 'store_rm']
    loaded = {call['addr_in']: call['addr_out'] for name, call in calls if name == 'load_rm'}
    join = calls[2][1]
    assert (join['addr_1'], join['addr_2']) == (loaded['0'], loaded['4096'])


def test_rank_0_result_is_selected(work_dir, tmp_path):
    calls = compiled_calls(work_dir, tmp_path, SPREAD)
    # The rank-0 sum is matched inside the instruction
    assert [name for name, _ in calls] == ['load_rm', 'spread', 'mov', 'store_rm']


def test_attribute_named_by_an_operation_word_is_read(work_dir, tmp_path):
    calls = compiled_calls(work_dir, tmp_path, SLICED)
    # The attribute named slice is read: upper keeps 32 rows
    assert [name for name, _ in calls] == ['load_rm', 'upper', 'mov', 'store_rm']
    assert [call['n'] for _, call in calls if 'n' in call] == ['64', '32', '32']


def test_or_and_the_shifts_are_selected(work_dir, tmp_path):
    calls = compiled_calls(work_dir, tmp_path, BITS)
    names = [name for name, _ in calls]
    assert names == ['load_u8', 'load_u8', 'bits_or', 'bits_shl', 'bits_shr', 'store_u8']
    # Each instruction takes its two values in the kernel's order
    loaded = {call['addr_in']: call['addr_out'] for _, call in calls[:2]}
    a, b = loaded['0'], loaded['2048']
    (_, joined), (_, left), (_, right) = calls[2:5]
    assert (joined['addr_1'], joined['addr_2']) == (a, b)
    assert (left['addr_1'], left['addr_2']) == (joined['addr_out'], b)
    assert (right['addr_1'], right['addr_2']) == (left['addr_out'], a)


def test_f32_values_are_selected(work_dir, tmp_path):
    calls = compiled_calls(work_dir, tmp_path, SUMMED)
    assert [name for name, _ in calls] == ['load_f32', 'load_f32', 'add_f32', 'store_f32']
    # Each input is four bytes an element
    assert {call['addr_in'] for _, call in calls[:2]} == {'0', '8192'}


def test_data_model_with_no_access_dimension_is_a_storage(work_dir):
    allocator = os.path.join(work_dir, 'targets', 'QKV', 'backend', 'cpp', 'malloc')
    with open(os.path.join(allocator, 'src', 'globals.cc')) as f:
        assert 'new Storage("D3", {}, 1)' in f.read()


def test_instruction_with_no_semantics_is_left_out(work_dir):
    crate = os.path.join(work_dir, 'targets', 'QKV', 'backend')
    for source in [('src', 'ir', 'egraph.rs'), ('cpp', 'malloc', 'include', 'instructions.h')]:
        with open(os.path.join(crate, *source)) as f:
            assert 'Configure' not in f.read()
