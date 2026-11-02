"""Tests of the extractors' candidates, on specs made for them"""

import os
import re
from collections import Counter

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    EXAMPLES_DIR,
    add_moves,
    add_on_d1,
    bf16,
    compiled,
    extracted,
    instruction_calls,
    simulated,
)
from taidl import Accelerator
from taidl_to import generate_oracle

from act_backend import generate_backend

EXAMPLE_DIR = os.path.join(EXAMPLES_DIR, 'mini_trn_double')


def add_copy(acc, name, source, target):
    """A copy of bf16 rows from one buffer to another"""
    instr = acc.add_instruction(name, [], ['addr_in', 'addr_out'])
    instr.set_inputs([[source, ['@a.addr_in'], ['64']]])
    instr.set_outputs([[target, ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    ROOT %Out0 = bf16[64,64] copy(%In0);
}
""")


def round_trip():
    """Moves among HBM, d1 and d2, and an exponential on d2"""
    acc = Accelerator('ROUND_TRIP')
    acc.add_data_model('d1', [128], [64], 'bf16')
    acc.add_data_model('d2', [128], [64], 'bf16')
    add_moves(acc)
    add_copy(acc, 'to_d2', 'd1', 'd2')
    add_copy(acc, 'to_d1', 'd2', 'd1')
    instr = acc.add_instruction('exp_d2', [], ['addr_in', 'addr_out'])
    instr.set_inputs([['d2', ['@a.addr_in'], ['64']]])
    instr.set_outputs([['d2', ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    ROOT %Out0 = bf16[64,64] exponential(%In0);
}
""")
    return acc


@pytest.fixture(scope='module')
def round_trip_backend(tmp_path_factory):
    """Generate the spec's backend once and return the binary's path"""
    work_dir = str(tmp_path_factory.mktemp('round_trip'))
    generate_backend(round_trip(), work_dir)
    return os.path.join(work_dir, 'backends', 'ROUND_TRIP')


# An exponential of the input, computed on d2
EXPONENTIAL = """HloModule exponential, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  ROOT %y = bf16[64,64] exponential(%x)
}
"""

# A candidate's node: its number, buffer, operation, and children
NODE = re.compile(r't(\d+): (\w+)\[-?\d+\] = \S+ (\w+)\S*\((.*)\)')
# The instructions that move a value unchanged
MOVES = {'instr_load_rm', 'instr_store_rm', 'instr_to_d1', 'instr_to_d2'}


def candidate_nodes(pii):
    """Return a candidate's nodes, each its buffer and e-class with its children"""
    nodes = {}
    for line in pii.read_text().splitlines():
        number, buffer, operation, children = NODE.fullmatch(line).groups()
        children = [nodes[int(child)] for child in re.findall(r't(\d+)', children)]
        # A move keeps its input's value
        if operation in MOVES:
            value = children[0][0][1]
        else:
            value = f'{operation}({", ".join(child[0][1] for child in children)})'
        nodes[int(number)] = ((buffer, value), children)
    return nodes.values()


def repeats(node, path=frozenset()):
    """Whether a path down from a node passes one e-class in one buffer twice"""
    key, children = node
    return key in path or any(repeats(child, path | {key}) for child in children)


def test_no_candidate_passes_one_eclass_in_one_buffer_twice(round_trip_backend, tmp_path):
    run, _ = compiled(round_trip_backend, tmp_path, EXPONENTIAL)
    assert run.returncode == 0, run.stderr
    # Check every logged candidate from each of its nodes
    piis = sorted((tmp_path / 'log').glob('*.pii'))
    assert piis
    repeating = [pii.name for pii in piis if any(map(repeats, candidate_nodes(pii)))]
    assert repeating == []


# A sum of two inputs, each loaded on its own
SUM = """HloModule sum, entry_computation_layout={(bf16[32,64]{1,0}, bf16[32,64]{1,0})->bf16[32,64]{1,0}}

ENTRY main {
  %a = bf16[32,64] parameter(0)
  %b = bf16[32,64] parameter(1)
  ROOT %c = bf16[32,64] add(%a, %b)
}
"""


def test_fast_extraction_orders_the_nodes_alike_on_every_compile(backend, tmp_path):
    candidates = set()
    for number in range(8):
        run_dir = tmp_path / str(number)
        run_dir.mkdir()
        run, _ = compiled(backend, run_dir, SUM, flags=['--verbose'])
        assert run.returncode == 0, run.stderr
        # The fast candidate is extracted first, so it is 0.pii
        assert extracted(run.stdout)[0] == ('Fast', 1)
        candidates.add((run_dir / 'log' / '0.pii').read_text())
    # Two independent leaves and loads, yet one node order
    assert len(candidates) == 1


def add_across(acc):
    """An add of 64 rows of d1 to 64 rows of d2, into d1"""
    instr = acc.add_instruction('add_across', [], ['addr_1', 'addr_2', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_1'], ['64']], ['d2', ['@a.addr_2'], ['64']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %In1 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] add(%In0, %In1);
}
""")


def add_load(acc, name, model):
    """A load of rows into a buffer"""
    instr = acc.add_instruction(name, ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['@c.n * 128']]])
    instr.set_outputs([[model, ['@a.addr_out'], ['@c.n']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In0);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")
    return instr


def two_loads():
    """Moves, a load into d2, and an add across buffers"""
    acc = Accelerator('TWO_LOADS')
    acc.add_data_model('d1', [128], [64], 'bf16')
    acc.add_data_model('d2', [128], [64], 'bf16')
    add_moves(acc)
    add_load(acc, 'load_d2', 'd2')
    add_across(acc)
    return acc


@pytest.fixture(scope='module')
def two_loads_dir(tmp_path_factory):
    """Generate the oracle and the backend of the spec that loads into two buffers"""
    work_dir = str(tmp_path_factory.mktemp('two_loads'))
    acc = two_loads()
    generate_oracle(acc, work_dir)
    generate_backend(acc, work_dir)
    return work_dir


def first_fast(stdout):
    """Return the number of a compile's first fast candidate"""
    reports = extracted(stdout)
    assert ('Fast', 1) in reports, 'the fast extractor found no graph'
    # Numbered as extracted, so count the ones before it
    return sum(count for _, count in reports[: reports.index(('Fast', 1))])


def fast_compiled(work_dir, name, tmp_path, kernel):
    """Compile a kernel and check the fast candidate won"""
    backend = os.path.join(work_dir, 'backends', name)
    run, output = compiled(backend, tmp_path, kernel, flags=['--verbose'])
    assert run.returncode == 0, run.stderr
    number = first_fast(run.stdout)
    assert f'PII #{number} dropped' not in run.stdout
    assert output.read_text() == (tmp_path / 'log' / f'{number}.py').read_text()
    return output


# Each row its own value, so a misplaced operand shows
ROWS = jnp.repeat(jnp.arange(64, dtype=jnp.bfloat16)[:, None], 64, axis=1)

# An input added to itself, read from d1 and d2
SIBLINGS = """HloModule siblings, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  ROOT %y = bf16[64,64] add(%x, %x)
}
"""


def test_fast_extraction_gives_a_value_read_in_two_buffers_a_node_in_each(two_loads_dir, tmp_path):
    output = fast_compiled(two_loads_dir, 'TWO_LOADS', tmp_path, SIBLINGS)
    result = simulated(two_loads_dir, 'TWO_LOADS', output, ROWS)
    assert np.array_equal(np.asarray(result), np.asarray(ROWS + ROWS))


# Two concatenates sharing an input, read from d1 and d2
SHARED = """HloModule shared, entry_computation_layout={(bf16[32,64]{1,0}, bf16[32,64]{1,0}, bf16[32,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[32,64] parameter(0)
  %b = bf16[32,64] parameter(1)
  %c = bf16[32,64] parameter(2)
  %ab = bf16[64,64] concatenate(%a, %b), dimensions={0}
  %ac = bf16[64,64] concatenate(%a, %c), dimensions={0}
  ROOT %y = bf16[64,64] add(%ab, %ac)
}
"""

# The concatenated inputs, each row its own value
A, B, C = ROWS[:32], ROWS[32:], ROWS[:32] + 64


def test_fast_extraction_takes_a_concatenates_inputs_from_its_buffer(two_loads_dir, tmp_path):
    output = fast_compiled(two_loads_dir, 'TWO_LOADS', tmp_path, SHARED)
    result = simulated(two_loads_dir, 'TWO_LOADS', output, A, B, C)
    expected = jnp.concatenate([A, B]) + jnp.concatenate([A, C])
    assert np.array_equal(np.asarray(result), np.asarray(expected))


# One concatenate, read from d1 and from d2
REPEATED = """HloModule repeated, entry_computation_layout={(bf16[32,64]{1,0}, bf16[32,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[32,64] parameter(0)
  %b = bf16[32,64] parameter(1)
  %ab = bf16[64,64] concatenate(%a, %b), dimensions={0}
  ROOT %y = bf16[64,64] add(%ab, %ab)
}
"""


def test_fast_extraction_gives_a_concatenate_read_in_two_buffers_a_node_in_each(
    two_loads_dir, tmp_path
):
    backend = os.path.join(two_loads_dir, 'backends', 'TWO_LOADS')
    run, _ = compiled(backend, tmp_path, REPEATED, flags=['--verbose'])
    assert run.returncode == 0, run.stderr
    number = first_fast(run.stdout)
    assert f'PII #{number} dropped' not in run.stdout
    # The fast candidate itself, whichever candidate the output is
    candidate = tmp_path / 'log' / f'{number}.py'
    result = simulated(two_loads_dir, 'TWO_LOADS', candidate, A, B)
    assert np.array_equal(np.asarray(result), np.asarray(2 * jnp.concatenate([A, B])))


def slow_numbers(stdout):
    """Return the numbers of a compile's slow candidates"""
    numbers, start = [], 0
    # Numbered as extracted, a run per report
    for extractor, count in extracted(stdout):
        if extractor == 'Slow':
            numbers += range(start, start + count)
        start += count
    return numbers


def slow_diffs(work_dir, tmp_path, kernel, expected, *inputs):
    """Compile a kernel; return each slow candidate's largest difference"""
    backend = os.path.join(work_dir, 'backends', 'TWO_LOADS')
    run, _ = compiled(backend, tmp_path, kernel, flags=['--verbose'])
    assert run.returncode == 0, run.stderr
    # The slow candidates themselves, whichever candidate the output is
    candidates = [tmp_path / 'log' / f'{number}.py' for number in slow_numbers(run.stdout)]
    # Dropped or unreached candidates have no assembly
    candidates = [candidate for candidate in candidates if candidate.exists()]
    assert candidates, 'no slow candidate was allocated'
    results = [simulated(work_dir, 'TWO_LOADS', candidate, *inputs) for candidate in candidates]
    return {float(jnp.max(jnp.abs(result - expected))) for result in results}


def test_slow_extraction_gives_a_concatenate_read_in_two_buffers_a_node_in_each(
    two_loads_dir, tmp_path
):
    expected = 2 * jnp.concatenate([A, B])
    assert slow_diffs(two_loads_dir, tmp_path, REPEATED, expected, A, B) == {0}


# One slice under two concatenates, read from d1 and d2
SLICED = """HloModule sliced, entry_computation_layout={(bf16[64,64]{1,0}, bf16[32,64]{1,0}, bf16[32,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  %b = bf16[32,64] parameter(1)
  %c = bf16[32,64] parameter(2)
  %s = bf16[32,64] slice(%x), slice={[0:32], [0:64]}
  %sb = bf16[64,64] concatenate(%s, %b), dimensions={0}
  %sc = bf16[64,64] concatenate(%s, %c), dimensions={0}
  ROOT %y = bf16[64,64] add(%sb, %sc)
}
"""


def test_slow_extraction_gives_a_slice_read_in_two_buffers_a_node_in_each(two_loads_dir, tmp_path):
    # The slice's rows are A
    expected = jnp.concatenate([A, B]) + jnp.concatenate([A, C])
    assert slow_diffs(two_loads_dir, tmp_path, SLICED, expected, ROWS, B, C) == {0}


def copying():
    """Moves, a copy to d2, an exponential, an add across buffers"""
    acc = Accelerator('COPYING')
    acc.add_data_model('d1', [128], [64], 'bf16')
    acc.add_data_model('d2', [128], [64], 'bf16')
    add_moves(acc)
    add_copy(acc, 'to_d2', 'd1', 'd2')
    instr = acc.add_instruction('exp_d1', [], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['64']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    ROOT %Out0 = bf16[64,64] exponential(%In0);
}
""")
    add_across(acc)
    return acc


@pytest.fixture(scope='module')
def copying_dir(tmp_path_factory):
    """Generate the oracle and the backend of the spec that copies to d2"""
    work_dir = str(tmp_path_factory.mktemp('copying'))
    acc = copying()
    generate_oracle(acc, work_dir)
    generate_backend(acc, work_dir)
    return work_dir


# An exponential added to itself, copied from d1 to d2
COPIED = """HloModule copied, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  %e = bf16[64,64] exponential(%x)
  ROOT %y = bf16[64,64] add(%e, %e)
}
"""


def test_fast_extraction_copies_a_value_needed_in_two_buffers(copying_dir, tmp_path):
    output = fast_compiled(copying_dir, 'COPYING', tmp_path, COPIED)
    x = ROWS / 64
    result = simulated(copying_dir, 'COPYING', output, x)
    # numpy's exponential, rounded to bf16 like the kernel
    expected = bf16(2 * bf16(np.exp(np.asarray(x, np.float64))))
    assert np.allclose(np.asarray(result, np.float64), expected, rtol=2**-7, atol=0)


# The sum of an add's two inputs
SUM_LINES = [
    '%In0 = bf16[64,64] parameter(0)',
    '%In1 = bf16[64,64] parameter(1)',
    'ROOT %Out0 = bf16[64,64] add(%In0, %In1)',
]


def exp_in_place(acc):
    """An exponential of 64 rows of d1 that overwrites its input"""
    instr = acc.add_instruction('exp_in_place', [], ['addr'])
    instr.set_inputs([['d1', ['@a.addr'], ['64']]])
    instr.set_outputs([['d1', ['@a.addr'], ['64']]])
    instr.set_inplace([True])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    ROOT %Out0 = bf16[64,64] exponential(%In0);
}
""")


def in_place():
    """Moves, an in-place exponential and add, and a plain add"""
    acc = Accelerator('IN_PLACE')
    acc.add_data_model('d1', [128], [64], 'bf16')
    add_moves(acc)
    exp_in_place(acc)
    # Declared first, so its e-class holds it first
    instr = acc.add_instruction('add_in_place', [], ['addr_1', 'addr_2'])
    instr.set_inputs([['d1', ['@a.addr_1'], ['64']], ['d1', ['@a.addr_2'], ['64']]])
    instr.set_outputs([['d1', ['@a.addr_1'], ['64']]])
    instr.set_inplace([True, False])
    instr.add_semantics('ENTRY {\n' + ''.join(f'    {line};\n' for line in SUM_LINES) + '}\n')
    add_on_d1(acc, 'add_d1', ['64', '64', '64'], SUM_LINES)
    return acc


@pytest.fixture(scope='module')
def in_place_dir(tmp_path_factory):
    """Generate the oracle and the backend of the spec whose instructions overwrite an input"""
    work_dir = str(tmp_path_factory.mktemp('in_place'))
    acc = in_place()
    generate_oracle(acc, work_dir)
    generate_backend(acc, work_dir)
    return work_dir


# A sum added to its first input
RESUMMED = """HloModule resummed, entry_computation_layout={(bf16[64,64]{1,0}, bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  %y = bf16[64,64] parameter(1)
  %s = bf16[64,64] add(%x, %y)
  ROOT %t = bf16[64,64] add(%s, %x)
}
"""


def test_fast_extraction_offers_a_plain_enode_before_an_in_place_one(in_place_dir, tmp_path):
    output = fast_compiled(in_place_dir, 'IN_PLACE', tmp_path, RESUMMED)
    # Plain adds, so x is loaded once and read twice
    assert instruction_calls(output) == {'load_rm': 2, 'add_d1': 2, 'store_rm': 1}
    result = simulated(in_place_dir, 'IN_PLACE', output, ROWS, 2 * ROWS)
    assert np.array_equal(np.asarray(result), np.asarray(4 * ROWS))


# An exponential added to its own input
RAISED = """HloModule raised, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  %e = bf16[64,64] exponential(%x)
  ROOT %y = bf16[64,64] add(%e, %x)
}
"""


def test_fast_extraction_copies_a_shared_operand_for_the_instruction_overwriting_it(
    in_place_dir, tmp_path
):
    output = fast_compiled(in_place_dir, 'IN_PLACE', tmp_path, RAISED)
    # exp_in_place overwrites one load of x, add_d1 reads the other
    calls = {'load_rm': 2, 'exp_in_place': 1, 'add_d1': 1, 'store_rm': 1}
    assert instruction_calls(output) == calls
    x = ROWS / 64
    result = simulated(in_place_dir, 'IN_PLACE', output, x)
    # numpy's exponential and sum, rounded to bf16 like the kernel
    x = np.asarray(x, np.float64)
    expected = bf16(bf16(np.exp(x)) + x)
    assert np.allclose(np.asarray(result, np.float64), expected, rtol=2**-7, atol=0)


# Each in-place instruction and the operand it overwrites
OVERWRITES = {'instr_exp_in_place': 0, 'instr_add_in_place': 0}


def rereads(pii):
    """Whether a candidate reads an operand an instruction overwrites again"""
    nodes = [NODE.fullmatch(line).groups() for line in pii.read_text().splitlines()]
    operands = {number: re.findall(r't(\d+)', children) for number, _, _, children in nodes}
    readers = Counter(child for children in operands.values() for child in children)
    return any(
        readers[operands[number][OVERWRITES[operation]]] > 1
        for number, _, operation, _ in nodes
        if operation in OVERWRITES
    )


def test_no_candidate_reads_an_overwritten_operand_again(in_place_dir, tmp_path):
    backend = os.path.join(in_place_dir, 'backends', 'IN_PLACE')
    run, _ = compiled(backend, tmp_path, RAISED)
    assert run.returncode == 0, run.stderr
    # Every logged candidate, the fast one and the slow ones
    piis = sorted((tmp_path / 'log').glob('*.pii'))
    assert len(piis) > 1
    assert [pii.name for pii in piis if rereads(pii)] == []


def pinned():
    """Two loads, one in place, and an add into HBM"""
    acc = Accelerator('PINNED')
    acc.add_data_model('d1', [128], [64], 'bf16')
    acc.add_data_model('d2', [128], [64], 'bf16')
    add_load(acc, 'load_d1', 'd1')
    add_load(acc, 'load_d2', 'd2').set_inplace([True])
    # No store, so no candidate copies x through HBM
    instr = acc.add_instruction('add_out', [], ['addr_1', 'addr_2', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_1'], ['64']], ['d2', ['@a.addr_2'], ['64']]])
    instr.set_outputs([['d0', ['@a.addr_out'], ['8192']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %In1 = bf16[64,64] parameter(1);
    %a = bf16[64,64] add(%In0, %In1);
    %b = u8[64,64,2] bitcast_convert(%a);
    ROOT %Out0 = u8[8192] reshape(%b);
}
""")
    return acc


@pytest.fixture(scope='module')
def pinned_backend(tmp_path_factory):
    """Generate the backend of the spec whose every candidate is dropped"""
    work_dir = str(tmp_path_factory.mktemp('pinned'))
    generate_backend(pinned(), work_dir)
    return os.path.join(work_dir, 'backends', 'PINNED')


def dropped_first(backend, tmp_path, kernel):
    """Return why a failing compile dropped its first candidate"""
    run, output = compiled(backend, tmp_path, kernel)
    assert run.returncode == 1
    assert not output.exists()
    # One stderr line, the first dropped candidate's reason last
    (line,) = run.stderr.splitlines()
    ended, reason = line.split(', and every candidate was dropped, the first because ')
    assert ended.startswith('Error: no assembly found: ')
    return reason


def test_overwritten_operand_pinned_in_hbm_drops_the_candidate(pinned_backend, tmp_path):
    # x loaded into d1 and, overwriting it, into d2
    reason = dropped_first(pinned_backend, tmp_path, SIBLINGS)
    assert reason == 'an overwritten operand has another reader and is pinned in HBM'


# An exponential of a concatenate, added to the concatenate
VIEWED = """HloModule viewed, entry_computation_layout={(bf16[32,64]{1,0}, bf16[32,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[32,64] parameter(0)
  %b = bf16[32,64] parameter(1)
  %ab = bf16[64,64] concatenate(%a, %b), dimensions={0}
  %e = bf16[64,64] exponential(%ab)
  ROOT %y = bf16[64,64] add(%e, %ab)
}
"""


def test_overwritten_view_of_shared_storage_drops_the_candidate(in_place_dir, tmp_path):
    backend = os.path.join(in_place_dir, 'backends', 'IN_PLACE')
    run, _ = compiled(backend, tmp_path, VIEWED, flags=['--verbose'])
    number = first_fast(run.stdout)
    # A copy of the concatenate would view the same loads
    reason = 'an overwritten operand views storage that has another reader'
    assert f'PII #{number} dropped: {reason}\n' in run.stdout


# A product by a constant, then a merging round trip
ROUND_TRIPPED = """HloModule round_tripped, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %x = bf16[64,64] parameter(0)
  %k = bf16[64,64] constant(3)
  %w = bf16[64,64] dot(%x, %k), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %t = bf16[4096,1] reshape(%w)
  %n = bf16[1,4096] reshape(%t)
  %m = bf16[1,4096] broadcast(%n), dimensions={0,1}
  %u = bf16[4096,1] reshape(%m)
  ROOT %c = bf16[64,64] reshape(%u)
}
"""


def test_output_keeps_its_hbm_offset_once_its_eclass_merges(plain_dir, tmp_path):
    backend = os.path.join(plain_dir, 'backends', 'mini_trn_double')
    run, output = compiled(backend, tmp_path, ROUND_TRIPPED, flags=['--verbose'])
    assert run.returncode == 0, run.stderr
    # The slow candidates themselves, whichever the search reached
    slow = [tmp_path / 'log' / f'{number}.pii' for number in slow_numbers(run.stdout)]
    # Unreached candidates have no graph
    assert [pii for pii in slow if pii.exists()], 'no slow candidate was logged'
    # Every candidate reached, however many the search allowed
    for pii in (tmp_path / 'log').glob('*.pii'):
        lines = pii.read_text().splitlines()
        # The output's offset on the root alone
        assert [line for line in lines if ': HBM[8192] = ' in line] == lines[-1:]
    # Each element its own value, so a misplaced read shows
    x = jnp.array(np.random.default_rng(0).normal(size=(64, 64)), dtype=jnp.bfloat16)
    result = np.asarray(simulated(plain_dir, 'mini_trn_double', output, x), np.float64)
    reference = bf16(np.asarray(x, np.float64) @ np.full((64, 64), 3.0))
    # Two bf16 ulps at max|reference| bound every output
    tol = 2 * 2.0 ** (np.floor(np.log2(np.abs(reference).max())) - 7)
    assert np.max(np.abs(result - reference)) <= tol
