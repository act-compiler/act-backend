"""Tests of the extractors' candidates, on specs made for them"""

import os
import re

import pytest
from conftest import add_moves, compiled
from taidl import Accelerator

from act_backend import generate_backend


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
