"""Tests of the allocator's checks: a candidate it cannot take is dropped with the reason."""

import json
import os
import subprocess

import pytest

# A candidate of the endings spec: the sum of two 32-row matrices
CANDIDATE = [
    "t0: HBM[4096] = u8[4096] Var['%b']()",
    "t1: HBM[0] = u8[4096] Var['%a']()",
    "t2: Modeld1[-1] = bf16[32,64] instr_load_rm[rows='32'](t1)",
    "t3: Modeld1[-1] = bf16[32,64] instr_load_rm[rows='32'](t0)",
    't4: Modeld1[-1] = bf16[32,64] instr_vadd_32(t2, t3)',
    "t5: HBM[8192] = u8[4096] instr_store_rm[rows='32'](t4)",
]

# Its kernel's metadata
METADATA = {
    'module_name': 'add_32',
    'input_file': 'add_32.hlo',
    'input': [
        {'addr': 0, 'shape': [32, 64], 'dtype': 'jnp.bfloat16'},
        {'addr': 4096, 'shape': [32, 64], 'dtype': 'jnp.bfloat16'},
    ],
    'output': [{'addr': 8192, 'shape': [32, 64], 'dtype': 'jnp.bfloat16'}],
}


@pytest.fixture(scope='module')
def allocator(endings_dir):
    """The endings spec's allocator, as its debug executable"""
    build = os.path.join(endings_dir, 'targets', 'ENDINGS', 'backend', 'cpp', 'malloc', 'build')
    return os.path.join(build, 'malloc')


def allocated(allocator, tmp_path, lines, metadata=METADATA):
    """Allocate a candidate's lines; return the run and its assembly's path"""
    pii = tmp_path / 'candidate.pii'
    pii.write_text(''.join(f'{line}\n' for line in lines))
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps(metadata))
    run = subprocess.run(
        [allocator, str(pii), str(path)], capture_output=True, text=True, check=False
    )
    return run, tmp_path / 'candidate.py'


def with_line(index, line):
    """The candidate with one of its lines replaced"""
    return [*CANDIDATE[:index], line, *CANDIDATE[index + 1 :]]


def test_candidate_is_allocated(allocator, tmp_path):
    run, assembly = allocated(allocator, tmp_path, CANDIDATE)
    assert run.returncode == 0, run.stderr
    assert assembly.exists()


# Each line the allocator cannot read, at its place, and the fault it names
UNREADABLE = {
    'colon': (
        2,
        "t2 Modeld1[-1] = bf16[32,64] instr_load_rm[rows='32'](t1)",
        "missing ':' separator",
    ),
    'equals': (4, 't4: Modeld1[-1] bf16[32,64] instr_vadd_32(t2, t3)', "missing '=' separator"),
    'offset': (
        2,
        "t2: Modeld1 = bf16[32,64] instr_load_rm[rows='32'](t1)",
        "missing '[' or ']' for storage offset",
    ),
    'space': (
        2,
        "t2: Modeld1[-1] = bf16[32,64]instr_load_rm[rows='32'](t1)",
        'missing type/shape before operation (no space found)',
    ),
    'shape': (
        2,
        "t2: Modeld1[-1] = bf16 instr_load_rm[rows='32'](t1)",
        "missing '[' or ']' for type shape",
    ),
    'operands': (
        2,
        "t2: Modeld1[-1] = bf16[32,64] instr_load_rm[rows='32']",
        "missing '(' or ')' for operands",
    ),
    'storage': (2, "t2: d9[-1] = bf16[32,64] instr_load_rm[rows='32'](t1)", 'unknown storage: d9'),
    'operand': (
        2,
        "t2: Modeld1[-1] = bf16[32,64] instr_load_rm[rows='32'](t9)",
        'unknown operand: t9',
    ),
    'slices': (4, 't4: Modeld1[-1] = bf16[32,64] slice(t2)', 'missing slices attribute'),
    'range': (
        4,
        "t4: Modeld1[-1] = bf16[32,64] slice[slices='0'](t2)",
        "missing ':' in slice specification",
    ),
    'dimension': (4, 't4: Modeld1[-1] = bf16[32,64] concat(t2, t3)', 'missing dimension attribute'),
    'fixed': (
        2,
        "t2: Modeld1[0] = bf16[32,64] instr_load_rm[rows='32'](t1)",
        'only HBM storage supports fixed offset specification',
    ),
    'negative': (1, "t1: HBM[-2] = u8[4096] Var['%a']()", 'storage offset must be non-negative'),
}


@pytest.mark.parametrize('index, line, fault', UNREADABLE.values(), ids=UNREADABLE)
def test_line_the_allocator_cannot_read_drops_the_candidate(
    allocator, tmp_path, index, line, fault
):
    run, assembly = allocated(allocator, tmp_path, with_line(index, line))
    assert run.returncode == 1
    assert run.stderr == f'the allocator cannot read the line `{line}`: {fault}\n'
    assert not assembly.exists()
