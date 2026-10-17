"""Tests of the allocator's checks: a candidate it cannot take is dropped with the reason."""

import copy
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


def test_tensor_defined_twice_drops_the_candidate(allocator, tmp_path):
    line = "t2: Modeld1[-1] = bf16[32,64] instr_load_rm[rows='32'](t0)"
    run, assembly = allocated(allocator, tmp_path, with_line(3, line))
    assert run.returncode == 1
    assert run.stderr == (
        f'the allocator cannot read the line `{line}`: the tensor t2 is defined twice\n'
    )
    assert not assembly.exists()


def metadata_with(change):
    """The kernel's metadata, as a function of it changes it"""
    metadata = copy.deepcopy(METADATA)
    change(metadata)
    return metadata


# Each fault of the metadata, and what the allocator says of it
FAULTY = {
    'addr': (lambda m: m['input'][0].pop('addr'), 'addr is required in metadata.json'),
    'shape': (lambda m: m['input'][0].pop('shape'), 'shape is required in metadata.json'),
    'entry': (
        lambda m: m['input'][0].update(shape=[32, '64']),
        'shape must be an array of integers',
    ),
    'dtype': (lambda m: m['output'][0].pop('dtype'), 'dtype is required in metadata.json'),
    'name': (lambda m: m.pop('module_name'), 'module_name is required in metadata.json'),
    'input': (lambda m: m.pop('input'), 'input is required in metadata.json'),
    'output': (lambda m: m.pop('output'), 'output is required in metadata.json'),
    'outputs': (
        lambda m: m['output'].append(m['output'][0]),
        'supports only one output in metadata.json',
    ),
}


@pytest.mark.parametrize('change, fault', FAULTY.values(), ids=FAULTY)
def test_metadata_the_allocator_cannot_parse_drops_the_candidate(
    allocator, tmp_path, change, fault
):
    run, assembly = allocated(allocator, tmp_path, CANDIDATE, metadata_with(change))
    assert run.returncode == 1
    assert run.stderr == f'the allocator cannot parse metadata.json: {fault}\n'
    assert not assembly.exists()


# Each line the allocator cannot build, at its place, and the fault it names
UNBUILDABLE = {
    'rank': (
        2,
        "t2: Modeld1[-1] = bf16[32] instr_load_rm[rows='32'](t1)",
        'the tensor t2 has a shape of rank 1, where its storage Modeld1 has rank 2',
    ),
    'slice-operands': (
        4,
        "t4: Modeld1[-1] = bf16[32,64] slice[slices='0:32'](t2, t3)",
        'a slice takes one operand, and t4 has 2',
    ),
    'slice-storage': (
        4,
        "t4: Modeld1[-1] = bf16[32,64] slice[slices='0:32'](t1)",
        'the slice t4 lies in Modeld1, and its operand t1 in HBM',
    ),
    'slice-ranges': (
        4,
        "t4: Modeld1[-1] = bf16[32,64] slice[slices=''](t2)",
        'the slice t4 gives 0 ranges, where it takes 1',
    ),
    'slice-bounds': (
        4,
        "t4: Modeld1[-1] = bf16[32,64] slice[slices='0:64'](t2)",
        'the slice t4 takes 0:64 of t2, which has 32',
    ),
    'slice-size': (
        4,
        "t4: Modeld1[-1] = bf16[16,64] slice[slices='0:32'](t2)",
        'the slice t4 has 16, where its range 0:32 takes 32',
    ),
    'concatenate-operands': (
        4,
        "t4: Modeld1[-1] = bf16[32,64] concat[dimension='0'](t2)",
        'a concatenate takes two operands, and t4 has 1',
    ),
    'concatenate-storage': (
        4,
        "t4: Modeld1[-1] = bf16[32,64] concat[dimension='0'](t2, t1)",
        'the concatenate t4 lies in Modeld1, and its operand t1 in HBM',
    ),
    'concatenate-sizes': (
        4,
        "t4: Modeld1[-1] = bf16[32,64] concat[dimension='0'](t2, t3)",
        'the concatenate t4 has 32 along dimension 0, where its operands give 32 and 32',
    ),
}


@pytest.mark.parametrize('index, line, fault', UNBUILDABLE.values(), ids=UNBUILDABLE)
def test_line_the_allocator_cannot_build_drops_the_candidate(
    allocator, tmp_path, index, line, fault
):
    run, assembly = allocated(allocator, tmp_path, with_line(index, line))
    assert run.returncode == 1
    assert run.stderr == f'the allocator cannot read the line `{line}`: {fault}\n'
    assert not assembly.exists()
