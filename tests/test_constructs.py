"""Tests of constructs a backend builds with: QKV's spec with each one added, generated once."""

import os

import pytest

import act_backend
from act_backend import generate_backend
from conftest import EXAMPLES_DIR, generate_oracle_and_backend, instruction_calls, run_backend

QKV_DIR = os.path.join(EXAMPLES_DIR, 'QKV')


def add_instruction_with_no_semantics(acc):
    """An instruction that sets a mode and computes nothing"""
    acc.add_instruction('configure', ['mode'], [])


@pytest.fixture(scope='module')
def work_dir(tmp_path_factory):
    """Generate QKV's backend with every construct above added to its accelerator"""
    work_dir = str(tmp_path_factory.mktemp('constructs'))

    def with_constructs(acc, *args, **kwargs):
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


def test_instruction_with_no_semantics_is_left_out(work_dir):
    crate = os.path.join(work_dir, 'targets', 'QKV', 'backend')
    for source in [('src', 'ir', 'egraph.rs'), ('cpp', 'malloc', 'include', 'instructions.h')]:
        with open(os.path.join(crate, *source)) as f:
            assert 'Configure' not in f.read()
