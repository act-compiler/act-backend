"""Tests of a whole generation: what generate_backend builds and what it says when it cannot."""

import os
import shutil

import pytest

from act_backend import generate_backend
from conftest import EXAMPLES_DIR, instruction_calls, run_backend

QKV_DIR = os.path.join(EXAMPLES_DIR, 'QKV')


def test_failed_build_shows_its_error(accelerator, tmp_path, monkeypatch):
    # The allocator's build stops on a C++ compiler that is not there
    monkeypatch.setenv('CXX', 'no-such-compiler')
    with pytest.raises(RuntimeError) as error:
        generate_backend(accelerator, str(tmp_path))
    assert 'no-such-compiler' in str(error.value)
    assert os.path.join(str(tmp_path), 'targets', 'PAIR', 'backend') in str(error.value)


def test_crate_holds_its_ignore_file(work_dir):
    # The crate comes whole from the installed package, its dot file too
    assert os.path.isfile(os.path.join(work_dir, 'targets', 'QKV', 'backend', '.gitignore'))


def test_backend_runs_without_its_crate(work_dir, tmp_path):
    # The binary alone, copied away, and the crate it was built in moved
    backend = shutil.copy(os.path.join(work_dir, 'backends', 'QKV'), tmp_path)
    crate = os.path.join(work_dir, 'targets', 'QKV', 'backend')
    os.rename(crate, crate + '.moved')
    try:
        run, output_path = run_backend(backend, QKV_DIR, str(tmp_path))
    finally:
        os.rename(crate + '.moved', crate)
    assert run.returncode == 0, run.stderr
    assert sum(instruction_calls(output_path).values()) == 9
