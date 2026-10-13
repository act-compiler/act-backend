"""Tests of a whole generation: what generate_backend builds and what it says when it cannot."""

import os
import re
import shutil
import subprocess

import pytest

import act_backend
from act_backend import generate_backend
from act_backend.build_rs_generator import generate_build_rs_file
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


def test_cargo_leaves_the_lock_file_alone(work_dir):
    # A build by hand finds the lock file as cargo writes it and rewrites nothing
    crate = os.path.join(work_dir, 'targets', 'QKV', 'backend')
    subprocess.run(['cargo', 'build', '--release'], cwd=crate, check=True, capture_output=True)
    package_dir = os.path.dirname(act_backend.__file__)
    with open(os.path.join(package_dir, 'generic', 'Cargo.lock')) as f:
        package_lock = f.read()
    with open(os.path.join(crate, 'Cargo.lock')) as f:
        assert f.read() == package_lock


def test_stale_lock_file_fails_the_build(accelerator, tmp_path, monkeypatch):
    # The crate's manifest moves on from its lock file just before the build
    def generate_stale(backend_gen_dir, cost):
        generate_build_rs_file(backend_gen_dir, cost)
        manifest = os.path.join(backend_gen_dir, 'Cargo.toml')
        with open(manifest) as f:
            content = f.read()
        with open(manifest, 'w') as f:
            f.write(re.sub(r'^version = ".*"$', 'version = "0.0.0"', content, count=1, flags=re.M))

    monkeypatch.setattr(act_backend.generator, 'generate_build_rs_file', generate_stale)
    with pytest.raises(RuntimeError, match='--locked'):
        generate_backend(accelerator, str(tmp_path))


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
