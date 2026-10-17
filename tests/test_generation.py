"""Tests of a whole generation: what generate_backend builds and what it says when it cannot."""

import os
import shutil
import subprocess

import pytest
from conftest import EXAMPLES_DIR, instruction_calls, run_backend
from taidl import Accelerator

import act_backend
from act_backend import generate_backend

EXAMPLE_DIR = os.path.join(EXAMPLES_DIR, 'mini_trn')


def test_failed_build_shows_its_error(accelerator, tmp_path, monkeypatch):
    # The allocator's build stops on a missing C++ compiler
    monkeypatch.setenv('CXX', 'no-such-compiler')
    with pytest.raises(RuntimeError) as error:
        generate_backend(accelerator, str(tmp_path))
    assert 'no-such-compiler' in str(error.value)
    assert os.path.join(str(tmp_path), 'targets', 'PAIR', 'backend') in str(error.value)


@pytest.mark.parametrize('cost, error', [('model.c', ValueError), (None, RuntimeError)])
def test_failed_generation_leaves_no_earlier_binary(
    cost, error, accelerator, tmp_path, monkeypatch
):
    # What an earlier generation left
    backend = tmp_path / 'backends' / 'PAIR'
    backend.parent.mkdir()
    backend.write_text('an earlier backend\n')
    # A refused cost library, or a build missing its compiler
    monkeypatch.setenv('CXX', 'no-such-compiler')
    with pytest.raises(error):
        generate_backend(accelerator, str(tmp_path), cost=cost)
    assert not backend.exists()


def built(command, cwd, **options):
    """A build that passes without running and leaves a binary where cargo would"""
    binary = os.path.join(cwd, 'target', 'release', 'backend')
    os.makedirs(os.path.dirname(binary))
    with open(binary, 'w') as f:
        f.write('a new backend')
    return subprocess.CompletedProcess(command, 0, stdout='')


@pytest.mark.parametrize('name', ['PAIR', 'PAIR.partial'])
def test_binary_is_not_written_through_a_link(name, accelerator, tmp_path, monkeypatch):
    # An earlier run's link at the binary's name, to outside
    outside = tmp_path / 'outside'
    outside.write_text('kept')
    (tmp_path / 'backends').mkdir()
    (tmp_path / 'backends' / name).symlink_to(outside)
    monkeypatch.setattr(subprocess, 'run', built)
    generate_backend(accelerator, str(tmp_path))
    backend = tmp_path / 'backends' / 'PAIR'
    assert not backend.is_symlink() and backend.read_text() == 'a new backend'
    assert outside.read_text() == 'kept'


def test_interrupted_copy_leaves_no_binary(accelerator, tmp_path, monkeypatch):
    # A copy that stops halfway
    def copy(source, destination):
        with open(destination, 'w') as f:
            f.write('half a backend')
        raise OSError('No space left on device')

    monkeypatch.setattr(subprocess, 'run', built)
    monkeypatch.setattr(shutil, 'copy', copy)
    with pytest.raises(OSError):
        generate_backend(accelerator, str(tmp_path))
    assert os.listdir(tmp_path / 'backends') == []


def test_build_directory_in_the_package_stays_out_of_the_crate(accelerator, tmp_path, monkeypatch):
    # Build directories a build by hand leaves in the package
    generic = os.path.join(os.path.dirname(act_backend.__file__), 'generic')
    left = [os.path.join(generic, 'target'), os.path.join(generic, 'cpp', 'malloc', 'build')]
    for directory in left:
        os.makedirs(directory)
        open(os.path.join(directory, 'CMakeCache.txt'), 'w').close()

    # The crate as its build finds it, the build stopped
    def build(command, **options):
        return subprocess.CompletedProcess(command, 1, stdout='')

    monkeypatch.setattr(subprocess, 'run', build)
    try:
        with pytest.raises(RuntimeError):
            generate_backend(accelerator, str(tmp_path))
    finally:
        for directory in left:
            shutil.rmtree(directory)
    crate = tmp_path / 'targets' / 'PAIR' / 'backend'
    assert not (crate / 'target').exists()
    assert not (crate / 'cpp' / 'malloc' / 'build').exists()


def test_crate_holds_its_ignore_file(work_dir):
    # The whole crate comes from the package, dot file too
    assert os.path.isfile(os.path.join(work_dir, 'targets', 'mini_trn', 'backend', '.gitignore'))


def test_cargo_leaves_the_lock_file_alone(work_dir):
    # A build by hand finds cargo's lock file, rewriting nothing
    crate = os.path.join(work_dir, 'targets', 'mini_trn', 'backend')
    subprocess.run(['cargo', 'build', '--release'], cwd=crate, check=True, capture_output=True)
    package_dir = os.path.dirname(act_backend.__file__)
    with open(os.path.join(package_dir, 'generic', 'Cargo.lock')) as f:
        package_lock = f.read()
    with open(os.path.join(crate, 'Cargo.lock')) as f:
        assert f.read() == package_lock


def test_backend_runs_without_its_crate(work_dir, tmp_path):
    # The binary copied away alone, its crate moved
    backend = shutil.copy(os.path.join(work_dir, 'backends', 'mini_trn'), tmp_path)
    crate = os.path.join(work_dir, 'targets', 'mini_trn', 'backend')
    os.rename(crate, crate + '.moved')
    try:
        run, output_path = run_backend(backend, EXAMPLE_DIR, str(tmp_path))
    finally:
        os.rename(crate + '.moved', crate)
    assert run.returncode == 0, run.stderr
    assert sum(instruction_calls(output_path).values()) == 9


def test_backend_finds_the_solver_by_itself(work_dir, tmp_path):
    # No library path: the binary has its run path alone
    env = {name: value for name, value in os.environ.items() if name != 'LD_LIBRARY_PATH'}
    run, _ = run_backend(
        os.path.join(work_dir, 'backends', 'mini_trn'), EXAMPLE_DIR, str(tmp_path), env
    )
    assert run.returncode == 0, run.stderr


def starts(backend):
    """Whether a backend starts with no library path in the environment"""
    env = {name: value for name, value in os.environ.items() if name != 'LD_LIBRARY_PATH'}
    run = subprocess.run([backend, '--help'], env=env, capture_output=True, check=False)
    return run.returncode == 0


def test_solver_found_where_the_environment_says(accelerator, tmp_path, monkeypatch):
    # OR-Tools under another path: a link to its directory
    solver_dir = tmp_path / 'solver'
    solver_dir.symlink_to(os.environ.get('ORTOOLS_DIR', '/opt/ortools'))
    monkeypatch.setenv('ORTOOLS_DIR', str(solver_dir))
    generate_backend(accelerator, str(tmp_path))

    # The backend loads it from that path alone
    backend = str(tmp_path / 'backends' / 'PAIR')
    assert starts(backend)
    solver_dir.unlink()
    assert not starts(backend)


def test_solver_libraries_found_under_lib64(accelerator, tmp_path, monkeypatch):
    # Another system's layout: libraries under lib64, lib the package files
    installed = os.environ.get('ORTOOLS_DIR', '/opt/ortools')
    solver_dir = tmp_path / 'solver'
    (solver_dir / 'lib64').mkdir(parents=True)
    for name in ['bin', 'include', 'share']:
        (solver_dir / name).symlink_to(os.path.join(installed, name))
    for entry in os.scandir(os.path.join(installed, 'lib')):
        if not entry.is_dir():
            (solver_dir / 'lib64' / entry.name).symlink_to(entry.path)
    shutil.copytree(os.path.join(installed, 'lib', 'cmake'), solver_dir / 'lib' / 'cmake')
    for package_file in (solver_dir / 'lib' / 'cmake').rglob('*.cmake'):
        content = package_file.read_text()
        package_file.write_text(content.replace('_IMPORT_PREFIX}/lib/', '_IMPORT_PREFIX}/lib64/'))

    monkeypatch.setenv('ORTOOLS_DIR', str(solver_dir))
    generate_backend(accelerator, str(tmp_path))
    assert starts(str(tmp_path / 'backends' / 'PAIR'))


@pytest.mark.parametrize('name', ['empty', 'co:lon'])
def test_wrong_solver_directory_named(name, accelerator, tmp_path, monkeypatch):
    # A directory without OR-Tools, and one no run path carries
    solver_dir = tmp_path / name
    solver_dir.mkdir()
    monkeypatch.setenv('ORTOOLS_DIR', str(solver_dir))
    with pytest.raises(RuntimeError) as error:
        generate_backend(accelerator, str(tmp_path / 'base'))
    assert 'ORTOOLS_DIR' in str(error.value)
    assert str(solver_dir) in str(error.value)


def test_json_headers_sought_where_the_environment_says(accelerator, tmp_path, monkeypatch):
    # A directory that holds no header stops the allocator's build
    monkeypatch.setenv('NLOHMANN_JSON_DIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='nlohmann/json.hpp'):
        generate_backend(accelerator, str(tmp_path))


def test_backend_builds_where_the_environment_names_nothing(accelerator, tmp_path, monkeypatch):
    # The defaults: the image's OR-Tools and JSON headers
    monkeypatch.delenv('ORTOOLS_DIR', raising=False)
    monkeypatch.delenv('NLOHMANN_JSON_DIR', raising=False)
    generate_backend(accelerator, str(tmp_path))
    assert starts(str(tmp_path / 'backends' / 'PAIR'))


def test_spec_with_every_default_generates(tmp_path):
    acc = Accelerator('ONE')
    acc.add_data_model('d1', [128], [64], 'bf16')
    # State that no instruction of the backend reads or updates
    acc.add_initial_state('mode', 0)
    mov = acc.add_instruction('mov', ['n'], ['addr_in', 'addr_out'])
    mov.set_inputs([['d1', ['@a.addr_in'], ['@c.n']]])
    mov.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])
    mov.add_semantics("""
ENTRY {
    %In0 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In0);
}
""")
    # A list that marks no input is the default
    mov.set_inplace([False])
    # No constraint and no placement mark are the defaults
    mov.add_constraints([])
    mov.set_apart([False])
    mov.set_no_partial_overlap([False])
    # An instruction with no semantics is left out, unchecked
    clear = acc.add_instruction('clear', [], ['addr'])
    clear.set_inputs([['d1', ['@a.addr'], ['64']]])
    clear.set_outputs([['d1', ['@a.addr'], ['64']]])
    clear.set_inplace([True])
    clear.add_constraints(['@a.addr <= 64'])
    acc.add_instruction('configure', ['mode'], [], update=['@s.mode = @c.mode'])
    generate_backend(acc, str(tmp_path))
    assert starts(str(tmp_path / 'backends' / 'ONE'))


def test_rebuild_by_hand_follows_the_solver_directory(work_dir, tmp_path):
    # OR-Tools under another path, the crate rebuilt by hand
    solver_dir = tmp_path / 'solver'
    solver_dir.symlink_to(os.environ.get('ORTOOLS_DIR', '/opt/ortools'))
    crate = os.path.join(work_dir, 'targets', 'mini_trn', 'backend')
    env = dict(os.environ, ORTOOLS_DIR=str(solver_dir))
    subprocess.run(
        ['cargo', 'build', '--release', '--locked'],
        cwd=crate,
        env=env,
        check=True,
        capture_output=True,
    )
    # The allocator's rebuild looked there, not at the old path
    with open(os.path.join(crate, 'cpp', 'malloc', 'build', 'CMakeCache.txt')) as f:
        assert f'ortools_DIR:PATH={solver_dir}/lib/cmake/ortools' in f.read()


def test_rebuild_by_hand_drops_a_removed_default_model(work_dir):
    # The default model's source removed, the crate rebuilt by hand
    crate = os.path.join(work_dir, 'targets', 'mini_trn', 'backend')
    source = os.path.join(crate, 'cpp', 'cost', 'count.cc')
    os.rename(source, source + '.moved')
    try:
        build = subprocess.run(
            ['cargo', 'build', '--release', '--locked'],
            cwd=crate,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.rename(source + '.moved', source)
    # The link finds no model the last build left behind
    assert build.returncode != 0
    assert 'cost_static' in build.stderr
