"""Tests of the cost model: the C function act_cost a backend is generated with."""

import functools
import os
import subprocess

import pytest
from conftest import EXAMPLES_DIR, generate_oracle_and_backend, instruction_calls, run_backend

import act_backend
from act_backend import generate_backend

EXAMPLE_DIR = os.path.join(EXAMPLES_DIR, 'mini_trn')
COST_DIR = os.path.join(os.path.dirname(__file__), 'cost')


def build_library(kind, source, out_dir):
    """Compile a C file with cc into an object file, a static library or a shared library"""
    object_file = os.path.join(out_dir, 'model.o')
    subprocess.run(['cc', '-c', '-fPIC', source, '-o', object_file], check=True)
    if kind == 'object':
        return object_file
    if kind == 'static':
        library = os.path.join(out_dir, 'libmodel.a')
        subprocess.run(['ar', 'rcs', library, object_file], check=True)
    else:
        library = os.path.join(out_dir, 'libmodel.so')
        # A build system gives a soname, cc alone none
        soname = ['-Wl,-soname,libmodel.so'] if kind == 'soname' else []
        subprocess.run(['cc', '-shared', object_file, '-o', library, *soname], check=True)
    os.remove(object_file)
    return library


@pytest.fixture(scope='module', params=['object', 'static', 'shared', 'soname'])
def work_dir(request, tmp_path_factory):
    """Generate mini_trn's backend with the test model, given as each kind of library"""
    work_dir = str(tmp_path_factory.mktemp('cost'))
    library = build_library(request.param, os.path.join(COST_DIR, 'model.c'), work_dir)
    with pytest.MonkeyPatch.context() as patch:
        # The spec calls generate_backend(mini_trn): name the library in that call
        with_cost = functools.partial(generate_backend, cost=library)
        patch.setattr(act_backend, 'generate_backend', with_cost)
        generate_oracle_and_backend(EXAMPLE_DIR, work_dir)
    # An object or static library is inside the binary
    if request.param in ('object', 'static'):
        os.remove(library)
    return work_dir


def compile_attention(work_dir, tmp_path, mode=None):
    """Compile the attention kernel over an earlier output, the test model in the given mode"""
    (tmp_path / 'attention.py').write_text('# an earlier compile\n')
    env = dict(os.environ)
    if mode is not None:
        env['ACT_TEST_COST'] = mode
    return run_backend(
        os.path.join(work_dir, 'backends', 'mini_trn'), EXAMPLE_DIR, str(tmp_path), env
    )


def test_model_ranks_the_candidates(work_dir, tmp_path):
    run, output_path = compile_attention(work_dir, tmp_path)
    assert run.returncode == 0, run.stderr
    # Nine calls by default, more by the test model
    assert sum(instruction_calls(output_path).values()) > 9


def test_model_exit_handler_runs(work_dir, tmp_path):
    run, _ = compile_attention(work_dir, tmp_path)
    assert run.returncode == 0, run.stderr
    assert (tmp_path / 'log' / '0.py.closed').exists()


def test_failing_model_ends_the_compile(work_dir, tmp_path):
    run, output_path = compile_attention(work_dir, tmp_path, 'fail')
    assert run.returncode == 1
    assert 'the cost model returned 3' in run.stderr
    assert not os.path.exists(output_path)


def test_cost_that_is_no_number_ends_the_compile(work_dir, tmp_path):
    run, output_path = compile_attention(work_dir, tmp_path, 'nan')
    assert run.returncode == 1
    assert 'the cost model gave the non-finite cost NaN' in run.stderr
    assert not os.path.exists(output_path)


def test_crate_holds_no_default_model(work_dir):
    assert not os.path.exists(
        os.path.join(work_dir, 'targets', 'mini_trn', 'backend', 'cpp', 'cost')
    )


@pytest.mark.parametrize('work_dir', ['object'], indirect=True)
def test_backend_relinks_when_the_library_changes(work_dir, tmp_path):
    # Another model in its place, the crate rebuilt by hand
    build_library('object', os.path.join(COST_DIR, 'fail.c'), work_dir)
    crate = os.path.join(work_dir, 'targets', 'mini_trn', 'backend')
    subprocess.run(['cargo', 'build', '--release', '--locked'], cwd=crate, check=True)
    backend = os.path.join(crate, 'target', 'release', 'backend')
    run, _ = run_backend(backend, EXAMPLE_DIR, str(tmp_path))
    assert run.returncode == 1
    assert 'the cost model returned 5' in run.stderr


def test_missing_library_refused(accelerator, tmp_path, monkeypatch):
    # A relative path is taken from the caller's directory
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as error:
        generate_backend(accelerator, str(tmp_path), cost='model.o')
    assert str(tmp_path / 'model.o') in str(error.value)


def test_source_file_refused(accelerator, tmp_path):
    source = os.path.join(COST_DIR, 'model.c')
    with pytest.raises(ValueError) as error:
        generate_backend(accelerator, str(tmp_path), cost=source)
    assert source in str(error.value)


def test_path_with_a_line_break_refused(accelerator, tmp_path):
    library = tmp_path / 'two\nlines' / 'model.o'
    library.parent.mkdir()
    library.touch()
    with pytest.raises(ValueError, match='line break'):
        generate_backend(accelerator, str(tmp_path / 'base'), cost=str(library))
    assert not (tmp_path / 'base').exists()


def test_shared_library_under_a_colon_refused(accelerator, tmp_path):
    library = tmp_path / 'co:lon' / 'libmodel.so'
    library.parent.mkdir()
    library.touch()
    with pytest.raises(ValueError, match='colon'):
        generate_backend(accelerator, str(tmp_path / 'base'), cost=str(library))
    assert not (tmp_path / 'base').exists()


def test_library_without_the_function_fails_the_build(accelerator, tmp_path):
    library = build_library('object', os.path.join(COST_DIR, 'no_cost.c'), str(tmp_path))
    # The link's own error names the function
    with pytest.raises(RuntimeError, match=r'undefined (symbol|reference).*act_cost'):
        generate_backend(accelerator, str(tmp_path), cost=library)
