"""Tests of how a compile ends: what it says and the output it leaves."""

import os
import subprocess
import time

import pytest
from conftest import add_moves, add_on_d1, compiled, instruction_calls, kernel
from taidl import Accelerator

from act_backend import generate_backend


def seven():
    """A buffer of 100 rows, its moves, and a sum of seven of its regions of 16 rows"""
    acc = Accelerator('SEVEN')
    acc.add_data_model('d1', [100], [64], 'bf16')
    add_moves(acc)
    # Its seven 16-row inputs alone outgrow the buffer
    add_on_d1(
        acc,
        'sum_7',
        ['16'] * 8,
        [
            *[f'%In{k} = bf16[16,64] parameter({k})' for k in range(7)],
            '%s1 = bf16[16,64] add(%In0, %In1)',
            *[f'%s{k} = bf16[16,64] add(%s{k - 1}, %In{k})' for k in range(2, 6)],
            'ROOT %Out0 = bf16[16,64] add(%s5, %In6)',
        ],
    )
    return acc


@pytest.fixture(scope='module')
def seven_backend(tmp_path_factory):
    """Generate the seven spec's backend once and return the binary's path"""
    work_dir = str(tmp_path_factory.mktemp('seven'))
    generate_backend(seven(), work_dir)
    return os.path.join(work_dir, 'backends', 'SEVEN')


# The sum of two matrices of 32 rows
ADD_32 = """HloModule add_32, entry_computation_layout={(bf16[32,64]{1,0}, bf16[32,64]{1,0})->bf16[32,64]{1,0}}

ENTRY main {
  %a = bf16[32,64] parameter(0)
  %b = bf16[32,64] parameter(1)
  ROOT %c = bf16[32,64] add(%a, %b)
}
"""

# A sum only the small buffer's add computes
ADD = """HloModule add, entry_computation_layout={(bf16[64,64]{1,0}, bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[64,64] parameter(0)
  %b = bf16[64,64] parameter(1)
  ROOT %c = bf16[64,64] add(%a, %b)
}
"""


# A sum of seven 16-row matrices, which only sum_7 computes
SEVEN = (
    'HloModule seven, entry_computation_layout={('
    + ', '.join(['bf16[16,64]{1,0}'] * 7)
    + ')->bf16[16,64]{1,0}}\n\nENTRY main {\n'
    + ''.join(f'  %i{k} = bf16[16,64] parameter({k})\n' for k in range(7))
    + '  %s1 = bf16[16,64] add(%i0, %i1)\n'
    + ''.join(f'  %s{k} = bf16[16,64] add(%s{k - 1}, %i{k})\n' for k in range(2, 6))
    + '  ROOT %s6 = bf16[16,64] add(%s5, %i6)\n}\n'
)

# s32 product and sum, which IR rewrite 0 keeps halving
HALVES = """HloModule halves, entry_computation_layout={(s32[65536,16]{1,0}, s32[16,16]{1,0}, s32[65536,16]{1,0})->s32[65536,16]{1,0}}

ENTRY main {
  %a = s32[65536,16] parameter(0)
  %b = s32[16,16] parameter(1)
  %d = s32[65536,16] parameter(2)
  %p = s32[65536,16] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %s = s32[65536,16] add(%p, %d)
}
"""

# A sum of an input and an undefined value
UNDEFINED = """HloModule undefined, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[64,64] parameter(0)
  ROOT %c = bf16[64,64] add(%a, %b)
}
"""

# One exponential, which no instruction computes
EXPONENTIAL = """HloModule exponential, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[64,64] parameter(0)
  ROOT %e = bf16[64,64] exponential(%a)
}
"""


def test_dropped_candidate_names_its_reason(backend, tmp_path):
    run, _ = compiled(backend, tmp_path, ADD)
    assert 'PII #0 dropped: the solver found no placement of its tensors' in run.stdout


def test_no_assembly_ends_in_one_line_and_status_1(backend, tmp_path):
    run, output = compiled(backend, tmp_path, ADD)
    assert run.returncode == 1
    assert run.stderr == (
        'Error: no assembly found: the rewrites saturated and the time limit of 1s ran out, and '
        'every candidate was dropped, the first because the solver found no placement of its '
        'tensors\n'
    )
    assert not output.exists()


def test_no_assembly_says_no_candidate_was_extracted(backend, tmp_path):
    run, output = compiled(backend, tmp_path, EXPONENTIAL)
    assert run.returncode == 1
    assert run.stderr == (
        'Error: no assembly found: the rewrites saturated and the time limit of 1s ran out, and '
        'no candidate was extracted\n'
    )
    assert not output.exists()


def test_no_assembly_names_the_node_limit(backend, tmp_path):
    run, output = compiled(backend, tmp_path, HALVES)
    assert run.returncode == 1
    assert run.stderr == (
        'Error: no assembly found: the rewrites reached the node limit of 10000 and the time '
        'limit of 1s ran out, and no candidate was extracted\n'
    )
    assert not output.exists()


def test_solver_that_runs_out_of_its_time_limit_says_so(seven_backend, tmp_path):
    run, output = compiled(seven_backend, tmp_path, SEVEN)
    assert run.returncode == 1
    # Seven 16-row regions in 100 rows outlast the solver's time limit
    assert run.stderr == (
        'Error: no assembly found: the time limit of 1s ran out, and every candidate was dropped, '
        'the first because the solver found no placement within its time limit of 500ms\n'
    )
    assert not output.exists()


def test_failed_compile_leaves_no_earlier_output(backend, tmp_path):
    (tmp_path / 'kernel.py').write_text('# an earlier compile\n')
    run, output = compiled(backend, tmp_path, UNDEFINED)
    assert run.returncode != 0
    assert not output.exists()


def test_usage_error_leaves_the_earlier_output(backend, tmp_path):
    (tmp_path / 'kernel.py').write_text('# an earlier compile\n')
    (tmp_path / 'kernel.hlo').write_text(UNDEFINED)
    # No directory after --log, a usage error
    command = ['--input', str(tmp_path / 'kernel.hlo'), '--output', str(tmp_path / 'kernel.py')]
    run = subprocess.run([backend, *command, '--log'], capture_output=True, text=True, check=False)
    assert run.returncode == 1
    assert (tmp_path / 'kernel.py').read_text() == '# an earlier compile\n'


def test_output_in_a_missing_directory_is_written(backend, tmp_path):
    run, output = compiled(backend, tmp_path, ADD_32, tmp_path / 'new' / 'kernel.py')
    assert run.returncode == 0, run.stderr
    assert output.exists()


def test_usage_error_makes_no_directory(backend, tmp_path):
    (tmp_path / 'kernel.hlo').write_text(ADD_32)
    output = tmp_path / 'new' / 'kernel.py'
    command = ['--input', str(tmp_path / 'kernel.hlo'), '--output', str(output), '--log']
    run = subprocess.run([backend, *command], capture_output=True, text=True, check=False)
    assert run.returncode == 1
    assert not (tmp_path / 'new').exists()


def test_output_the_backend_cannot_write_ends_the_compile_at_once(backend, tmp_path):
    # No user, root included, makes a file in /proc
    run, _ = compiled(backend, tmp_path, ADD_32, '/proc/kernel.py')
    assert run.returncode == 1
    assert run.stderr.startswith("Error: cannot write the output '/proc/kernel.py': ")
    assert run.stderr.count('\n') == 1
    assert 'Starting Phase 1' not in run.stdout


def test_missing_flag_is_a_usage_error(backend, tmp_path):
    run = subprocess.run(
        [backend, '--input', str(tmp_path / 'kernel.hlo')],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1
    assert run.stderr == 'Error: missing the required flag --output\n'


def test_help_exits_0(backend):
    run = subprocess.run([backend, '--help'], capture_output=True, text=True, check=False)
    assert run.returncode == 0
    assert run.stdout.startswith('Usage: ')


# 40 converts, then a slice of half of the rows
CHAIN = kernel(
    '%c0 = bf16[64,64] parameter(0)',
    *[f'%c{k} = {"f32" if k % 2 else "bf16"}[64,64] convert(%c{k - 1})' for k in range(1, 41)],
    'ROOT %s = bf16[32,64] slice(%c40), slice={[0:32:1], [0:64:1]}',
    header='HloModule chain, entry_computation_layout={(bf16[64,64]{1,0})->bf16[32,64]{1,0}}',
)


def test_compile_at_the_iteration_limit_keeps_its_best_candidate(backend, tmp_path):
    run, output = compiled(backend, tmp_path, CHAIN)
    assert run.returncode == 0, run.stderr
    # The slice moves down 39 passes, past the limit's 30
    assert 'the rewrites reached the iteration limit of 30' in run.stdout
    assert instruction_calls(output) == {'load_rm': 1, 'converts': 1, 'store_rm': 1}


def test_assembly_header_names_the_kernel_path(backend, tmp_path):
    run, output = compiled(backend, tmp_path, ADD_32)
    assert run.returncode == 0, run.stderr
    assert f'# Input file: {tmp_path / "kernel.hlo"}\n' in output.read_text()


def test_assembly_header_escapes_a_line_break_of_the_kernel_path(backend, tmp_path):
    # A path holding a Python line between two line breaks
    path = tmp_path / "k\nprint('run')\n#.hlo"
    path.write_text(ADD_32)
    output = tmp_path / 'kernel.py'
    command = ['--input', str(path), '--output', str(output), '--log', str(tmp_path / 'log')]
    run = subprocess.run([backend, *command], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    header = output.read_text().splitlines()
    assert header[0] == f"# Input file: {tmp_path}/k\\nprint('run')\\n#.hlo"
    assert header[1] == '# Kernel name: add_32'


def test_two_compiles_at_once_keep_their_log_directories(backend, tmp_path):
    temporary = tmp_path / 'tmp'
    temporary.mkdir()
    env = dict(os.environ, TMPDIR=str(temporary))
    runs = []
    for name, text in [('chain', CHAIN), ('sum', ADD_32)]:
        (tmp_path / f'{name}.hlo').write_text(text)
        command = [
            backend,
            '--input',
            str(tmp_path / f'{name}.hlo'),
            '--output',
            str(tmp_path / f'{name}.py'),
        ]
        runs.append(
            subprocess.Popen(
                command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
        )
        # The second starts while the longer first still reads candidates
        time.sleep(0.3)
    for run in runs:
        assert run.wait() == 0, run.stderr.read()
    assert (tmp_path / 'chain.py').exists() and (tmp_path / 'sum.py').exists()
    # Each compile leaves the directory it made for itself
    assert len(list(temporary.iterdir())) == 2


def test_log_directory_that_holds_files_is_kept(backend, tmp_path):
    log = tmp_path / 'log'
    log.mkdir()
    (log / 'notes.txt').write_text('a file of the user\n')
    (tmp_path / 'kernel.hlo').write_text(ADD_32)
    command = ['--input', str(tmp_path / 'kernel.hlo'), '--output', str(tmp_path / 'kernel.py')]
    run = subprocess.run(
        [backend, *command, '--log', str(log)], capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stderr
    assert (log / 'notes.txt').read_text() == 'a file of the user\n'


def test_log_directory_that_is_a_file_is_a_usage_error(backend, tmp_path):
    (tmp_path / 'log').write_text('a file of the user\n')
    run, _ = compiled(backend, tmp_path, ADD_32)
    assert run.returncode == 1
    assert run.stderr == f"Error: the log directory '{tmp_path / 'log'}' is not a directory\n"


def test_log_directory_the_backend_cannot_make_leaves_the_earlier_output(backend, tmp_path):
    (tmp_path / 'kernel.py').write_text('# an earlier compile\n')
    (tmp_path / 'kernel.hlo').write_text(ADD_32)
    # No user, root included, makes a directory in /proc
    command = ['--input', str(tmp_path / 'kernel.hlo'), '--output', str(tmp_path / 'kernel.py')]
    run = subprocess.run(
        [backend, *command, '--log', '/proc/log'], capture_output=True, text=True, check=False
    )
    assert run.returncode == 1
    assert run.stderr.startswith("Error: cannot create the log directory '/proc/log': ")
    assert run.stderr.count('\n') == 1
    assert (tmp_path / 'kernel.py').read_text() == '# an earlier compile\n'
