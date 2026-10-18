"""Tests of how a compile ends: what it says and the output it leaves."""

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
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
    run, _ = compiled(backend, tmp_path, ADD, flags=['--verbose'])
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


def test_earlier_output_is_named_and_overwritten(backend, tmp_path):
    (tmp_path / 'kernel.py').write_text('# an earlier compile\n')
    run, output = compiled(backend, tmp_path, ADD_32)
    assert run.returncode == 0, run.stderr
    assert run.stdout == (
        f"Warning: Output file '{output}' already exists and will be overwritten.\n"
    )
    assert output.read_text() != '# an earlier compile\n'


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
    run, _ = compiled(backend, tmp_path, ADD_32, '/proc/kernel.py', flags=['--verbose'])
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
    run, output = compiled(backend, tmp_path, CHAIN, flags=['--verbose'])
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
        if runs:
            # The second starts while the longer first still reads candidates
            time.sleep(0.3)
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
    # The most directories held at once, until both compiles end
    held = set()
    while any(run.poll() is None for run in runs):
        names = {path.name for path in temporary.iterdir()}
        held = names if len(names) > len(held) else held
        time.sleep(0.005)
    for run in runs:
        assert run.wait() == 0, run.stderr.read()
    assert (tmp_path / 'chain.py').exists() and (tmp_path / 'sum.py').exists()
    # Each compile made its own directory and left it
    assert len(held) == 2 and all(name.startswith('act-backend-') for name in held)
    assert {path.name for path in temporary.iterdir()} == held


def test_progress_names_its_own_log_directory_as_made(backend, tmp_path):
    temporary = tmp_path / 'tmp'
    temporary.mkdir()
    (tmp_path / 'kernel.hlo').write_text(ADD_32)
    command = ['--input', str(tmp_path / 'kernel.hlo'), '--output', str(tmp_path / 'kernel.py')]
    env = dict(os.environ, TMPDIR=str(temporary))
    run = subprocess.run(
        [backend, *command, '--verbose'], env=env, capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stderr
    # The directory as made, its template's X's filled in
    (line,) = [line for line in run.stdout.splitlines() if line.startswith('Log directory: ')]
    made = Path(line.removeprefix('Log directory: '))
    assert made.parent == temporary and made.name.startswith('act-backend-')
    assert made.name != 'act-backend-XXXXXX'


@pytest.fixture
def shared(backend):
    """A directory every user reaches, the backend copied into it"""
    shared = Path(tempfile.mkdtemp())
    shared.chmod(0o777)
    shutil.copy(backend, shared)
    yield shared
    # Every entry opened again before the tree is removed
    for path in shared.iterdir():
        path.chmod(0o777)
    shutil.rmtree(shared)


def test_leftover_directory_of_another_user_is_left_alone(shared):
    (shared / 'kernel.hlo').write_text(ADD_32)
    temporary = shared / 'tmp'
    temporary.mkdir()
    # Sticky like /tmp: any user writes, removes only its own
    temporary.chmod(0o1777)
    # Another user's directory, which the compile leaves alone
    held = temporary / 'act-backend-held' / 'held'
    held.mkdir(parents=True)
    (held / 'file').touch()
    held.chmod(0o555)
    # As uid 65534, since root passes every mode
    user = ['setpriv', '--reuid=65534', '--regid=65534', '--clear-groups']
    command = [*(user if os.geteuid() == 0 else []), str(shared / 'ENDINGS')]
    arguments = ['--input', str(shared / 'kernel.hlo'), '--output', str(shared / 'kernel.py')]
    env = dict(os.environ, TMPDIR=str(temporary))
    run = subprocess.run(
        [*command, *arguments], env=env, capture_output=True, text=True, check=False
    )
    # Opened again before the fixture removes the tree
    held.chmod(0o755)
    assert run.returncode == 0, run.stderr
    assert (shared / 'kernel.py').exists()
    # The leftover keeps its file, beside the compile's own directory
    assert (held / 'file').exists()
    assert held.parent in list(temporary.iterdir())
    assert len(list(temporary.iterdir())) == 2


def test_own_log_directory_the_backend_cannot_make_is_a_usage_error(backend, tmp_path):
    (tmp_path / 'kernel.hlo').write_text(ADD_32)
    command = ['--input', str(tmp_path / 'kernel.hlo'), '--output', str(tmp_path / 'kernel.py')]
    # A missing temporary directory, where no directory can be made
    env = dict(os.environ, TMPDIR=str(tmp_path / 'missing'))
    run = subprocess.run([backend, *command], env=env, capture_output=True, text=True, check=False)
    assert run.returncode == 1
    template = tmp_path / 'missing' / 'act-backend-XXXXXX'
    assert run.stderr.startswith(f"Error: cannot create the log directory '{template}': ")
    assert run.stderr.count('\n') == 1


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


def test_compile_is_quiet_unless_asked_for_its_search(backend, tmp_path):
    quiet, output = compiled(backend, tmp_path, ADD_32)
    assert quiet.returncode == 0, quiet.stderr
    assert (quiet.stdout, quiet.stderr) == ('', '')
    assert output.exists()
    (tmp_path / 'verbose').mkdir()
    verbose, _ = compiled(backend, tmp_path / 'verbose', ADD_32, flags=['--verbose'])
    assert verbose.returncode == 0, verbose.stderr
    # The search's progress, the allocator's among it
    assert 'Starting Phase 1: Instruction Selection...' in verbose.stdout
    assert 'Starting Google OR-Tools constraint solver...' in verbose.stdout
    assert verbose.stderr == ''


# The header of a kernel of one bf16 input of 32 rows
ROWS_32 = 'HloModule constant, entry_computation_layout={(bf16[32,64]{1,0})->bf16[32,64]{1,0}}'

# An input plus a constant other than the identity
CONSTANT = kernel(
    '%a = bf16[32,64] parameter(0)',
    '%k = bf16[32,64] constant(3)',
    'ROOT %c = bf16[32,64] add(%a, %k)',
    header=ROWS_32,
)

# An input plus a constant that operators build
BUILT = kernel(
    '%a = bf16[32,64] parameter(0)',
    '%f = s32[] constant(5)',
    '%b = s32[32,64] broadcast(%f), dimensions={}',
    '%s = s32[32,64] add(%b, %b)',
    '%k = bf16[32,64] convert(%s)',
    'ROOT %c = bf16[32,64] add(%a, %k)',
    header=ROWS_32,
)


def plus(*lines, regions=''):
    """A kernel of an input plus the constant %k the lines build"""
    return kernel(
        '%a = bf16[32,64] parameter(0)',
        *lines,
        'ROOT %c = bf16[32,64] add(%a, %k)',
        header=ROWS_32,
        regions=regions,
    )


def summed(dtype):
    """A reduce region that sums two scalars of one type"""
    return (
        f'%sum (x: {dtype}[], y: {dtype}[]) -> {dtype}[] {{\n  %x = {dtype}[] parameter(0)\n'
        f'  %y = {dtype}[] parameter(1)\n  ROOT %v = {dtype}[] add(%x, %y)\n}}\n'
    )


# Each constant an operator builds and the value it holds
BUILT_BY = {
    'divide_then_narrow': (
        plus(
            '%n = s32[32,64] constant(1000)',
            '%d = s32[32,64] constant(300)',
            '%q = s32[32,64] divide(%n, %d)',
            '%s = s8[32,64] convert(%q)',
            '%k = bf16[32,64] convert(%s)',
        ),
        3,
    ),
    'divide_then_shift': (
        plus(
            '%n = s32[32,64] constant(-8)',
            '%d = s32[32,64] constant(2)',
            '%q = s32[32,64] divide(%n, %d)',
            '%one = s32[32,64] constant(1)',
            '%s = s32[32,64] shift-right-logical(%q, %one)',
            '%k = bf16[32,64] convert(%s)',
        ),
        2147483646,
    ),
    'negative_divide': (
        plus(
            '%n = s32[32,64] constant(-7)',
            '%d = s32[32,64] constant(2)',
            '%q = s32[32,64] divide(%n, %d)',
            '%k = bf16[32,64] convert(%q)',
        ),
        -3,
    ),
    'or': (
        plus(
            '%p = s32[32,64] constant(5)',
            '%q = s32[32,64] constant(3)',
            '%o = s32[32,64] or(%p, %q)',
            '%k = bf16[32,64] convert(%o)',
        ),
        7,
    ),
    'xor': (
        plus(
            '%p = s32[32,64] constant(5)',
            '%q = s32[32,64] constant(3)',
            '%o = s32[32,64] xor(%p, %q)',
            '%k = bf16[32,64] convert(%o)',
        ),
        6,
    ),
    'shift_left': (
        plus(
            '%p = s32[32,64] constant(1)',
            '%q = s32[32,64] constant(3)',
            '%o = s32[32,64] shift-left(%p, %q)',
            '%k = bf16[32,64] convert(%o)',
        ),
        8,
    ),
    'signed_shift_right': (
        plus(
            '%p = s32[32,64] constant(-8)',
            '%q = s32[32,64] constant(1)',
            '%o = s32[32,64] shift-right-logical(%p, %q)',
            '%k = bf16[32,64] convert(%o)',
        ),
        2147483644,
    ),
    'exponential': (
        plus('%one = bf16[32,64] constant(1)', '%k = bf16[32,64] exponential(%one)'),
        jnp.exp(jnp.bfloat16(1)),
    ),
    'float_divide': (
        plus(
            '%p = bf16[32,64] constant(7)',
            '%q = bf16[32,64] constant(2)',
            '%k = bf16[32,64] divide(%p, %q)',
        ),
        3.5,
    ),
    'reduce': (
        plus(
            '%m = bf16[32,64] constant(1)',
            '%z = bf16[] constant(0)',
            '%r = bf16[32] reduce(%m, %z), dimensions={1}, to_apply=%sum',
            '%k = bf16[32,64] broadcast(%r), dimensions={0}',
            regions=summed('bf16'),
        ),
        64,
    ),
    'reduce_from_two': (
        plus(
            '%m = bf16[32,64] constant(1)',
            '%z = bf16[] constant(2)',
            '%r = bf16[32] reduce(%m, %z), dimensions={1}, to_apply=%sum',
            '%k = bf16[32,64] broadcast(%r), dimensions={0}',
            regions=summed('bf16'),
        ),
        66,
    ),
    'widened_reduce': (
        plus(
            '%m = s8[32,64] constant(100)',
            '%z = s32[] constant(0)',
            '%r = s32[32] reduce(%m, %z), dimensions={1}, to_apply=%sum',
            '%b = s32[32,64] broadcast(%r), dimensions={0}',
            '%k = bf16[32,64] convert(%b)',
            regions=summed('s32'),
        ),
        6400,
    ),
    'dot': (
        plus(
            '%p = s8[32,64] constant(2)',
            '%q = s8[64,64] constant(3)',
            '%d = s32[32,64] dot(%p, %q), lhs_contracting_dims={1}, rhs_contracting_dims={0}',
            '%k = bf16[32,64] convert(%d)',
        ),
        384,
    ),
    'transpose': (
        plus('%t = bf16[64,32] constant(5)', '%k = bf16[32,64] transpose(%t), dimensions={1,0}'),
        5,
    ),
    'negative_zero': (plus('%k = bf16[32,64] constant(-0)'), -0.0),
}

# An input plus rows of a constant too large for the buffer
SLICED = plus(
    '%f = bf16[1024,64] constant(3)', '%k = bf16[32,64] slice(%f), slice={[96:128], [0:64]}'
)

# An input plus a constant joined from two halves of its columns
JOINED = plus(
    '%f = bf16[32,32] constant(3)',
    '%g = bf16[32,32] constant(4)',
    '%k = bf16[32,64] concatenate(%f, %g), dimensions={1}',
)

# An input plus a constant given as an array of ones
ARRAY = '{ ' + ', '.join('{' + ', '.join(['1'] * 64) + '}' for _ in range(32)) + ' }'
ARRAYED = plus(f'%k = bf16[32,64] constant({ARRAY})')


def declared_constants(assembly):
    """The constants an assembly declares to its kernel decorator"""
    spec = importlib.util.spec_from_file_location('assembly', assembly)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    declared = {}
    (function,) = re.findall(r'^def (\w+)\(kernel, api\):', assembly.read_text(), re.MULTILINE)
    getattr(module, function)(lambda **fields: declared.update(fields) or (lambda body: body), None)
    return declared['constant']


@pytest.mark.parametrize(
    'text, value',
    [(CONSTANT, 3), (BUILT, 10), *BUILT_BY.values()],
    ids=['filled', 'built', *BUILT_BY],
)
def test_constant_is_written_at_its_value(backend, tmp_path, text, value):
    run, output = compiled(backend, tmp_path, text)
    assert run.returncode == 0, run.stderr
    assert (run.stdout, run.stderr) == ('', '')
    # The bytes of the bf16 value HBM holds
    want = np.full((32, 64), value, dtype=jnp.bfloat16).view(np.uint8).reshape(-1)
    (constant,) = declared_constants(output)
    assert (constant['shape'], np.dtype(constant['dtype'])) == ((want.size,), np.uint8)
    assert np.asarray(constant['value']).dtype == np.uint8
    assert np.array_equal(np.asarray(constant['value']), want)


@pytest.mark.parametrize(
    'text, reason',
    [
        (SLICED, 'a constant built by slice'),
        (JOINED, 'a constant built by concatenate'),
        (ARRAYED, 'a bf16[32,64] constant given as an array'),
    ],
    ids=['slice', 'concatenate', 'array'],
)
def test_constant_the_allocator_cannot_write_drops_its_candidate(backend, tmp_path, text, reason):
    run, output = compiled(backend, tmp_path, text, flags=['--verbose'])
    assert run.returncode == 1
    assert run.stderr.startswith('Error: no assembly found: ')
    assert run.stderr.count('\n') == 1
    assert run.stderr.endswith(f'the first because the allocator cannot write {reason}\n')
    # Its candidates are dropped, each by that one reason
    assert f'dropped: the allocator cannot write {reason}\n' in run.stdout
    assert not output.exists()
