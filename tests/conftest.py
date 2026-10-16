"""Shared test utilities for ACT-Backend tests"""

import importlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from taidl import Accelerator

from act_backend import generate_backend


def pytest_configure(config):
    config.addinivalue_line('markers', 'incremental: skip remaining tests after first failure')


def pytest_runtest_makereport(item, call):
    if 'incremental' in item.keywords and call.excinfo is not None:
        parent = item.parent
        parent.previousfailed = item


def pytest_runtest_setup(item):
    if 'incremental' in item.keywords:
        previousfailed = getattr(item.parent, 'previousfailed', None)
        if previousfailed is not None:
            pytest.skip(f'previous test failed ({previousfailed.name})')


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples')
# The platform every oracle simulates on
PLATFORM = 'CPU'
# Each imported oracle gets a package name of its own
oracle_ids = itertools.count()


def load_bf16_matrix(path, shape):
    np_uint8 = np.fromfile(path, dtype=np.uint8)
    np_uint8 = np_uint8.reshape(shape[0], shape[1], 2)
    j_uint8 = jnp.array(np_uint8, dtype=jnp.uint8)
    return jax.lax.bitcast_convert_type(j_uint8, jnp.bfloat16)


def generate_oracle_and_backend(example_dir, work_dir):
    """Run the ISA spec to generate oracle + backend into work_dir"""
    # An example's spec and accelerator are named as its directory
    name = os.path.basename(example_dir)
    path = shutil.copy(os.path.join(example_dir, f'{name}.py'), work_dir)
    spec = importlib.util.spec_from_file_location(f'{name}_spec', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def unfilled_sources(work_dir, example_dir):
    """Return the generated backend sources that still hold a placeholder"""
    backend_dir = os.path.join(work_dir, 'targets', os.path.basename(example_dir), 'backend')
    assert os.path.isdir(backend_dir), f'no generated backend at {backend_dir}'
    unfilled = []
    for root, dirs, names in os.walk(backend_dir):
        # The build trees hold no generated source
        dirs[:] = [d for d in dirs if d not in ('target', 'build')]
        for name in names:
            if name.endswith(('.rs', '.cc', '.h', '.txt')):
                with open(os.path.join(root, name)) as f:
                    if re.search(r'\{\{[A-Z0-9_]+\}\}', f.read()):
                        unfilled.append(os.path.join(root, name))
    return unfilled


# The tree whose formatter settings the examples are held to
TREE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unformatted_sources(backend_dir):
    """Return what rustfmt and clang-format would change in a generated backend"""
    sources = []
    for root, dirs, names in os.walk(backend_dir):
        # The build trees hold no generated source
        dirs[:] = [d for d in dirs if d not in ('target', 'build')]
        sources += [os.path.join(root, name) for name in names]
    rust = [path for path in sources if path.endswith('.rs')]
    cpp = [path for path in sources if path.endswith(('.cc', '.h'))]
    # A backend with no sources would pass unchecked
    assert rust and cpp, f'no generated sources in {backend_dir}'
    checks = [
        ['rustfmt', '--check', '--config-path', TREE_DIR, '--config', 'skip_children=true', *rust],
        ['clang-format', '--dry-run', '-Werror', f'--style=file:{TREE_DIR}/.clang-format', *cpp],
    ]
    runs = [subprocess.run(check, capture_output=True, text=True, check=False) for check in checks]
    return ''.join(run.stdout + run.stderr for run in runs if run.returncode != 0)


def compile_hlo(work_dir, example_dir):
    """Run the backend binary on the HLO file to produce a compiled kernel"""
    backend_bin = os.path.join(work_dir, 'backends', os.path.basename(example_dir))
    hlo_path = os.path.join(example_dir, 'kernels', 'attention.hlo')
    asm_dir = os.path.join(work_dir, 'asm')
    os.makedirs(asm_dir, exist_ok=True)
    output_path = os.path.join(asm_dir, 'attention.py')

    subprocess.run(
        [backend_bin, '--input', hlo_path, '--output', output_path], check=True, cwd=work_dir
    )
    return output_path


def run_backend(backend_bin, example_dir, out_dir, env=None):
    """Run a backend on the example's attention kernel; return the run and the output's path"""
    output_path = os.path.join(out_dir, 'attention.py')
    run = subprocess.run(
        [
            backend_bin,
            '--input',
            os.path.join(example_dir, 'kernels', 'attention.hlo'),
            '--output',
            output_path,
            '--log',
            os.path.join(out_dir, 'log'),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return run, output_path


def instruction_calls(assembly_path):
    """Count the calls to each instruction in a compiled kernel"""
    with open(assembly_path) as f:
        return Counter(re.findall(r'api\.(\w+)\(', f.read()))


def import_oracle(work_dir, example_dir):
    """Import the oracle in work_dir as a package of its own and return (decorator, api)"""
    oracle_dir = os.path.join(work_dir, 'targets', os.path.basename(example_dir), 'oracle')
    name = f'oracle_{next(oracle_ids)}'
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(oracle_dir, '__init__.py'), submodule_search_locations=[oracle_dir]
    )
    sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[name])
    decorator = importlib.import_module(f'{name}.decorator')
    decorator.set_simulation_backend(PLATFORM)
    return decorator, importlib.import_module(f'{name}.api')


def run_compiled_kernel(work_dir, assembly_path, example_dir):
    """Import oracle + compiled kernel, run simulation, return its output"""
    oracle_decorator, oracle_api = import_oracle(work_dir, example_dir)

    # The compiled kernel, loaded from its own path
    spec = importlib.util.spec_from_file_location('attention', assembly_path)
    kernel_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_module)

    kernel = kernel_module.attention(oracle_decorator.kernel, oracle_api)
    kernel('fsim-compile')()

    data_dir = os.path.join(example_dir, 'data')
    Q = load_bf16_matrix(os.path.join(data_dir, 'Q.dat'), (64, 64))
    K = load_bf16_matrix(os.path.join(data_dir, 'K.dat'), (64, 64))
    V = load_bf16_matrix(os.path.join(data_dir, 'V.dat'), (64, 64))

    outputs, _ = kernel('fsim')(Q, K, V)
    return outputs[0]


def load_golden(data_dir):
    """Return the attention golden and the record of where it came from"""
    with open(os.path.join(data_dir, 'goldens.json')) as f:
        record = json.load(f)['attention.dat']
    return load_bf16_matrix(os.path.join(data_dir, 'attention.dat'), (64, 64)), record


def bf16(x):
    """Round float64 values to bf16, kept in float64"""
    return x.astype(ml_dtypes.bfloat16).astype(np.float64)


def attention_reference(data_dir):
    """Return softmax(Q K^T) V in numpy, rounded to bf16 like the kernel, and its tolerance"""
    Q, K, V = (
        np.fromfile(os.path.join(data_dir, f'{n}.dat'), dtype=ml_dtypes.bfloat16)
        .reshape(64, 64)
        .astype(np.float64)
        for n in 'QKV'
    )
    S = bf16(Q @ K.T)
    E = bf16(np.exp(S))
    P = bf16(E / bf16(E.sum(axis=1, keepdims=True)))
    # Two bf16 ulps at max|V| bound every output
    tol = 2 * 2.0 ** (np.floor(np.log2(np.abs(V).max())) - 7)
    return bf16(P @ V), tol


def add_moves(acc):
    """A load of bf16 rows into d1 and a store of them from it"""
    instr = acc.add_instruction('load_rm', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['@c.n * 128']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In0);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

    instr = acc.add_instruction('store_rm', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['@c.n']]])
    instr.set_outputs([['d0', ['@a.addr_out'], ['@c.n * 128']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In0);
    ROOT %Out0 = u8[`@c.n * 128`] reshape(%a);
}
""")


def add_on_d1(acc, name, sizes, lines, attributes=()):
    """An instruction from regions of d1 to a region of d1, the output's size last"""
    addresses = [f'addr_{number}' for number in range(1, len(sizes))]
    instr = acc.add_instruction(name, list(attributes), [*addresses, 'addr_out'])
    instr.set_inputs([['d1', [f'@a.{address}'], [size]] for address, size in zip(addresses, sizes)])
    instr.set_outputs([['d1', ['@a.addr_out'], [sizes[-1]]]])
    instr.add_semantics('ENTRY {\n' + ''.join(f'    {line};\n' for line in lines) + '}\n')


def endings():
    """A buffer of 100 rows, its moves, and the instructions the ending tests compile"""
    acc = Accelerator('ENDINGS')
    acc.add_data_model('d1', [100], [64], 'bf16')
    add_moves(acc)
    # The add of 32-row regions, which the buffer holds
    add_on_d1(
        acc,
        'vadd_32',
        ['32', '32', '32'],
        [
            '%In0 = bf16[32,64] parameter(0)',
            '%In1 = bf16[32,64] parameter(1)',
            'ROOT %Out0 = bf16[32,64] add(%In0, %In1)',
        ],
    )
    # The converts instruction: 31 converts, a slice, nine more
    chain = ['%In0 = bf16[64,64] parameter(0)']
    for k in range(1, 41):
        rows = 64 if k <= 31 else 32
        source = '%In0' if k == 1 else f'%c{k - 1}'
        if k == 32:
            chain.append('%s = f32[32,64] slice(%c31), slice={[0:32:1], [0:64:1]}')
            source = '%s'
        chain.append(f'%c{k} = {"f32" if k % 2 else "bf16"}[{rows},64] convert({source})')
    chain[-1] = chain[-1].replace('%c40 =', 'ROOT %Out0 =')
    add_on_d1(acc, 'converts', ['64', '32'], chain)
    # Its two inputs alone hold more rows than the buffer
    add_on_d1(
        acc,
        'vadd',
        ['64', '64', '64'],
        [
            '%In0 = bf16[64,64] parameter(0)',
            '%In1 = bf16[64,64] parameter(1)',
            'ROOT %Out0 = bf16[64,64] add(%In0, %In1)',
        ],
    )
    return acc


@pytest.fixture(scope='session')
def endings_dir(tmp_path_factory):
    """Generate the endings spec's backend once, in a directory of its own"""
    work_dir = str(tmp_path_factory.mktemp('endings'))
    generate_backend(endings(), work_dir)
    return work_dir


@pytest.fixture(scope='session')
def backend(endings_dir):
    return os.path.join(endings_dir, 'backends', 'ENDINGS')


def compiled(backend, tmp_path, kernel, output=None, flags=()):
    """Compile a kernel's text with the flags given; return the run and the output's path"""
    path = tmp_path / 'kernel.hlo'
    path.write_text(kernel)
    output = output or tmp_path / 'kernel.py'
    arguments = ['--input', str(path), '--output', str(output), '--log', str(tmp_path / 'log')]
    return subprocess.run(
        [backend, *arguments, *flags], capture_output=True, text=True, check=False
    ), output


# A one-input, one-output bf16 kernel around the given lines
HEADER = 'HloModule bad, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}'


def kernel(*lines, header=HEADER, regions=''):
    """The text of a kernel: its header, its regions and an ENTRY of the lines given"""
    return (
        f'{header}\n\n{regions}\nENTRY main {{\n' + ''.join(f'  {line}\n' for line in lines) + '}\n'
    )


@pytest.fixture(scope='module')
def accelerator():
    """One instruction with a computational attribute, load_rm, and one without, gemm"""
    acc = Accelerator('PAIR')
    acc.add_data_model('d1', [128], [64], 'bf16')
    acc.add_data_model('d2', [64], [64], 'bf16')

    instr = acc.add_instruction('load_rm', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['@c.n * 128']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])
    instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In0);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")

    instr = acc.add_instruction('gemm', [], ['addr_1', 'addr_2', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_1'], ['64']], ['d1', ['@a.addr_2'], ['64']]])
    instr.set_outputs([['d2', ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %In1 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In0, %In1), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")
    return acc


def long_names(length=60):
    """Long names: two attributes of about the length given, four instructions, two data models"""
    rows = ('the_number_of_rows_' * 9)[:length].rstrip('_')
    first = ('the_first_row_' * 12)[:length].rstrip('_')
    staging = 'the_staging_buffer_of_the_vector_engine_x'
    result = 'the_result_buffer_of_the_vector_engine_xy'
    acc = Accelerator('LONG')
    acc.add_data_model(staging, [512], [64], 'bf16')
    acc.add_data_model(result, [512], [64], 'bf16')

    # Each attribute read from a shape, both in a slice
    instr = acc.add_instruction(
        'load_the_window_of_rows_into_the_staging_buffer', [rows, first], ['addr_in', 'addr_out']
    )
    instr.set_inputs([['d0', ['@a.addr_in'], [f'@c.{rows} * 128']]])
    instr.set_outputs([[staging, ['@a.addr_out'], [f'@c.{first}']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[`@c.{rows} * 128`] parameter(0);
    %a = u8[`@c.{rows}`,64,2] reshape(%In0);
    %b = bf16[`@c.{rows}`,64] bitcast_convert(%a);
    ROOT %Out0 = bf16[`@c.{first}`,64] slice(%b), slice={{[`@c.{rows} - @c.{first}`:`@c.{rows}`], [0:64]}};
}}
""")

    # One attribute, read from a second dimension
    instr = acc.add_instruction(
        'exponentiate_the_rows_of_the_staging_buffer_now', [rows], ['addr_in', 'addr_out']
    )
    instr.set_inputs([[staging, ['@a.addr_in'], ['64']]])
    instr.set_outputs([[result, ['@a.addr_out'], ['64']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = bf16[64,`@c.{rows}`] parameter(0);
    ROOT %Out0 = bf16[64,`@c.{rows}`] exponential(%In0);
}}
""")

    # No attribute, two inputs
    instr = acc.add_instruction(
        'add_the_rows_of_the_staging_buffer_to_the_result', [], ['addr_1', 'addr_2', 'addr_out']
    )
    instr.set_inputs([[staging, ['@a.addr_1'], ['64']], [result, ['@a.addr_2'], ['64']]])
    instr.set_outputs([[result, ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %In1 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] add(%In0, %In1);
}
""")

    # A dimension that sums both attributes
    instr = acc.add_instruction(
        'join_the_rows_of_the_staging_buffer_into_the_result',
        [rows, first],
        ['addr_1', 'addr_2', 'addr_out'],
    )
    instr.set_inputs(
        [[staging, ['@a.addr_1'], [f'@c.{rows}']], [staging, ['@a.addr_2'], [f'@c.{first}']]]
    )
    instr.set_outputs([[result, ['@a.addr_out'], [f'@c.{rows} + @c.{first}']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = bf16[`@c.{rows}`,64] parameter(0);
    %In1 = bf16[`@c.{first}`,64] parameter(1);
    ROOT %Out0 = bf16[`@c.{rows} + @c.{first}`,64] concatenate(%In0, %In1), dimensions={{0}};
}}
""")
    return acc


@pytest.fixture(scope='module')
def work_dir(request):
    """Generate the module's oracle and backend in a directory of its own, removed after"""
    work_dir = tempfile.mkdtemp()
    try:
        generate_oracle_and_backend(request.module.EXAMPLE_DIR, work_dir)
        # The generated backends held to the tree's formatters
        targets = os.path.join(work_dir, 'targets')
        for name in os.listdir(targets):
            complaints = unformatted_sources(os.path.join(targets, name, 'backend'))
            assert not complaints, complaints
        yield work_dir
    finally:
        shutil.rmtree(work_dir)


@pytest.fixture(scope='module')
def assembly(request, work_dir):
    """Compile the module's attention kernel and return the assembly's path"""
    return compile_hlo(work_dir, request.module.EXAMPLE_DIR)


@pytest.fixture(scope='module')
def output(request, work_dir, assembly):
    """Simulate the module's compiled kernel once and return its output"""
    return run_compiled_kernel(work_dir, assembly, request.module.EXAMPLE_DIR)
