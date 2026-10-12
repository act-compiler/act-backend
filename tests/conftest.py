"""Shared test utilities for ACT-Backend tests"""

import os
import re
import sys
import json
import shutil
import subprocess
import tempfile
import importlib
import importlib.util
import itertools
from collections import Counter

import pytest
import numpy as np
import ml_dtypes
import jax
import jax.numpy as jnp


def pytest_configure(config):
    config.addinivalue_line('markers', 'incremental: skip remaining tests after first failure')


def pytest_runtest_makereport(item, call):
    if 'incremental' in item.keywords:
        if call.excinfo is not None:
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
    # Copy the spec into work_dir
    for name in ['QKV.py', 'QKV_new.py']:
        src = os.path.join(example_dir, name)
        if os.path.exists(src):
            shutil.copy(src, work_dir)

    for name in ['QKV.py', 'QKV_new.py']:
        path = os.path.join(work_dir, name)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location('QKV_spec', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            break


def unfilled_sources(work_dir):
    """Return the generated backend sources that still hold a placeholder"""
    backend_dir = os.path.join(work_dir, 'targets', 'QKV', 'backend')
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


def compile_hlo(work_dir, example_dir):
    """Run the backend binary on the HLO file to produce a compiled kernel"""
    backend_bin = os.path.join(work_dir, 'backends', 'QKV')
    hlo_path = os.path.join(example_dir, 'kernels', 'attention.hlo')
    asm_dir = os.path.join(work_dir, 'asm')
    os.makedirs(asm_dir, exist_ok=True)
    output_path = os.path.join(asm_dir, 'compiled_qkv.py')

    subprocess.run(
        [backend_bin, '--input', hlo_path, '--output', output_path], check=True, cwd=work_dir
    )
    return output_path


def instruction_calls(assembly_path):
    """Count the calls to each instruction in a compiled kernel"""
    with open(assembly_path) as f:
        return Counter(re.findall(r'api\.(\w+)\(', f.read()))


def import_oracle(work_dir):
    """Import the oracle in work_dir as a package of its own and return (decorator, api)"""
    oracle_dir = os.path.join(work_dir, 'targets', 'QKV', 'oracle')
    name = f'oracle_{next(oracle_ids)}'
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(oracle_dir, '__init__.py'), submodule_search_locations=[oracle_dir]
    )
    sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[name])
    decorator = importlib.import_module(f'{name}.decorator')
    decorator.set_simulation_backend(PLATFORM)
    return decorator, importlib.import_module(f'{name}.api')


def run_compiled_kernel(work_dir, assembly_path, data_dir):
    """Import oracle + compiled kernel, run simulation, return its output"""
    oracle_decorator, oracle_api = import_oracle(work_dir)

    # The compiled kernel, loaded from its own path
    spec = importlib.util.spec_from_file_location('compiled_qkv', assembly_path)
    kernel_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_module)

    qkv_kernel = kernel_module.qkv(oracle_decorator.kernel, oracle_api)
    qkv_kernel('fsim-compile')()

    Q = load_bf16_matrix(os.path.join(data_dir, 'Q.dat'), (64, 64))
    K = load_bf16_matrix(os.path.join(data_dir, 'K.dat'), (64, 64))
    V = load_bf16_matrix(os.path.join(data_dir, 'V.dat'), (64, 64))

    outputs, _ = qkv_kernel('fsim')(Q, K, V)
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


@pytest.fixture(scope='module')
def work_dir(request):
    """Generate the module's oracle and backend in a directory of its own, removed after"""
    work_dir = tempfile.mkdtemp()
    try:
        generate_oracle_and_backend(request.module.QKV_DIR, work_dir)
        yield work_dir
    finally:
        shutil.rmtree(work_dir)


@pytest.fixture(scope='module')
def assembly(request, work_dir):
    """Compile the module's attention kernel and return the assembly's path"""
    return compile_hlo(work_dir, request.module.QKV_DIR)


@pytest.fixture(scope='module')
def output(request, work_dir, assembly):
    """Simulate the module's compiled kernel once and return its output"""
    return run_compiled_kernel(work_dir, assembly, request.module.DATA_DIR)
