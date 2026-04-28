"""Shared test utilities for ACT-Backend tests"""

import os
import sys
import shutil
import subprocess
import tempfile
import importlib
import importlib.util

import pytest
import numpy as np
import jax
import jax.numpy as jnp


def pytest_configure(config):
    config.addinivalue_line("markers", "incremental: skip remaining tests after first failure")


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.skip(f"previous test failed ({previousfailed.name})")


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def load_bf16_matrix(path, shape):
    np_uint8 = np.fromfile(path, dtype=np.uint8)
    np_uint8 = np_uint8.reshape(shape[0], shape[1], 2)
    j_uint8 = jnp.array(np_uint8, dtype=jnp.uint8)
    return jax.lax.bitcast_convert_type(j_uint8, jnp.bfloat16)


def generate_oracle_and_backend(example_dir, work_dir):
    """Run the ISA spec to generate oracle + backend into work_dir"""
    # Copy spec and HLO into work_dir
    for name in ["QKV.py", "QKV_new.py"]:
        src = os.path.join(example_dir, name)
        if os.path.exists(src):
            shutil.copy(src, work_dir)

    os.chdir(work_dir)
    for name in ["QKV.py", "QKV_new.py"]:
        path = os.path.join(work_dir, name)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("QKV_spec", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            break


def compile_hlo(work_dir, example_dir):
    """Run the backend binary on the HLO file to produce a compiled kernel"""
    backend_bin = os.path.join(work_dir, "backends", "QKV")
    hlo_path = os.path.join(example_dir, "kernels", "attention.hlo")
    asm_dir = os.path.join(work_dir, "asm")
    os.makedirs(asm_dir, exist_ok=True)
    output_path = os.path.join(asm_dir, "compiled_qkv.py")

    subprocess.run(
        [backend_bin, "--input", hlo_path, "--output", output_path],
        check=True, cwd=work_dir
    )
    return output_path


def run_compiled_kernel(work_dir, data_dir):
    """Import oracle + compiled kernel, run simulation, return max diff against golden"""
    oracle_dir = os.path.join(work_dir, "targets", "QKV")
    sys.path.insert(0, oracle_dir)

    oracle_decorator = importlib.import_module("oracle.decorator")
    oracle_api = importlib.import_module("oracle.api")
    oracle_decorator.set_simulation_backend('CPU')

    asm_dir = os.path.join(work_dir, "asm")
    sys.path.insert(0, asm_dir)
    kernel_module = importlib.import_module("compiled_qkv")

    qkv_kernel = kernel_module.qkv(oracle_decorator.kernel, oracle_api)
    qkv_kernel('fsim-compile')()

    Q = load_bf16_matrix(os.path.join(data_dir, "Q.dat"), (64, 64))
    K = load_bf16_matrix(os.path.join(data_dir, "K.dat"), (64, 64))
    V = load_bf16_matrix(os.path.join(data_dir, "V.dat"), (64, 64))

    outputs, _ = qkv_kernel('fsim')(Q, K, V)
    golden = load_bf16_matrix(os.path.join(data_dir, "attention.dat"), (64, 64))

    return float(jnp.max(jnp.abs(outputs[0] - golden)))
