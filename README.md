# ACT Generator: Compiler Backend

This repository contains the source code and documentation for `act-backend`, a compiler backend generator in the [ACT Ecosystem](https://github.com/act-compiler/act) that automatically generates compiler backends from ISA specifications written in [TAIDL](https://github.com/act-compiler/taidl).

For more details about the ACT Ecosystem, refer to the top-level repository: [act-compiler/act](https://github.com/act-compiler/act).

## TAIDL: Tensor Accelerator ISA Definition Language

TAIDL is a domain-specific language designed to define instruction set architectures (ISAs) for tensor accelerators. It is published at [MICRO 2025](https://doi.org/10.1145/3725843.3756075).
TAIDL not only standardizes the way tensor accelerator ISAs are specified but also enables automated generation of tools such as test oracles (functional simulators) and compiler backends, significantly reducing the effort required to develop and maintain these components.

## Compiler Backend Generation

`act-backend` is one of the tool generators in the [ACT Ecosystem](https://github.com/act-compiler/act) that consumes TAIDL specifications and emits out a compiler backend instantaneously.
The generated compiler backend translates high-level tensor kernels in [XLA-HLO IR](https://openxla.org/xla) into low-level machine code for the target tensor accelerator defined in TAIDL.
Therefore, it can be directly integrated into existing ML compilers like [XLA](https://www.tensorflow.org/xla), enabling seamless support for tensor accelerators in popular machine learning frameworks like JAX and PyTorch with minimal engineering effort.

The compiler backend has provable guarantees of soundness (i.e., it will never generate invalid machine code) and completeness (i.e., it can generate machine code for all valid programs).
Furthermore, the generated compiler backend supports integration with cost models to enable performance-aware code generation.
The generated compiler backend is agnostic to the choice of the cost model and can be easily adapted to work with different cost models like cycle-accurate simulators, analytical models, or ML-based models.

Details on automatically generating compiler backends from TAIDL definitions is present in our [arXiv release](https://arxiv.org/abs/2510.09932).

## Installation

ACT-Backend needs Python 3.14 or later.

```bash
pip install git+https://github.com/act-compiler/act-backend.git@v1.5
```

Generating a backend builds it, which needs `cargo` with access to crates.io, CMake, `make`, a C++17 compiler, [OR-Tools](https://developers.google.com/optimization) and the [nlohmann/json](https://github.com/nlohmann/json) headers on the machine.
OR-Tools is read from the directory `ORTOOLS_DIR` names, by default `/opt/ortools`, and the headers from the one `NLOHMANN_JSON_DIR` names, by default `/opt/nlohmann_json`.
The images `devanshdvj/act-env:py3.14-jax0.10.2-amd64` and `devanshdvj/act-env:py3.14-jax0.10.2-arm64` hold all of them.

## Usage

```python
from act_backend import generate_backend

generate_backend(acc)  # acc: a TAIDL Accelerator
```

`generate_backend(acc)` writes the backend's crate into `targets/<name>/backend/` under the accelerator's `base_dir`, by default the directory of the file that creates the accelerator, builds it and copies the compiler to `backends/<name>`.
A build that fails raises `RuntimeError` with the build's own output, and a generation that fails leaves no `backends/<name>` from an earlier one.
ACT-Backend reads TAIDL v2.3.

```bash
./backends/<name> --input kernel.hlo --output kernel.py
```

The backend compiles the kernel in an XLA-HLO file into assembly: a Python file of calls to the accelerator's instructions, which an oracle that [TAIDL-TO](https://github.com/act-compiler/act-oracle) generates for the same accelerator simulates.
The backend is one file that holds its rewrite rules, and its run path names the OR-Tools directory it was built with.

## Cost function

Among the candidates it finds for a kernel, a backend emits the one of lowest cost.
The cost comes from one C function:

```c
int act_cost(const char *asm_path, double *cost);
```

It reads the candidate's assembly at `asm_path`, writes its cost and returns 0; any other return ends the compile with an error.
By default it counts the candidate's instructions.
`generate_backend(acc, cost='model.o')` links the library that defines it into the backend in the default's place: an object file, a static library or a shared library, which its author builds.

```bash
cc -c -fPIC model.c -o model.o
```

The library's author is responsible for it being complete: beyond the C and C++ runtime libraries, the link adds nothing to it.
An object file or a static library is carried inside the backend, while a shared library must stay where it is.
The author returns a large finite cost for a candidate the model cannot price, and closes what the model opened in an `atexit` handler.

## Examples

- `examples/QKV/`: `QKV.py`, an accelerator with two on-chip buffers and seven instructions, and its `attention` kernel under `kernels/`
- `examples/QKV_new/`: `QKV_new.py`, a variant with a third on-chip buffer and ten instructions, and the same kernel

`pip install -r tests/requirements.txt` installs TAIDL-TO, which each spec needs for its oracle.
Running a spec, such as `cd examples/QKV && python QKV.py`, generates its oracle and its backend there, and `./backends/QKV --input kernels/attention.hlo --output attention.py` then compiles the kernel.
`tests/conftest.py` does both in a temporary directory and simulates the compiled kernel on the matrices under `data/`.

## Testing

```bash
./tests/pytest.sh             # every test
./tests/pytest.sh -k qkv_new  # its arguments go on to pytest
./tests/regenerate.sh         # rewrite the goldens from the compiled kernels
./tests/lint.sh               # lint the backend each example generates
```

`tests/pytest.sh` runs the tests in `devanshdvj/act-env:py3.14-jax0.10.2`, the image `docker/act.env` names, in its tag for the host's architecture, amd64 or arm64, and in its CUDA 13 tag where `nvidia-smi` succeeds and docker lists the nvidia runtime; the CI pulls its tag for the runner's architecture.
It installs the package and what `tests/requirements.txt` names, pytest and TAIDL-TO v2.5, whose generated oracle simulates each compiled kernel on the CPU.
A compiled attention kernel equals its golden bit for bit and is held to its NumPy reference within two bf16 ulps of the largest magnitude in V; `tests/regenerate.sh` rewrites the goldens and records the oracle, jax, platform and machine they come from.
`tests/lint.sh` generates each example's backend, runs `cargo clippy` over its crate with warnings denied and builds its allocator with `-Wall -Wextra -Werror`; the CI runs it after the tests, and generating a backend never does.
