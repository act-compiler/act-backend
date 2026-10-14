"""Lint the backend an example generates: its crate with clippy, its allocator with the compiler

Usage: python tests/lint.py <example_dir>...; tests/lint.sh runs it for every example in the
image the tests run in, and for the first one once more with a cost library.
"""

import functools
import os
import subprocess
import sys
import tempfile
from unittest import mock

from conftest import generate_oracle_and_backend

import act_backend

# The warnings the allocator's build is held to
CXX_FLAGS = '-Wall -Wextra -Werror'
# The cost library a crate is generated with
COST_MODEL = os.path.join(os.path.dirname(__file__), 'cost', 'model.c')


def lint_crate(backend_dir):
    """Lint a generated crate, its generated sources and build script among them"""
    subprocess.run(
        ['cargo', 'clippy', '--release', '--', '-D', 'warnings'], cwd=backend_dir, check=True
    )


if __name__ == '__main__':
    for example in sys.argv[1:]:
        # The generated trees go with their directory
        with tempfile.TemporaryDirectory() as work_dir:
            generate_oracle_and_backend(os.path.abspath(example), work_dir)
            # An example's accelerator is named as its directory
            name = os.path.basename(os.path.abspath(example))
            backend_dir = os.path.join(work_dir, 'targets', name, 'backend')
            lint_crate(backend_dir)

            # The allocator, built apart from the crate's own tree
            allocator_dir = os.path.join(backend_dir, 'cpp', 'malloc')
            build_dir = os.path.join(work_dir, 'lint')
            subprocess.run(
                ['cmake', '-S', allocator_dir, '-B', build_dir, f'-DCMAKE_CXX_FLAGS={CXX_FLAGS}'],
                check=True,
            )
            subprocess.run(['cmake', '--build', build_dir], check=True)
        print(f'{example}: no lint warning')

    # A shared cost library: each link line of build.rs
    example = sys.argv[1]
    with tempfile.TemporaryDirectory() as work_dir:
        library = os.path.join(work_dir, 'libmodel.so')
        subprocess.run(['cc', '-shared', '-fPIC', COST_MODEL, '-o', library], check=True)
        # The spec calls generate_backend(mini_trn): name the library in that call
        with_cost = functools.partial(act_backend.generate_backend, cost=library)
        with mock.patch.object(act_backend, 'generate_backend', with_cost):
            generate_oracle_and_backend(os.path.abspath(example), work_dir)
        name = os.path.basename(os.path.abspath(example))
        lint_crate(os.path.join(work_dir, 'targets', name, 'backend'))
    print(f'{example} with a cost library: no lint warning')
