"""Lint the backend an example generates: its crate with clippy, its allocator with the compiler

Usage: python tests/lint.py <example_dir>...; tests/lint.sh runs it for every example in the
image the tests run in.
"""

import os
import sys
import subprocess
import tempfile

from conftest import generate_oracle_and_backend

# The warnings the allocator's build is held to
CXX_FLAGS = '-Wall -Wextra -Werror'

if __name__ == '__main__':
    for example in sys.argv[1:]:
        # The generated trees go with their directory
        with tempfile.TemporaryDirectory() as work_dir:
            generate_oracle_and_backend(os.path.abspath(example), work_dir)
            backend_dir = os.path.join(work_dir, 'targets', 'QKV', 'backend')

            # The crate, its generated sources among them
            subprocess.run(
                ['cargo', 'clippy', '--release', '--', '-D', 'warnings'],
                cwd=backend_dir,
                check=True,
            )

            # The allocator, in a build tree apart from the crate's own
            allocator_dir = os.path.join(backend_dir, 'cpp', 'malloc')
            build_dir = os.path.join(work_dir, 'lint')
            subprocess.run(
                ['cmake', '-S', allocator_dir, '-B', build_dir, f'-DCMAKE_CXX_FLAGS={CXX_FLAGS}'],
                check=True,
            )
            subprocess.run(['cmake', '--build', build_dir], check=True)
        print(f'{example}: no lint warning')
