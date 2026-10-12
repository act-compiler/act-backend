"""Regenerate an example's golden from its compiled kernel

Usage: python tests/regenerate.py <example_dir>, one example per process; tests/regenerate.sh
runs it for every example in the image the tests run in.
"""

import os
import sys
import json
import platform
import tempfile

import jax
import jaxlib
import numpy as np
import taidl_to

from conftest import (
    PLATFORM,
    generate_oracle_and_backend,
    compile_hlo,
    run_compiled_kernel,
    attention_reference,
)

if __name__ == '__main__':
    example_dir = os.path.abspath(sys.argv[1])
    data_dir = os.path.join(example_dir, 'data')

    # The generated trees go with their directory
    with tempfile.TemporaryDirectory() as work_dir:
        generate_oracle_and_backend(example_dir, work_dir)
        assembly = compile_hlo(work_dir, example_dir)
        output = np.asarray(run_compiled_kernel(work_dir, assembly, data_dir))

    # Never write a golden the reference rejects
    reference, tol = attention_reference(data_dir)
    max_diff = np.max(np.abs(output.astype(np.float64) - reference))
    assert max_diff <= tol, f'attention.dat is {max_diff} from the reference, beyond {tol}'
    output.tofile(os.path.join(data_dir, 'attention.dat'))

    # Record where the golden came from
    record = {
        'kernel': 'attention.hlo',
        'oracle': taidl_to.__version__,
        'jax': jax.__version__,
        'jaxlib': jaxlib.__version__,
        'platform': PLATFORM.lower(),
        'machine': platform.machine(),
    }
    with open(os.path.join(data_dir, 'goldens.json'), 'w') as f:
        json.dump({'attention.dat': record}, f, indent=2)
        f.write('\n')
    print(f'Wrote attention.dat, {max_diff} from the reference')
