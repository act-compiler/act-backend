"""Tests of the kernel reader: what it reads of a kernel and why it refuses one."""

import pytest
from conftest import compiled

# A one-input, one-output bf16 kernel around the given lines
HEADER = 'HloModule bad, entry_computation_layout={(bf16[64,64]{1,0})->bf16[64,64]{1,0}}'


def kernel(*lines, header=HEADER, regions=''):
    """The text of a kernel: its header, its regions and an ENTRY of the lines given"""
    return (
        f'{header}\n\n{regions}\nENTRY main {{\n' + ''.join(f'  {line}\n' for line in lines) + '}\n'
    )


def region(name, operation):
    """A reduce region over two bf16 scalars that applies one binary operation"""
    return (
        f'%{name} (x: bf16[], y: bf16[]) -> bf16[] {{\n  %x = bf16[] parameter(0)\n'
        f'  %y = bf16[] parameter(1)\n  ROOT %v = bf16[] {operation}(%x, %y)\n}}\n'
    )


A = '%a = bf16[64,64] parameter(0)'
ZERO = '%z = bf16[] constant(0)'


@pytest.mark.parametrize(
    'text, reason',
    [
        (
            kernel(A, 'ROOT %c = bf16[64,64] multiply(%a, %a)'),
            '%c applies multiply, an operation the backend does not support',
        ),
        (
            kernel(A, 'ROOT %c = bf16[64,64] and(%a, %a)'),
            '%c applies and, an operation the backend does not support',
        ),
        (
            kernel(A, 'ROOT %c = bf16[64,64] add(%a, %b)'),
            'the kernel reads %b, which no earlier line of it defines',
        ),
        (
            kernel(
                A,
                ZERO,
                '%r = bf16[64] reduce(%a, %z), dimensions={1}, to_apply=%none',
                'ROOT %c = bf16[64,64] broadcast(%r), dimensions={0}',
                regions=region('sum', 'add'),
            ),
            'a reduce applies the region %none, which the kernel does not define',
        ),
        (
            kernel(
                A,
                ZERO,
                '%r = bf16[64] reduce(%a, %z), dimensions={1}, to_apply=%max',
                'ROOT %c = bf16[64,64] broadcast(%r), dimensions={0}',
                regions=region('max', 'maximum'),
            ),
            '%v applies maximum, an operation the backend does not support',
        ),
        (
            kernel(
                A,
                ZERO,
                '%r = bf16[64] reduce(%a, %z), dimensions={1}, to_apply=%quotient',
                'ROOT %c = bf16[64,64] broadcast(%r), dimensions={0}',
                regions=region('quotient', 'divide'),
            ),
            'the region %quotient computes something other than a sum, the only reduce the backend supports',
        ),
        (
            kernel(
                A,
                ZERO,
                '%r = bf16[64] reduce(%a, %z, %z), dimensions={1}, to_apply=%sum',
                'ROOT %c = bf16[64,64] broadcast(%r), dimensions={0}',
                regions=region('sum', 'add'),
            ),
            '%r reduces 3 operands, where a reduce takes 2, a value and its initial value',
        ),
        (
            kernel(A, 'ROOT %c = f8e4m3fn[64,64] convert(%a)'),
            "the element type 'f8e4m3fn' is not one the backend supports",
        ),
        (
            kernel(A, 'ROOT %c = bf16[64,n] add(%a, %a)'),
            "the shape [64,n] holds 'n', which is not a dimension",
        ),
        (
            kernel(A, 'ROOT %c = bf16 add(%a, %a)'),
            "the type 'bf16' is not an element type with a shape",
        ),
        (
            kernel(A, 'ROOT %c = bf16[64,64] add(%a, %a)', header='HloModule bad'),
            "the kernel's first line 'HloModule bad' is not an HloModule line with an entry_computation_layout",
        ),
        (
            kernel(A, '%b = bf16[64,64] parameter(3)', 'ROOT %c = bf16[64,64] add(%a, %b)'),
            "%b reads parameter 3, which the kernel's header does not give",
        ),
        (
            kernel(A, '%b = bf16[64,64] parameter(0)', 'ROOT %c = bf16[64,64] add(%a, %b)'),
            '%b reads parameter 0, which an earlier line reads',
        ),
        (kernel(), "the kernel's ENTRY computes no value"),
        (
            kernel('ROOT %a = bf16[64,64] parameter(0)'),
            "the kernel's root %a is a parameter, which no instruction computes",
        ),
        (
            kernel(A, 'ROOT %c = bf16[64,64] concatenate(%a), dimensions={0}'),
            '%c concatenates 1 value, where a concatenate takes 2',
        ),
        (
            kernel('%a = bf16[64,64]{1,0:T(8)} parameter(0)', 'ROOT %c = bf16[64,64] add(%a, %a)'),
            'the tile (8) of the shape [64,64] does not give one size per dimension',
        ),
        ('', 'the kernel is empty'),
    ],
    ids=[
        'multiply',
        'and',
        'undefined_value',
        'no_region',
        'maximum_region',
        'divide_region',
        'three_operands',
        'element_type',
        'dimension',
        'type',
        'header',
        'parameter_index',
        'parameter_twice',
        'empty_entry',
        'parameter_root',
        'one_value',
        'tile',
        'empty',
    ],
)
def test_unreadable_kernel_ends_in_no_assembly(backend, tmp_path, text, reason):
    (tmp_path / 'kernel.py').write_text('# an earlier compile\n')
    run, output = compiled(backend, tmp_path, text)
    assert run.returncode == 1
    assert run.stderr == f'Error: no assembly found: {reason}\n'
    assert not output.exists()
