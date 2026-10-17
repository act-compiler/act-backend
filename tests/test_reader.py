"""Tests of the kernel reader: what it reads of a kernel and why it refuses one."""

import os
import subprocess

import pytest
from conftest import compiled, instruction_calls, kernel


def region(name, operation):
    """A reduce region over two bf16 scalars that applies one binary operation"""
    return (
        f'%{name} (x: bf16[], y: bf16[]) -> bf16[] {{\n  %x = bf16[] parameter(0)\n'
        f'  %y = bf16[] parameter(1)\n  ROOT %v = bf16[] {operation}(%x, %y)\n}}\n'
    )


A = '%a = bf16[64,64] parameter(0)'
# A kernel of one bf16[64,60] input and output
TILED_HEADER = 'HloModule tiled, entry_computation_layout={(bf16[64,60]{1,0})->bf16[64,60]{1,0}}'
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
            kernel(
                A,
                ZERO,
                '%r = bf16[64] reduce(%a, %z), dimensions={y}, to_apply=%sum',
                'ROOT %c = bf16[64,64] broadcast(%r), dimensions={0}',
                regions=region('sum', 'add'),
            ),
            "%r gives 'y' in dimensions, which is not a dimension",
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
        (
            kernel(
                '%a = bf16[64,60]{1,0:T(32,16)} parameter(0)',
                'ROOT %c = bf16[64,60] add(%a, %a)',
                header=TILED_HEADER,
            ),
            'the tile (32,16) does not divide the shape [64,60]',
        ),
        (
            kernel(
                '%a = bf16[64,64]{1,0:T(32,16)(2,1)} parameter(0)',
                'ROOT %c = bf16[64,64] add(%a, %a)',
            ),
            (
                'the layout {1,0:T(32,16)(2,1)} of the shape [64,64] gives a second tile, '
                'which the backend does not support'
            ),
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
        'reduce_not_a_number',
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
        'undivided_tile',
        'second_tile',
        'empty',
    ],
)
def test_unreadable_kernel_ends_in_no_assembly(backend, tmp_path, text, reason):
    (tmp_path / 'kernel.py').write_text('# an earlier compile\n')
    run, output = compiled(backend, tmp_path, text)
    assert run.returncode == 1
    assert run.stderr == f'Error: no assembly found: {reason}\n'
    assert not output.exists()


# A library caller printing every node of a kernel's e-graph
NODES = """use backend::isel;

fn main() {
  let kernel = std::env::args().nth(1).unwrap();
  let (egraph, ..) = isel::initializer::parse_hlo_module_to_egraph(&kernel).unwrap();
  for class in egraph.classes() {
    for node in &class.nodes {
      println!("{}", node);
    }
  }
}
"""

# Each node with its class's type and shape
TYPED_NODES = """use backend::isel;

fn main() {
  let kernel = std::env::args().nth(1).unwrap();
  let (egraph, ..) = isel::initializer::parse_hlo_module_to_egraph(&kernel).unwrap();
  for class in egraph.classes() {
    for node in &class.nodes {
      println!("{} {}{:?}", node, class.data.dtype, class.data.shape);
    }
  }
}
"""

# A concatenate and a reduce, each with attributes and metadata
ATTRIBUTES = kernel(
    '%a = bf16[32,64] parameter(0)',
    '%b = bf16[32,64] parameter(1)',
    '%c = bf16[64,64] concatenate(%a, %b), dimensions={0}, metadata={op_name="cat, of two" source_line=31}',
    '%z = bf16[] constant(0)',
    'ROOT %r = bf16[] reduce(%c, %z), dimensions={0,1}, to_apply=%sum',
    header='HloModule attributes, entry_computation_layout={(bf16[32,64]{1,0}, bf16[32,64]{1,0})->bf16[]}',
    regions=region('sum', 'add'),
)


def read_nodes(work_dir, tmp_path, text, program=NODES):
    """Read a kernel's text through the crate's library; return what the program prints of its e-graph"""
    crate = os.path.join(work_dir, 'targets', 'ENDINGS', 'backend')
    os.makedirs(os.path.join(crate, 'examples'), exist_ok=True)
    with open(os.path.join(crate, 'examples', 'nodes.rs'), 'w') as f:
        f.write(program)
    path = tmp_path / 'kernel.hlo'
    path.write_text(text)
    command = ['cargo', 'run', '--release', '--locked', '--example', 'nodes', str(path)]
    run = subprocess.run(command, cwd=crate, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    return run.stdout.splitlines()


def test_kernel_attributes_are_read_by_name(endings_dir, tmp_path):
    nodes = read_nodes(endings_dir, tmp_path, ATTRIBUTES)
    assert "concat[axis='0']" in nodes
    assert "reduce[dims='0,1']" in nodes


def test_dot_dimensions_are_read_by_name(endings_dir, tmp_path):
    line = 'ROOT %c = bf16[64,64] dot(%a, %a), rhs_contracting_dims={1}, lhs_contracting_dims={0}'
    # Each input's batch list, left out, then its contracting list
    assert "dot[dims=';0;;1']" in read_nodes(endings_dir, tmp_path, kernel(A, line))
    line = 'ROOT %c = bf16[] dot(%a, %a), lhs_contracting_dims={1,0}, rhs_contracting_dims={0,1}'
    # The contracting pairs as written
    assert "dot[dims=';1,0;;0,1']" in read_nodes(endings_dir, tmp_path, kernel(A, line))
    line = (
        'ROOT %c = bf16[64] dot(%a, %a), rhs_batch_dims={1}, lhs_batch_dims={0}, '
        'lhs_contracting_dims={1}, rhs_contracting_dims={0}'
    )
    # Each input's batch list, then its contracting list
    assert "dot[dims='0;1;1;0']" in read_nodes(endings_dir, tmp_path, kernel(A, line))


@pytest.mark.parametrize(
    'line, reason',
    [
        (
            'ROOT %c = bf16[64,64] dot(%a, %a), lhs_contracting_dims={1}',
            '%c gives dot no rhs_contracting_dims attribute',
        ),
        (
            'ROOT %c = bf16[64,64] dot(%a, %a), lhs_contracting_dims={y}, rhs_contracting_dims={0}',
            "%c gives 'y' in lhs_contracting_dims, which is not a dimension",
        ),
        (
            'ROOT %c = bf16[64,64] transpose(%a)',
            '%c gives transpose no dimensions attribute',
        ),
        (
            'ROOT %c = bf16[64,64] transpose(%a), dimensions={1,y}',
            "%c gives 'y' in dimensions, which is not a dimension",
        ),
        (
            'ROOT %c = bf16[32,64] slice(%a)',
            '%c gives slice no slice attribute',
        ),
        (
            'ROOT %c = bf16[64,64] broadcast(%a)',
            '%c gives broadcast no dimensions attribute',
        ),
        (
            'ROOT %c = bf16[64,64] broadcast(%a), dimensions={0,y}',
            "%c gives 'y' in dimensions, which is not a dimension",
        ),
        (
            'ROOT %c = bf16[192,64] concatenate(%a, %a, %a), dimensions={0}',
            '%c concatenates 3 values, where a concatenate takes 2',
        ),
        (
            'ROOT %c = bf16[64,64] reverse(%a), dimensions={0}',
            '%c applies reverse, an operation the backend does not support',
        ),
    ],
    ids=[
        'no_contracting',
        'contracting_not_a_number',
        'no_dimensions',
        'transpose_not_a_number',
        'no_slice',
        'no_map',
        'map_not_a_number',
        'three_values',
        'unread_operation',
    ],
)
def test_attribute_the_reader_does_not_take_ends_in_no_assembly(backend, tmp_path, line, reason):
    run, _ = compiled(backend, tmp_path, kernel(A, line))
    assert run.returncode == 1
    assert run.stderr == f'Error: no assembly found: {reason}\n'


@pytest.mark.parametrize(
    'text',
    [
        kernel(A, 'ROOT %c = bf16[64,64]{0,1} add(%a, %a)'),
        kernel(
            A,
            'ROOT %c = bf16[64,64] add(%a, %a)',
            header='HloModule bad, entry_computation_layout={(bf16[64,64]{0,1})->bf16[64,64]{1,0}}',
        ),
    ],
    ids=['line', 'header'],
)
def test_layout_other_than_row_major_ends_in_no_assembly(backend, tmp_path, text):
    run, _ = compiled(backend, tmp_path, text)
    assert run.returncode == 1
    assert run.stderr == (
        'Error: no assembly found: the layout {0,1} of the shape [64,64] is not row-major, '
        'the only layout the backend supports\n'
    )


def test_layout_that_gives_no_order_is_read(endings_dir, tmp_path):
    # A tile alone after the shape, which gives no order
    text = kernel('%a = bf16[64,64]{:T(32,64)} parameter(0)', 'ROOT %c = bf16[64,64] add(%a, %a)')
    assert "Var['%a']" in read_nodes(endings_dir, tmp_path, text)


def test_tiled_layout_is_read_through_its_permuted_tiles(endings_dir, tmp_path):
    # An input and a root each in tiles of 32 by 16
    text = kernel(
        '%a = bf16[64,64]{1,0:T(32,16)} parameter(0)',
        'ROOT %c = bf16[64,64]{1,0:T(32,16)} add(%a, %a)',
    )
    nodes = read_nodes(endings_dir, tmp_path, text, TYPED_NODES)
    # The input's tiles back into its shape
    assert 'bitcvt bf16[2, 4, 32, 16]' in nodes
    assert "transpose[perm='0,2,1,3'] bf16[2, 32, 4, 16]" in nodes
    assert "reshape[shape='64,64'] bf16[64, 64]" in nodes
    # The root's shape into its tiles
    assert "reshape[shape='2,32,4,16'] bf16[2, 32, 4, 16]" in nodes
    assert "transpose[perm='0,2,1,3'] bf16[2, 4, 32, 16]" in nodes


# A kernel of one bf16 scalar input and output
SCALAR = kernel(
    '%s = bf16[] parameter(0)',
    'ROOT %t = bf16[] add(%s, %s)',
    header='HloModule scalar, entry_computation_layout={(bf16[])->bf16[]}',
)


def test_rank_0_input_is_read_as_one_element(endings_dir, tmp_path):
    nodes = read_nodes(endings_dir, tmp_path, SCALAR)
    # The input viewed from its two bytes as one element
    assert "reshape[shape='1,2']" in nodes
    assert not [node for node in nodes if '[]' in node]


@pytest.mark.parametrize(
    'line, reason',
    [
        (
            'ROOT %s = bf16[32,64] slice(%a), slice={[0:64:2], [0:64:1]}',
            '%s slices with the stride 2, where the backend supports only a stride of one',
        ),
        (
            'ROOT %s = bf16[32,64] slice(%a), slice={[0:32:1], [7]}',
            "%s slices with '[7]', which is not a [start:limit:stride] range",
        ),
    ],
    ids=['stride', 'no_limit'],
)
def test_slice_the_reader_does_not_take_ends_in_no_assembly(backend, tmp_path, line, reason):
    run, _ = compiled(backend, tmp_path, kernel(A, line))
    assert run.returncode == 1
    assert run.stderr == f'Error: no assembly found: {reason}\n'


@pytest.mark.parametrize(
    'line, reason',
    [
        (
            'ROOT %s = bf16[32,64] slice(%a), slice={[x:32:1], [0:64:1]}',
            "%s slices with '[x:32:1]', which is not a [start:limit:stride] range",
        ),
        (
            'ROOT %s = bf16[96,64] slice(%a), slice={[0:96:1], [0:64:1]}',
            '%s slices [0:96:1] of a dimension of 64, where 0 <= start < limit <= 64',
        ),
        (
            'ROOT %s = bf16[32] slice(%a), slice={[0:32:1]}',
            '%s slices a value of rank 2 with bounds for rank 1',
        ),
        (
            'ROOT %s = bf16[16,64] slice(%a), slice={[0:32:1], [0:64:1]}',
            '%s gives the shape [16,64] to a slice of the shape [32,64]',
        ),
        (
            'ROOT %c = bf16[64,128] concatenate(%a, %a), dimensions={2}',
            '%c concatenates values of rank 2 along dimension 2',
        ),
        (
            'ROOT %c = bf16[96,64] concatenate(%a, %a), dimensions={0}',
            '%c concatenates [64,64] and [64,64] along dimension 0, which does not give its shape [96,64]',
        ),
    ],
    ids=['bound_word', 'past_source', 'rank', 'result', 'dimension', 'joined'],
)
def test_slice_or_concatenate_of_shapes_that_disagree_ends_in_no_assembly(
    backend, tmp_path, line, reason
):
    run, _ = compiled(backend, tmp_path, kernel(A, line))
    assert run.returncode == 1
    assert run.stderr == f'Error: no assembly found: {reason}\n'


# A sum of two 32-row matrices, then that sum doubled
SUM_32 = (
    '%a = bf16[32,64] parameter(0)',
    '%b = bf16[32,64] parameter(1)',
    '%c = bf16[32,64] add(%a, %b)',
    '%d = bf16[32,64] add(%c, %c)',
)
HEADER_32 = 'HloModule sum, entry_computation_layout={(bf16[32,64]{1,0}, bf16[32,64]{1,0})->bf16[32,64]{1,0}}'


def test_kernel_is_compiled_for_the_value_its_root_marks(backend, tmp_path):
    a, b, c, d = SUM_32
    run, output = compiled(backend, tmp_path, kernel(a, b, f'ROOT {c}', d, header=HEADER_32))
    assert run.returncode == 0, run.stderr
    assert instruction_calls(output)['vadd_32'] == 1


@pytest.mark.parametrize(
    'roots, reason',
    [
        ((2, 3), "the kernel's ENTRY marks 2 lines ROOT, where the reader takes one"),
    ],
    ids=['two'],
)
def test_kernel_without_one_root_ends_in_no_assembly(backend, tmp_path, roots, reason):
    lines = [f'ROOT {line}' if k in roots else line for k, line in enumerate(SUM_32)]
    run, _ = compiled(backend, tmp_path, kernel(*lines, header=HEADER_32))
    assert run.returncode == 1
    assert run.stderr == f'Error: no assembly found: {reason}\n'


@pytest.mark.parametrize(
    'line, reason',
    [
        (
            '%e bf16[32,64] add(%c, %c)',
            (
                "the reader cannot split the kernel's line '%e bf16[32,64] add(%c, %c)' into a name, "
                'a type and an operation'
            ),
        ),
        (
            '%e = bf16[32,64] add(%c, %c',
            (
                "the reader cannot split the kernel's line '%e = bf16[32,64] add(%c, %c' into a name, "
                'a type and an operation'
            ),
        ),
        (
            '%e = bf16[32,64] parameter(x)',
            '%e reads parameter x, which is not a parameter number',
        ),
    ],
    ids=['no_equals', 'no_parenthesis', 'parameter_word'],
)
def test_kernel_line_the_reader_cannot_split_ends_in_no_assembly(backend, tmp_path, line, reason):
    a, b, c, _ = SUM_32
    run, _ = compiled(backend, tmp_path, kernel(a, b, f'ROOT {c}', line, header=HEADER_32))
    assert run.returncode == 1
    assert run.stderr == f'Error: no assembly found: {reason}\n'
