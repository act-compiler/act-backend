"""Unit tests of the generator functions, without a build."""

import re

import pytest
from taidl import Accelerator

from act_backend import ir_egraph_rs_generator
from act_backend.generator import get_instruction_metadata
from act_backend.instruction_info import extract_instruction_info
from act_backend.isel_applier_rs_generator import generate_function_mappings
from act_backend.isel_ir2isa_rewrites_rs_generator import generate_ir2isa_rust_functions
from act_backend.isel_ir2isa_rewrites_txt_generator import generate_rewrite_rule
from act_backend.malloc_act_malloc_cc_generator import generate_buffer_names
from act_backend.malloc_globals_cc_generator import generate_globals_models
from act_backend.malloc_instructions_h_generator import generate_instruction_class
from act_backend.malloc_parser_cc_generator import generate_parser_instructions
from act_backend.template_loader import fill_placeholders, get_backend_template_loader


@pytest.fixture(scope='module')
def templates():
    return get_backend_template_loader()


def squeeze(text):
    """Collapse each run of whitespace, so a test reads past the layout of generated code"""
    return ' '.join(text.split())


def test_rewrite_rules(accelerator, templates):
    rules = [
        generate_rewrite_rule(m, templates)
        for m in get_instruction_metadata(accelerator.instructions)
    ]
    # The semantics as a pattern over its inputs, `_?` wherever an attribute is carried
    assert [squeeze(rule) for rule in rules] == [
        'load-rm: (bitcvt (reshape_? ?a)) => (load-rm_? ?a)',
        'gemm: (dot ?a ?b) => (gemm ?a ?b)',
    ]


def test_buffer_assignments(accelerator):
    info = extract_instruction_info(accelerator.instructions)
    # The output's buffer first, then each input's, d0 as HBM
    assert [i.buffer_assignment for i in info] == [
        'Some(vec![Buffer::D1, Buffer::HBM])',
        'Some(vec![Buffer::D2, Buffer::D1, Buffer::D1])',
    ]


def test_enum_variants(accelerator, templates):
    info = extract_instruction_info(accelerator.instructions)
    parts = ir_egraph_rs_generator.generate_instruction_variants(info, templates)
    # A computational attribute is carried as a string beside the inputs
    assert [squeeze(variant) for variant in parts['enum_variants']] == [
        'LoadRm(String, [Id; 1]),',
        'Gemm([Id; 2]),',
    ]


def test_preconditions(accelerator, templates):
    load_rm, gemm = get_instruction_metadata(accelerator.instructions)
    functions = squeeze(generate_ir2isa_rust_functions(load_rm, templates))
    # The attribute is read from the root's first dimension
    assert 'let n = lhs_metadata[2].shape[0];' in functions
    # Every value of the semantics is held to its shape and type
    assert functions.count('if lhs_metadata[') == 3
    for index, shape, dtype in [(1, 'n, 64, 2', 'U8'), (2, 'n, 64', 'BF16')]:
        value = f'TensorInfo {{ shape: vec![{shape}], dtype: Dtype::{dtype}, is_const: false, }}'
        assert f'if lhs_metadata[{index}] != ({value})' in functions
    # An instruction with no attribute reads none
    assert 'shape[0]' not in generate_ir2isa_rust_functions(gemm, templates)


def test_address_ranges(accelerator, templates):
    load_rm = squeeze(generate_instruction_class(accelerator.instructions[0], templates))
    # Range 0 is the output's, range 1 the input's, each its address and its size
    assert 'get_h0() { return {{addr_out, MAKE_SUM(addr_out, n)}}; }' in load_rm
    assert 'get_h1() { return {{addr_in, MAKE_SUM(addr_in, n * 128)}}; }' in load_rm


def test_storages(accelerator):
    models = generate_globals_models(accelerator.data_model)
    # One storage per data model but d0: its access dimensions, then its unit dimension count
    assert [squeeze(model).rstrip(',') for model in models] == [
        '{"D1", new Storage("D1", {128}, 1)}',
        '{"D2", new Storage("D2", {64}, 1)}',
    ]


def test_storage_with_no_access_dimension():
    acc = Accelerator('ONE')
    acc.add_data_model('d1', [], [64], 'bf16')
    (model,) = generate_globals_models(acc.data_model)
    assert squeeze(model) == '{"D1", new Storage("D1", {}, 1)}'


def test_rule_functions(accelerator):
    precond, metadata, set_shapes = generate_function_mappings(accelerator.instructions)
    # A rule is named in kebab-case, its three functions after the instruction
    assert [squeeze(arm) for arm in precond] == [
        '"load-rm" => precond_load_rm,',
        '"gemm" => precond_gemm,',
    ]
    assert [squeeze(arm) for arm in metadata] == [
        '"load-rm" => metadata_load_rm,',
        '"gemm" => metadata_gemm,',
    ]
    assert [squeeze(arm) for arm in set_shapes] == [
        '"load-rm" => set_shapes_load_rm,',
        '"gemm" => set_shapes_gemm,',
    ]


def test_buffer_names(accelerator, templates):
    # Every buffer but HBM, in the order the spec declares them
    assert generate_buffer_names(accelerator.data_model, templates) == '"D1", "D2"'


def test_parser_cases(accelerator, templates):
    cases = squeeze(generate_parser_instructions(accelerator.instructions, templates))
    # One case per instruction, in the spec's order, an attribute passed to its class
    assert re.findall(r'op_name == "(\w+)"', cases) == ['load_rm', 'gemm']
    assert 'return new LoadRm(lhs_tensor, rhs_tensors, n);' in cases
    assert 'return new Gemm(lhs_tensor, rhs_tensors);' in cases


def test_fill_placeholders():
    source = 'enum E {\n  A,\n  // {{VARIANTS}}\n}\nf(x, {/* {{NAMES}} */});\n{{RULES}}\n'
    values = {'VARIANTS': '  B,\n  C,', 'NAMES': '"B", "C"', 'RULES': 'r: a => b'}
    # A comment line goes whole, its indentation included; a block comment and a bare name in place
    filled = 'enum E {\n  A,\n  B,\n  C,\n}\nf(x, {"B", "C"});\nr: a => b\n'
    assert fill_placeholders(source, values, 'e.rs') == filled


def test_placeholder_without_value_refused():
    source = '// {{VARIANTS}}\n/* {{NAMES}} */\n'
    with pytest.raises(ValueError, match=r'^e\.rs: no value for the placeholder NAMES$'):
        fill_placeholders(source, {'VARIANTS': ''}, 'e.rs')


def test_value_without_placeholder_refused():
    with pytest.raises(ValueError, match=r'^e\.rs: no placeholder for the value NAMES$'):
        fill_placeholders('// {{VARIANTS}}\n', {'VARIANTS': '', 'NAMES': ''}, 'e.rs')


def test_fragment_key_without_value_refused(templates):
    fragment = 'egraph.rs.ISA_ENUM_VARIANTS.variant.txt'
    with pytest.raises(
        ValueError, match=rf'^{re.escape(fragment)}: no value for the key signature$'
    ):
        templates.render(fragment, variant_name='Gemm')


def test_value_without_fragment_key_refused(templates):
    fragment = 'egraph.rs.ISA_ENUM_VARIANTS.variant.txt'
    with pytest.raises(ValueError, match=rf'^{re.escape(fragment)}: no key for the value arity$'):
        templates.render(fragment, variant_name='Gemm', signature='([Id; 2])', arity='2')
