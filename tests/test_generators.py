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
    # The semantics as a pattern, `_?` for each carried attribute
    assert [squeeze(rule) for rule in rules] == [
        'load-rm: (bitcvt (reshape_? ?v0)) => (load-rm_? ?v0)',
        'gemm: (dot ?v0 ?v1) => (gemm ?v0 ?v1)',
    ]


def test_pattern_variables_are_numbered(templates):
    # One input read 27 times; later reads get fresh variables
    acc = Accelerator('SUMS')
    acc.add_data_model('d1', [128], [64], 'bf16')
    instr = acc.add_instruction('sums', [], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['64']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['64']]])
    lines = ['%In0 = bf16[64,64] parameter(0)', '%a0 = bf16[64,64] add(%In0, %In0)']
    lines += [f'%a{k} = bf16[64,64] add(%a{k - 1}, %In0)' for k in range(1, 25)]
    lines += ['ROOT %Out0 = bf16[64,64] add(%a24, %In0)']
    instr.add_semantics('ENTRY {\n' + ''.join(f'    {line};\n' for line in lines) + '}\n')
    rewrite = generate_rewrite_rule(get_instruction_metadata(acc.instructions)[0], templates)
    # Each variable a name of its own
    variables = re.findall(r'(?<=[\s(])\?[^\s()]+', rewrite.split('=>')[0])
    assert len(set(variables)) == 27
    assert all(re.fullmatch(r'\?\w+', variable) for variable in variables)


def test_copy_of_a_copy_has_variables_of_its_own(templates):
    # A sum of the input with itself, read twice
    acc = Accelerator('FOUR')
    acc.add_data_model('d1', [128], [64], 'bf16')
    instr = acc.add_instruction('four', [], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['64']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %a = bf16[64,64] add(%In0, %In0);
    ROOT %Out0 = bf16[64,64] add(%a, %a);
}
""")
    rewrite = generate_rewrite_rule(get_instruction_metadata(acc.instructions)[0], templates)
    assert squeeze(rewrite) == 'four: (add (add ?v0 ?v1) (add ?v2 ?v3)) => (four ?v0)'


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
    # A computational attribute rides as a string beside the inputs
    assert [squeeze(variant) for variant in parts['enum_variants']] == [
        'LoadRm(String, [Id; 1]),',
        'Gemm([Id; 2]),',
    ]


def test_display_names_each_attribute(templates):
    acc = Accelerator('ONE')
    acc.add_data_model('d1', [128], [64], 'bf16')
    instr = acc.add_instruction('top', ['m', 'n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['@c.m']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[`@c.m`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] slice(%In0), slice={[0:`@c.n`], [0:64]};
}
""")
    info = extract_instruction_info(acc.instructions)
    parts = ir_egraph_rs_generator.generate_instruction_variants(info, templates)
    # The values the allocator reads, each attribute named
    assert squeeze(parts['display_arms'][0]) == (
        'TensorOp::Top(data, _) => write!(f, "top[m,n=\'{}\']", data),'
    )


def test_preconditions(accelerator, templates):
    load_rm, gemm = get_instruction_metadata(accelerator.instructions)
    functions = squeeze(generate_ir2isa_rust_functions(load_rm, templates))
    # Read from the input's dimension, over its factor
    assert 'let attr_n = lhs_metadata[0].shape[0] / 128;' in functions
    # Each semantics value is held to its shape and type
    assert functions.count('if lhs_metadata[') == 3
    shapes = [(0, 'attr_n * 128', 'U8'), (1, 'attr_n, 64, 2', 'U8'), (2, 'attr_n, 64', 'BF16')]
    for index, shape, dtype in shapes:
        value = f'TensorInfo {{ shape: vec![{shape}], dtype: Dtype::{dtype}, is_const: false, }}'
        assert f'if lhs_metadata[{index}] != ({value})' in functions
    # An instruction with no attribute reads none
    assert 'shape[0]' not in generate_ir2isa_rust_functions(gemm, templates)


def with_one_attribute(lines):
    """The rewrite functions of an instruction on d1 with the attribute n and the given semantics"""
    acc = Accelerator('ONE')
    acc.add_data_model('d1', [128], [64], 'bf16')
    instr = acc.add_instruction('one', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d1', ['@a.addr_in'], ['@c.n']]])
    instr.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])
    instr.add_semantics('ENTRY {\n' + ''.join(f'    {line};\n' for line in lines) + '}\n')
    (one,) = get_instruction_metadata(acc.instructions)
    return squeeze(generate_ir2isa_rust_functions(one, get_backend_template_loader()))


@pytest.mark.parametrize('dimension', ['@c.n * (128)', '128 * @c.n', '(@c.n) * 128'])
def test_attribute_read_through_its_factor(dimension):
    functions = with_one_attribute(
        [f'%In0 = u8[`{dimension}`] parameter(0)', f'ROOT %Out0 = u8[`{dimension}`] copy(%In0)']
    )
    # The dimension over its factor, either side, through brackets
    assert functions.count('let attr_n = lhs_metadata[0].shape[0] / 128;') == 3


def test_attribute_read_from_a_second_dimension():
    functions = with_one_attribute(
        [
            '%In0 = bf16[64,`@c.n`] parameter(0)',
            'ROOT %Out0 = bf16[64,`@c.n`] exponential(%In0)',
        ]
    )
    # The precondition first checks that the value has two dimensions
    guard = 'if lhs_metadata[0].shape.len() <= 1 { return false; }'
    assert f'{guard} let attr_n = lhs_metadata[0].shape[1];' in functions
    assert functions.count('let attr_n = lhs_metadata[0].shape[1];') == 3


def test_address_ranges(accelerator, templates):
    load_rm = squeeze(generate_instruction_class(accelerator.instructions[0], templates))
    # Range 0 the output's, 1 the input's: address and size
    assert 'get_h0() { return {{attr_addr_out, MAKE_SUM(attr_addr_out, attr_n)}}; }' in load_rm
    assert 'get_h1() { return {{attr_addr_in, MAKE_SUM(attr_addr_in, attr_n * 128)}}; }' in load_rm
    # The assembly prints each attribute by the spec's name
    assert 'n = " + std::to_string(attr_n)' in load_rm
    assert 'addr_in = " + std::to_string(attr_addr_in->Value())' in load_rm


def test_size_written_from_its_tree(templates):
    acc = Accelerator('ONE')
    acc.add_data_model('d1', [128], [64], 'bf16')
    instr = acc.add_instruction('load', ['n'], ['addr_in', 'addr_out'])
    instr.set_inputs([['d0', ['@a.addr_in'], ['(@c.n+1)*2-@c.n/2']]])
    instr.set_outputs([['d1', [' @a.addr_out '], [64]]])
    load = squeeze(generate_instruction_class(instr, templates))
    # Operators spaced, the spec's brackets, attributes by generated name
    assert 'MAKE_SUM(attr_addr_in, (attr_n + 1) * 2 - attr_n / 2)' in load
    assert 'MAKE_SUM(attr_addr_out, 64)' in load


def test_storages(accelerator):
    models = generate_globals_models(accelerator.data_model)
    # Per data model but d0: access dimensions, then unit count
    assert [squeeze(model) for model in models] == [
        '{"D1", new Storage("D1", {128}, 1)},',
        '{"D2", new Storage("D2", {64}, 1)},',
    ]


def test_storage_with_no_access_dimension():
    acc = Accelerator('ONE')
    acc.add_data_model('d1', [], [64], 'bf16')
    (model,) = generate_globals_models(acc.data_model)
    assert squeeze(model) == '{"D1", new Storage("D1", {}, 1)},'


def test_rule_functions(accelerator):
    precond, metadata, set_shapes = generate_function_mappings(accelerator.instructions)
    # Rewrites in kebab-case, functions named after the instruction
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
    # Every buffer but HBM, in the spec's order
    assert generate_buffer_names(accelerator.data_model, templates) == '"D1", "D2"'


def test_parser_cases(accelerator, templates):
    cases = squeeze(generate_parser_instructions(accelerator.instructions, templates))
    # A case per instruction, in order, attributes to its class
    assert re.findall(r'op_name == "(\w+)"', cases) == ['load_rm', 'gemm']
    assert 'return new LoadRm(lhs_tensor, rhs_tensors, attr_n);' in cases
    assert 'return new Gemm(lhs_tensor, rhs_tensors);' in cases


def test_fill_placeholders():
    source = 'enum E {\n  A,\n  // {{VARIANTS}}\n}\nf(x, {/* {{NAMES}} */});\n{{RULES}}\n'
    values = {'VARIANTS': '  B,\n  C,', 'NAMES': '"B", "C"', 'RULES': 'r: a => b'}
    # Comment lines replaced whole; block and bare ones in place
    filled = 'enum E {\n  A,\n  B,\n  C,\n}\nf(x, {"B", "C"});\nr: a => b\n'
    assert fill_placeholders(source, values, 'e.rs') == filled


def test_placeholder_with_no_code_leaves_no_line():
    source = 'enum E {\n  A,\n  // {{VARIANTS}}\n}\n\n// {{FUNCTIONS}}\n\nfn f() {}\n'
    # Nor the blank line above it
    filled = 'enum E {\n  A,\n}\n\nfn f() {}\n'
    assert fill_placeholders(source, {'VARIANTS': '', 'FUNCTIONS': ''}, 'e.rs') == filled


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
