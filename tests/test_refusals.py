"""Tests of what generation refuses: a spec no backend can be built for, each by its error."""

import os
import re

import pytest
import taidl.antlr4
import taidl.operations
from taidl import Accelerator

import act_backend
from act_backend import generate_backend, spec_checks
from act_backend.generator import get_instruction_metadata
from act_backend.isel_ir2isa_rewrites_rs_generator import (
    ELEMENT_TYPES,
    generate_ir2isa_rust_functions,
)
from act_backend.isel_ir2isa_rewrites_txt_generator import OPERATIONS
from act_backend.malloc_instructions_h_generator import generate_instruction_class
from act_backend.malloc_parser_cc_generator import generate_parser_instructions
from act_backend.template_loader import get_backend_template_loader


def add_load(acc, instruction='load', attribute='n', address='addr_out', model='d1'):
    """Add an instruction that loads rows into a buffer, each of its names the caller's"""
    instr = acc.add_instruction(instruction, [attribute], ['addr_in', address])
    instr.set_inputs([['d0', ['@a.addr_in'], [f'@c.{attribute} * 128']]])
    instr.set_outputs([[model, [f'@a.{address}'], [f'@c.{attribute}']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[`@c.{attribute} * 128`] parameter(0);
    %a = u8[`@c.{attribute}`,64,2] reshape(%In0);
    ROOT %Out0 = bf16[`@c.{attribute}`,64] bitcast_convert(%a);
}}
""")
    return instr


def loader(name='ONE', model='d1', **names):
    """An accelerator with one data model and one instruction that loads into it"""
    acc = Accelerator(name)
    acc.add_data_model(model, [128], [64], 'bf16')
    add_load(acc, model=model, **names)
    return acc


def with_semantics(lines, inputs=2):
    """An accelerator whose one instruction, gemm, has the given inputs and semantics lines

    With inputs None the instruction never gives its inputs.
    """
    acc = Accelerator('ONE')
    acc.add_data_model('d1', [128], [64], 'bf16')
    acc.add_data_model('d2', [64], [64], 'bf16')
    addresses = [f'addr_{number}' for number in range(1, (inputs or 0) + 1)]
    instr = acc.add_instruction('gemm', [], [*addresses, 'addr_out'])
    if inputs is not None:
        instr.set_inputs([['d1', [f'@a.{address}'], ['64']] for address in addresses])
    instr.set_outputs([['d2', ['@a.addr_out'], ['64']]])
    instr.add_semantics('ENTRY {\n' + ''.join(f'    {line};\n' for line in lines) + '}\n')
    return acc


# The two inputs of gemm, and its result by an operation
IN0 = '%In0 = bf16[64,64] parameter(0)'
IN1 = '%In1 = bf16[64,64] parameter(1)'


def result(operation):
    return f'ROOT %Out0 = bf16[64,64] {operation}'


# The product of gemm's two inputs
DOT = 'dot(%In0, %In1), lhs_contracting_dims={1}, rhs_contracting_dims={0}'


def grammar_words(rule):
    """The words the installed grammar lexes by one rule, OPERATION or TYPE"""
    with open(os.path.join(os.path.dirname(taidl.antlr4.__file__), 'IDLV2.g4')) as f:
        grammar = re.sub(r'//.*', '', f.read())
    words = re.search(rf'^{rule}\s*:(.*?);', grammar, re.MULTILINE | re.DOTALL).group(1)
    return re.findall(r"'(\w+)'", words)


def with_operands(inputs=None, outputs=None, addresses=None, models=(), lines=None, state=()):
    """An accelerator whose one instruction, gemm, has the given operands and address attributes

    With outputs False the instruction never gives its outputs.
    """
    acc = Accelerator('ONE')
    acc.add_data_model('d1', [128], [64], 'bf16')
    acc.add_data_model('d2', [64], [64], 'bf16')
    for model in models:
        acc.add_data_model(*model)
    for variable in state:
        acc.add_initial_state(variable, 0)
    if inputs is None:
        inputs = [['d1', ['@a.addr_1'], ['64']], ['d1', ['@a.addr_2'], ['64']]]
    if outputs is None:
        outputs = [['d2', ['@a.addr_out'], ['64']]]
    if addresses is None:
        addresses = ['addr_1', 'addr_2', 'addr_out']
    instr = acc.add_instruction('gemm', [], addresses)
    instr.set_inputs(inputs)
    if outputs is not False:
        instr.set_outputs(outputs)
    lines = lines or [IN0, IN1, result(DOT)]
    instr.add_semantics('ENTRY {\n' + ''.join(f'    {line};\n' for line in lines) + '}\n')
    return acc


def refusal(accelerator, tmp_path):
    """Return the error a generation ends on, which has written nothing"""
    with pytest.raises(ValueError) as error:
        generate_backend(accelerator, str(tmp_path / 'base'))
    assert not (tmp_path / 'base').exists()
    return str(error.value)


@pytest.mark.parametrize(
    'name, reason',
    [
        ('dot', "is also the e-graph's operator dot"),
        ('shift_left', "is also the e-graph's operator shift-left"),
        ('Variant', 'starts with Var or DCC, which the allocator reads as a declaration'),
        ('_2', "gives the generated name '2', which is not an identifier"),
        ('var', 'gives the generated name Var, which the backend already holds'),
        ('tensor', 'gives the generated name Tensor, which the backend already holds'),
    ],
)
def test_instruction_name_refused(name, reason, tmp_path):
    message = refusal(loader(instruction=name), tmp_path)
    assert message == f"Instruction name '{name}' {reason}"


def test_instruction_names_that_fold_to_one_refused(tmp_path):
    acc = loader(instruction='load_2')
    add_load(acc, instruction='load2')
    message = refusal(acc, tmp_path)
    assert message == "Instruction names 'load_2' and 'load2' both give the generated name Load2"


@pytest.mark.parametrize(
    'names, reason',
    [
        (
            {'attribute': 'type'},
            "computational attribute 'type' is a Rust or C++ keyword, which the generated code cannot spell",
        ),
        (
            {'address': 'register'},
            "address attribute 'register' is a C++ keyword, which the generated code cannot spell",
        ),
        (
            {'attribute': 'lhs_metadata'},
            "computational attribute 'lhs_metadata' is a name the backend already holds",
        ),
        (
            {'attribute': 'start'},
            "computational attribute 'start' is a name the backend already holds",
        ),
        (
            {'attribute': 'attr'},
            "computational attribute 'attr' is a name the backend already holds",
        ),
        (
            {'attribute': 'Load'},
            "computational attribute 'Load' is a name the backend already holds",
        ),
        ({'address': 'solver'}, "address attribute 'solver' is a name the backend already holds"),
        ({'address': 'get_h0'}, "address attribute 'get_h0' is a name the backend already holds"),
        ({'address': 'Tensor'}, "address attribute 'Tensor' is a name the backend already holds"),
    ],
)
def test_attribute_name_refused(names, reason, tmp_path):
    assert refusal(loader(**names), tmp_path) == f'load: {reason}'


@pytest.mark.parametrize(
    'name, reason',
    [
        ('hbm', 'gives the buffer HBM, which the backend already holds'),
        ('any', 'gives the buffer ANY, which the backend already holds'),
    ],
)
def test_data_model_name_refused(name, reason, tmp_path):
    assert refusal(loader(model=name), tmp_path) == f"Data model name '{name}' {reason}"


def test_data_model_names_that_fold_to_one_refused(tmp_path):
    acc = loader(model='d1')
    acc.add_data_model('D1', [64], [64], 'bf16')
    assert refusal(acc, tmp_path) == "Data model names 'd1' and 'D1' both give the buffer D1"


def spelled(text):
    """The names a generated source spells bare: outside a string, a comment and a member path"""
    text = re.sub(r'"[^"\n]*"|//[^\n]*', '', text)
    return set(re.findall(r'(?<![\w.:>])([A-Za-z_]\w*)\b(?!\s*:(?!:))', text))


def test_reserved_names_are_the_skeletons_own(accelerator):
    generic = os.path.join(os.path.dirname(act_backend.__file__), 'generic')

    def read(*path):
        with open(os.path.join(generic, *path)) as f:
            return f.read()

    # Each word the e-graph reads an operator by, and each variant beside the instructions'
    egraph = read('src', 'ir', 'egraph.rs')
    assert set(re.findall(r'== "([a-z-]+)"', egraph)) == spec_checks.EGRAPH_WORDS
    variants = egraph[egraph.index('pub enum TensorOp') : egraph.index('impl TensorOp')]
    variants = set(re.findall(r'^  (\w+)[(,]', variants, re.MULTILINE))
    assert variants | {'Self'} == spec_checks.EGRAPH_VARIANTS

    # Each buffer beside the data models'
    buffer = read('src', 'ir', 'buffer.rs')
    buffers = buffer[buffer.index('pub enum Buffer') : buffer.index('impl ')]
    assert set(re.findall(r'^  (\w+),', buffers, re.MULTILINE)) == spec_checks.BUFFERS

    # Each type the allocator declares
    allocator = os.path.join('cpp', 'malloc')
    sources = [
        read(allocator, directory, name)
        for directory in ['include', 'src']
        for name in sorted(os.listdir(os.path.join(generic, allocator, directory)))
    ]
    types = re.findall(r'^(?:class|struct) (\w+)|^using (\w+) =', '\n'.join(sources), re.MULTILINE)
    assert {a or b for a, b in types} == spec_checks.ALLOCATOR_TYPES

    # Each name the sources generated for load_rm spell, its own apart, with its macros' bodies
    templates = get_backend_template_loader()
    load_rm = accelerator.instructions[0]
    own = {'LoadRm', 'n', 'n_str', 'addr_in', 'addr_out'}
    in_class = spelled(generate_instruction_class(load_rm, templates))
    macros = re.findall(
        r'^#define (\w+)\(([^)]*)\) (.*)$', read(allocator, 'include', 'solver.h'), re.MULTILINE
    )
    for macro, parameters, body in macros:
        if macro in in_class:
            in_class |= spelled(body) - set(re.findall(r'\w+', parameters))
    in_class = {name for name in in_class - own if not spec_checks.RANGE_METHOD.fullmatch(name)}
    in_class -= spec_checks.CPP_KEYWORDS
    assert in_class == spec_checks.ADDRESS_ATTRIBUTE_NAMES
    in_case = spelled(generate_parser_instructions([load_rm], templates)) - own
    functions = generate_ir2isa_rust_functions(get_instruction_metadata([load_rm])[0], templates)
    in_functions = {name for name in spelled(functions) - own if not name.endswith('_load_rm')}
    beside = (
        in_class | in_case - spec_checks.CPP_KEYWORDS | in_functions - spec_checks.RUST_KEYWORDS
    )
    assert beside == spec_checks.COMPUTATIONAL_ATTRIBUTE_NAMES


def test_operation_the_selector_lacks_refused(tmp_path):
    message = refusal(with_semantics([IN0, IN1, result('maximum(%In0, %In1)')]), tmp_path)
    assert message == (
        "gemm: its semantics uses the operation 'maximum', which the selector does not have"
    )


def line_of(word):
    """A line taidl reads for an operation word: its operands, and each attribute it needs"""
    takes, attributes = taidl.operations.OPERATIONS[word]
    if takes == taidl.operations.LITERAL:
        operands = ['0']
    else:
        # The operator's value count, or the fewest taidl reads
        least = takes[0] if isinstance(takes, tuple) else takes
        operands = ['%In0'] * OPERATIONS.get(word, (word, least))[1]
    forms = {
        taidl.operations.INT: '0',
        taidl.operations.INTS: '{0}',
        taidl.operations.ONE_INT: '{0}',
        taidl.operations.RANGES: '{[0:1]}',
        taidl.operations.BOOL: 'true',
        taidl.operations.NUMBER: '0',
    }
    # A closed set of words is given by its first
    given = [
        f'{name}={forms[form] if isinstance(form, str) else form[0]}'
        for name, (form, required) in attributes.items()
        if required
    ]
    return result(', '.join([f'{word}({", ".join(operands)})', *given]))


@pytest.mark.parametrize(
    'word', [word for word in grammar_words('OPERATION') if word != 'parameter']
)
def test_operation_word_handled_or_refused(word):
    lines = spec_checks.SemanticsChecker('gemm')
    semantics = with_semantics([IN0, line_of(word)], 1).instructions[0].instr_semantics_ast
    # Handled: a line that gives the word the values it takes passes
    if word in OPERATIONS:
        lines.visit(semantics)
    else:
        with pytest.raises(ValueError, match=f"uses the operation '{word}', which the selector"):
            lines.visit(semantics)


def test_operations_are_words_of_the_grammar():
    assert set(OPERATIONS) <= set(grammar_words('OPERATION'))


def test_operations_are_the_egraphs_own():
    generic = os.path.join(os.path.dirname(act_backend.__file__), 'generic')
    with open(os.path.join(generic, 'src', 'ir', 'egraph.rs')) as f:
        egraph = f.read()
    # Each operator's word, whether it carries an attribute, and its number of children
    arms = re.findall(
        r"op(\.split\('_'\)\.next\(\)\.unwrap\(\))? == \"([a-z-]+)\"\s+&& <\[Id; (\d)\]", egraph
    )
    carries = {word for carried, word, _ in arms if carried}
    # eye is no operation of the grammar
    children = {word: int(count) for _, word, count in arms if word != 'eye'}
    assert {word.removesuffix('_?'): count for word, count in OPERATIONS.values()} == children
    assert {word[:-2] for word, _ in OPERATIONS.values() if word.endswith('_?')} <= carries


@pytest.mark.parametrize('element_type', grammar_words('TYPE'))
def test_element_type_handled_or_refused(element_type, tmp_path):
    lines = [
        f'%In0 = {element_type}[64,64] parameter(0)',
        f'ROOT %Out0 = {element_type}[64,64] copy(%In0)',
    ]
    acc = with_semantics(lines, 1)
    if element_type in ELEMENT_TYPES:
        spec_checks.check_semantics(acc.instructions)
    else:
        assert refusal(acc, tmp_path) == (
            f"gemm: its semantics uses the element type '{element_type}', "
            'which the selector does not have'
        )


def test_element_types_are_the_egraphs_own():
    generic = os.path.join(os.path.dirname(act_backend.__file__), 'generic')
    with open(os.path.join(generic, 'src', 'ir', 'dtype.rs')) as f:
        dtype = f.read()
    # The kernel reader's element types, by their kernel word
    assert dict(re.findall(r'"(\w+)" => (Dtype::\w+),', dtype)) == ELEMENT_TYPES


def test_concatenate_of_three_refused(tmp_path):
    lines = [IN0, IN1, result('concatenate(%In0, %In1, %In0), dimensions={0}')]
    message = refusal(with_semantics(lines), tmp_path)
    assert message == (
        "gemm: its semantics gives the operation 'concatenate' 3 values, "
        "where the selector's takes 2"
    )


@pytest.mark.parametrize(
    'lines, inputs, value',
    [
        # An input no operation reads, and a line whose value none reads
        ([IN0, IN1, '%In2 = bf16[64,64] parameter(2)', result(DOT)], 3, '%In2'),
        ([IN0, IN1, '%a = bf16[64,64] exponential(%In0)', result(DOT)], 2, '%a'),
    ],
)
def test_value_no_line_reads_refused(lines, inputs, value, tmp_path):
    message = refusal(with_semantics(lines, inputs), tmp_path)
    assert message == f"gemm: its semantics defines '{value}', which no line below reads"


def test_root_above_the_last_line_refused(tmp_path):
    lines = [IN0, IN1, result(DOT), '%after = bf16[64,64] exponential(%Out0)']
    message = refusal(with_semantics(lines), tmp_path)
    assert message == "gemm: its semantics marks '%Out0' ROOT above its last line"


def test_semantics_that_reads_inputs_never_given_refused(tmp_path):
    # taidl ties a parameter to a region after set_inputs
    message = refusal(with_semantics([IN0, IN1, result(DOT)], None), tmp_path)
    assert message == 'gemm: its semantics reads 2 inputs by parameter, and the instruction has 0'


def test_instruction_with_no_inputs_refused(tmp_path):
    message = refusal(with_semantics([result('constant(0)')], inputs=0), tmp_path)
    assert message == 'gemm: the instruction has no inputs, which the selector cannot match'


def test_computational_attribute_no_line_names_refused(tmp_path):
    acc = with_semantics([IN0, IN1, result(DOT)])
    acc.instructions[0].comp_attr.append('k')
    message = refusal(acc, tmp_path)
    assert message == "gemm: no line of its semantics names the computational attribute 'k'"


def test_address_attribute_in_a_semantics_refused(tmp_path):
    lines = [IN0, IN1, f'ROOT %Out0 = bf16[`64 + @a.addr_out`,64] {DOT}']
    message = refusal(with_semantics(lines), tmp_path)
    assert message == (
        "gemm: its semantics reads the address attribute '@a.addr_out', "
        'which is not known when an instruction is selected'
    )


SECOND = ['d1', ['@a.addr_2'], ['64']]


@pytest.mark.parametrize(
    'operands, reason',
    [
        # Two outputs, neither reading the other
        (
            {
                'outputs': [['d2', ['@a.addr_out'], ['64']], ['d2', ['@a.addr_1'], ['64']]],
                'lines': [
                    IN0,
                    IN1,
                    f'%Out0 = bf16[64,64] {DOT}',
                    'ROOT %Out1 = bf16[64,64] copy(%In0)',
                ],
            },
            'the instruction has 2 outputs, and the allocator places one',
        ),
        # An instruction that never gives its outputs
        ({'outputs': False}, 'the instruction has 0 outputs, and the allocator places one'),
        (
            {
                'inputs': [['d3', ['@a.addr_1', '0'], ['64', '4']], SECOND],
                'models': [('d3', [64, 4], [64], 'bf16')],
            },
            "its operand on 'd3' is addressed along 2 dimensions, and the allocator addresses one",
        ),
        (
            {'inputs': [['d3', [], []], SECOND], 'models': [('d3', [], [64, 64], 'bf16')]},
            "its operand on 'd3' is addressed along 0 dimensions, and the allocator addresses one",
        ),
        (
            {'inputs': [['d1', ['@a.addr_1 * 2'], ['64']], SECOND]},
            (
                "its operand on 'd1' has the address '@a.addr_1 * 2', "
                'which is not one address attribute of the instruction'
            ),
        ),
        (
            {'inputs': [['d1', ['0'], ['64']], SECOND]},
            (
                "its operand on 'd1' has the address '0', "
                'which is not one address attribute of the instruction'
            ),
        ),
    ],
)
def test_operand_the_allocator_cannot_address_refused(operands, reason, tmp_path):
    assert refusal(with_operands(**operands), tmp_path) == f'gemm: {reason}'


@pytest.mark.parametrize('size', ['-64', '64 % 48', '@a.addr_out', '@s.rows', '64 * (2 - -1)'])
def test_size_the_allocator_cannot_read_refused(size, tmp_path):
    acc = with_operands(outputs=[['d2', ['@a.addr_out'], [size]]], state=['rows'])
    message = refusal(acc, tmp_path)
    assert message == (
        f"gemm: its operand on 'd2' has the size '{size}', which is not integers and "
        "computational attributes joined by '*', '/', '+' and '-'"
    )


def test_sizes_the_allocator_reads():
    acc = loader()
    (instruction,) = acc.instructions
    # An integer, an attribute, and their products, quotients, sums and differences
    for size in [64, '64', '@c.n', '@c.n * 128', '(@c.n + 1) * 2 - @c.n / 2']:
        instruction.set_outputs([['d1', ['@a.addr_out'], [size]]])
        spec_checks.check_operands(acc.instructions, acc.data_model)


def test_address_attribute_no_operand_reads_refused(tmp_path):
    acc = with_operands(addresses=['addr_1', 'addr_2', 'addr_out', 'spare'])
    assert refusal(acc, tmp_path) == "gemm: no operand reads the address attribute 'spare'"
