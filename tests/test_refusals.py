"""Tests of what generation refuses: a spec no backend can be built for, each by its error."""

import os
import re

import pytest
from taidl import Accelerator

import taidl.antlr4

import act_backend
from act_backend import generate_backend, spec_checks
from act_backend.generator import get_instruction_metadata
from act_backend.isel_ir2isa_rewrites_rs_generator import generate_ir2isa_rust_functions
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
ENTRY load {{
    %In1 = u8[`@c.{attribute} * 128`] parameter(0);
    %a = u8[`@c.{attribute}`,64,2] reshape(%In1);
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
    """An accelerator whose one instruction, gemm, has the given inputs and semantics lines"""
    acc = Accelerator('ONE')
    acc.add_data_model('d1', [128], [64], 'bf16')
    acc.add_data_model('d2', [64], [64], 'bf16')
    addresses = [f'addr_{number}' for number in range(1, inputs + 1)]
    instr = acc.add_instruction('gemm', [], [*addresses, 'addr_out'])
    instr.set_inputs([['d1', [f'@a.{address}'], ['64']] for address in addresses])
    instr.set_outputs([['d2', ['@a.addr_out'], ['64']]])
    instr.add_semantics('ENTRY gemm {\n' + ''.join(f'    {line};\n' for line in lines) + '}\n')
    return acc


# The two inputs of gemm, and its result by an operation
IN1 = '%In1 = bf16[64,64] parameter(0)'
IN2 = '%In2 = bf16[64,64] parameter(1)'


def result(operation):
    return f'ROOT %Out0 = bf16[64,64] {operation}'


def grammar_words():
    """The operation words the installed grammar lexes"""
    with open(os.path.join(os.path.dirname(taidl.antlr4.__file__), 'IDLV2.g4')) as f:
        grammar = re.sub(r'//.*', '', f.read())
    rule = re.search(r'^OPERATION\s*:(.*?);', grammar, re.M | re.S).group(1)
    return re.findall(r"'(\w+)'", rule)


def refusal(accelerator, tmp_path):
    """Return the error a generation ends on, which has written nothing"""
    with pytest.raises(ValueError) as error:
        generate_backend(accelerator, str(tmp_path / 'base'))
    assert not (tmp_path / 'base').exists()
    return str(error.value)


@pytest.mark.parametrize('name', ['', '.', '..', '../../outside', 'a/b'])
def test_accelerator_name_that_is_no_directory_refused(name, tmp_path):
    # What the name would reach from the base directory
    kept = tmp_path / 'outside' / 'backend' / 'kept'
    kept.parent.mkdir(parents=True)
    kept.touch()
    message = refusal(loader(name=name), tmp_path)
    assert message == f"Accelerator name '{name}' is not the name of one directory"
    assert kept.exists()


@pytest.mark.parametrize(
    'name, reason',
    [
        ('load-2', 'is not an identifier'),
        ('while', 'is a Python keyword, which no assembly can spell'),
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
        ({'attribute': 'n-1'}, "computational attribute 'n-1' is not an identifier"),
        (
            {'address': 'class'},
            "address attribute 'class' is a Python keyword, which no assembly can spell",
        ),
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
        ({'address': 'n'}, "attribute 'n' is given twice"),
        ({'address': 'addr_in'}, "attribute 'addr_in' is given twice"),
    ],
)
def test_attribute_name_refused(names, reason, tmp_path):
    assert refusal(loader(**names), tmp_path) == f'load: {reason}'


@pytest.mark.parametrize(
    'name, reason',
    [
        ('d-1', 'is not an identifier'),
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
    variants = set(re.findall(r'^  (\w+)[(,]', variants, re.M))
    assert variants | {'Self'} == spec_checks.EGRAPH_VARIANTS

    # Each buffer beside the data models'
    buffer = read('src', 'ir', 'buffer.rs')
    buffers = buffer[buffer.index('pub enum Buffer') : buffer.index('impl ')]
    assert set(re.findall(r'^  (\w+),', buffers, re.M)) == spec_checks.BUFFERS

    # Each type the allocator declares
    allocator = os.path.join('cpp', 'malloc')
    sources = [
        read(allocator, directory, name)
        for directory in ['include', 'src']
        for name in sorted(os.listdir(os.path.join(generic, allocator, directory)))
    ]
    types = re.findall(r'^(?:class|struct) (\w+)|^using (\w+) =', '\n'.join(sources), re.M)
    assert {a or b for a, b in types} == spec_checks.ALLOCATOR_TYPES

    # Each name the sources generated for load_rm spell, its own apart, with its macros' bodies
    templates = get_backend_template_loader()
    load_rm = accelerator.instructions[0]
    own = {'LoadRm', 'n', 'n_str', 'addr_in', 'addr_out'}
    in_class = spelled(generate_instruction_class(load_rm, templates))
    macros = re.findall(
        r'^#define (\w+)\(([^)]*)\) (.*)$', read(allocator, 'include', 'solver.h'), re.M
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


@pytest.mark.parametrize(
    'lines, reason',
    [
        (
            [IN1, IN2, result('maximum(%In1, %In2)')],
            "uses the operation 'maximum', which the selector does not have",
        ),
        (
            [IN1, IN2, result('add(%In1, %In2, %In1)')],
            "gives the operation 'add' 3 values, where the selector's takes 2",
        ),
        (
            [IN1, IN2, '%a = bf16[64,64] add(%In1, %In2)', result('divide(%a, 2)')],
            "gives the operation 'divide' 1 values, where the selector's takes 2",
        ),
        (
            [IN1, '%In2 = bf16[64,64] parameter(%In1)', result('dot(%In1, %In2)')],
            "reads parameter(%In1), not one input's number",
        ),
    ],
)
def test_operation_the_selector_lacks_refused(lines, reason, tmp_path):
    assert refusal(with_semantics(lines), tmp_path) == f'gemm: its semantics {reason}'


@pytest.mark.parametrize('word', [word for word in grammar_words() if word != 'parameter'])
def test_operation_word_handled_or_refused(word):
    lines = spec_checks.SemanticsChecker('gemm')
    # Handled: a line that gives the word the values it takes passes
    if word in OPERATIONS:
        _, count = OPERATIONS[word]
        operands = ', '.join(['%In1'] * count)
        lines.visit(
            with_semantics([IN1, result(f'{word}({operands})')], 1)
            .instructions[0]
            .instr_semantics_ast
        )
    else:
        acc = with_semantics([IN1, result(f'{word}(%In1)')], 1)
        with pytest.raises(ValueError, match=f"uses the operation '{word}', which the selector"):
            lines.visit(acc.instructions[0].instr_semantics_ast)


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


def test_concatenate_of_three_refused(tmp_path):
    lines = [IN1, IN2, result('concatenate(%In1, %In2, %In1), dimensions={0}')]
    message = refusal(with_semantics(lines), tmp_path)
    assert message == (
        "gemm: its semantics gives the operation 'concatenate' 3 values, "
        "where the selector's takes 2"
    )


@pytest.mark.parametrize(
    'lines, value',
    [
        ([IN1, IN2, result('dot(%In1, %In3)')], '%In3'),
        (
            [
                IN1,
                IN2,
                '%a = bf16[64,64] add(%In1, %b)',
                '%b = bf16[64,64] copy(%In2)',
                result('copy(%a)'),
            ],
            '%b',
        ),
    ],
)
def test_operand_that_names_no_value_refused(lines, value, tmp_path):
    message = refusal(with_semantics(lines), tmp_path)
    assert message == f"gemm: its semantics reads '{value}', which no line above defines"


def test_value_defined_twice_refused(tmp_path):
    lines = [
        IN1,
        IN2,
        '%a = bf16[64,64] add(%In1, %In2)',
        '%a = bf16[64,64] copy(%a)',
        result('copy(%a)'),
    ]
    assert refusal(with_semantics(lines), tmp_path) == "gemm: its semantics defines '%a' twice"


@pytest.mark.parametrize(
    'lines, inputs, value',
    [
        # An input no operation reads, and a line whose value none reads
        ([IN1, IN2, '%In3 = bf16[64,64] parameter(2)', result('dot(%In1, %In2)')], 3, '%In3'),
        ([IN1, IN2, '%a = bf16[64,64] exponential(%In1)', result('dot(%In1, %In2)')], 2, '%a'),
    ],
)
def test_value_no_line_reads_refused(lines, inputs, value, tmp_path):
    message = refusal(with_semantics(lines, inputs), tmp_path)
    assert message == f"gemm: its semantics defines '{value}', which no line below reads"


def test_root_above_the_last_line_refused(tmp_path):
    lines = [IN1, IN2, result('dot(%In1, %In2)'), '%after = bf16[64,64] exponential(%Out0)']
    message = refusal(with_semantics(lines), tmp_path)
    assert message == "gemm: its semantics marks '%Out0' ROOT above its last line"


def test_parameter_under_another_name_refused(tmp_path):
    # The oracle would read %In1 as input 0, where this semantics says input 1
    lines = ['%In1 = bf16[64,64] parameter(1)', '%In2 = bf16[64,64] parameter(0)']
    message = refusal(with_semantics([*lines, result('dot(%In1, %In2)')]), tmp_path)
    assert message == "gemm: its semantics names parameter(1) '%In1', not '%In2'"
