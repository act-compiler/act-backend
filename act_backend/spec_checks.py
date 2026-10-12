"""What a backend cannot be generated for: each refused by name, before anything is written"""

import os
import re
import keyword

from .instruction_info import get_buffer_name, get_kebab_name, to_camel_case
from .isel_ir2isa_rewrites_txt_generator import OPERATIONS
from .semantics_visitor import OperandVisitor

# A name the generated sources can spell as an identifier
IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

# Operators of the e-graph, by the word a rule spells each with
EGRAPH_WORDS = {
    'add',
    'bitcvt',
    'broadcast',
    'concat',
    'constant',
    'convert',
    'copy',
    'divide',
    'dot',
    'exponential',
    'eye',
    'or',
    'reduce',
    'reshape',
    'shift-left',
    'shift-right-logical',
    'slice',
    'xor',
    'transpose',
}

# Variants the e-graph holds beside the instructions', and the name Rust keeps for the type
EGRAPH_VARIANTS = {
    'OpAdd',
    'OpBitcvt',
    'OpBroadcast',
    'OpConcat',
    'OpConstant',
    'OpConvert',
    'OpCopy',
    'OpDivide',
    'OpDot',
    'OpExp',
    'OpEye',
    'OpOr',
    'OpReduceSum',
    'OpReshape',
    'OpShiftLeft',
    'OpShiftRightLogical',
    'OpSlice',
    'OpXor',
    'OpTranspose',
    'DetectedConst',
    'Var',
    'Self',
}

# Types the allocator declares beside the instructions' classes
ALLOCATOR_TYPES = {
    'Instruction',
    'Slice',
    'Concat',
    'Tensor',
    'Storage',
    'MetaData',
    'MetaDataInfo',
    'ParsedLine',
    'IntExpr',
    'IntVar',
    'Constraints',
    'LiveRange',
    'InterferenceGraph',
    'EsunCache',
}

# Words the allocator reads as a declaration where a line's instruction stands
DECLARATION_WORDS = ('Var', 'DCC')

# Buffers the selector holds beside the data models'
BUFFERS = {'HBM', 'ANY'}

RUST_KEYWORDS = set(
    'abstract as async await become box break const continue crate do dyn else enum extern false '
    'final fn for if impl in let loop macro match mod move mut override priv pub ref return self '
    'Self static struct super trait true try type typeof unsafe unsized use virtual where while '
    'yield'.split()
)

CPP_KEYWORDS = set(
    'alignas alignof and and_eq asm auto bitand bitor bool break case catch char char16_t '
    'char32_t class compl const const_cast constexpr continue decltype default delete do double '
    'dynamic_cast else enum explicit export extern false float for friend goto if inline int long '
    'mutable namespace new noexcept not not_eq nullptr operator or or_eq private protected public '
    'register reinterpret_cast return short signed sizeof static static_assert static_cast struct '
    'switch template this thread_local throw true try typedef typeid typename union unsigned '
    'using virtual void volatile wchar_t while xor xor_eq'.split()
)

# Names the generated class of the allocator spells beside an attribute
ADDRESS_ATTRIBUTE_NAMES = {
    'Constraints',
    'INT64_MAX',
    'INT64_MIN',
    'Instruction',
    'IntExpr',
    'IntVar',
    'MAKE_INT_VAR',
    'MAKE_SUM',
    'Tensor',
    'X',
    'assert',
    'constraints',
    'get_e',
    'get_h',
    'get_int_var',
    'int64',
    'num',
    'op_name',
    'override',
    'solver',
    'std',
    'str',
    'y',
}

# Those, and the names the parser's case and the rule functions spell beside a computational one
COMPUTATIONAL_ATTRIBUTE_NAMES = ADDRESS_ATTRIBUTE_NAMES | {
    'attr_str',
    'end',
    'lhs_tensor',
    'panic',
    'rhs_tensors',
    'start',
    'Dtype',
    'EGraph',
    'Id',
    'None',
    'Option',
    'Some',
    'String',
    'TensorInfo',
    'TensorOp',
    'Vec',
    'assert_eq',
    'bool',
    'egraph',
    'id',
    'lhs_eclasses',
    'lhs_metadata',
    'rhs_eclasses',
    'rhs_metadata',
    'vec',
}

# The method the generated class gives each operand's range
RANGE_METHOD = re.compile(r'get_h\d+')


def check_accelerator_name(name):
    """Refuse an accelerator name that is no directory's name: the backend is written under it"""
    if name in ('', '.', '..') or os.path.basename(name) != name:
        raise ValueError(f"Accelerator name '{name}' is not the name of one directory")


def check_identifier(what, name):
    """Refuse a name the generated sources cannot spell"""
    if not IDENTIFIER.fullmatch(name):
        raise ValueError(f"{what} '{name}' is not an identifier")


def check_assembly_name(what, name):
    """Refuse a name the assembly, a Python file, cannot spell"""
    check_identifier(what, name)
    if keyword.iskeyword(name):
        raise ValueError(f"{what} '{name}' is a Python keyword, which no assembly can spell")


def check_instruction_names(instructions):
    """Refuse an instruction name the selector or the allocator cannot carry"""
    carried = {}
    for instruction in instructions:
        name = instruction.instruction
        check_assembly_name('Instruction name', name)
        if get_kebab_name(name) in EGRAPH_WORDS:
            raise ValueError(
                f"Instruction name '{name}' is also the e-graph's operator {get_kebab_name(name)}"
            )
        if name.startswith(DECLARATION_WORDS):
            raise ValueError(
                f"Instruction name '{name}' starts with Var or DCC, "
                'which the allocator reads as a declaration'
            )
        # The variant of the e-graph and the class of the allocator
        generated = to_camel_case(name)
        if not generated[:1].isalpha():
            raise ValueError(
                f"Instruction name '{name}' gives the generated name '{generated}', "
                'which is not an identifier'
            )
        if generated in EGRAPH_VARIANTS | ALLOCATOR_TYPES:
            raise ValueError(
                f"Instruction name '{name}' gives the generated name {generated}, "
                'which the backend already holds'
            )
        if generated in carried:
            raise ValueError(
                f"Instruction names '{carried[generated]}' and '{name}' "
                f'both give the generated name {generated}'
            )
        carried[generated] = name


def check_attribute_names(instruction):
    """Refuse an attribute name the generated sources cannot carry beside their own names"""
    name = instruction.instruction
    seen = set()
    for kind, attributes, languages, keywords, held in [
        (
            'computational',
            instruction.comp_attr,
            'Rust or C++',
            RUST_KEYWORDS | CPP_KEYWORDS,
            COMPUTATIONAL_ATTRIBUTE_NAMES,
        ),
        ('address', instruction.parameters, 'C++', CPP_KEYWORDS, ADDRESS_ATTRIBUTE_NAMES),
    ]:
        for attribute in attributes:
            what = f'{name}: {kind} attribute'
            check_assembly_name(what, attribute)
            if attribute in keywords:
                raise ValueError(
                    f"{what} '{attribute}' is a {languages} keyword, "
                    'which the generated code cannot spell'
                )
            # The parser's case reads a computational attribute through <name>_str
            spelled = {attribute, f'{attribute}_str'} if kind == 'computational' else {attribute}
            if (
                spelled & held
                or RANGE_METHOD.fullmatch(attribute)
                or attribute == to_camel_case(name)
            ):
                raise ValueError(f"{what} '{attribute}' is a name the backend already holds")
            if attribute in seen:
                raise ValueError(f"{name}: attribute '{attribute}' is given twice")
            seen.add(attribute)


def check_data_model_names(data_models):
    """Refuse a data model name that gives no buffer of its own"""
    carried = {}
    for data_model in data_models:
        name = data_model.var_name
        # d0 is the HBM, which the backend holds itself
        if name == 'd0':
            continue
        check_identifier('Data model name', name)
        buffer = get_buffer_name(name)
        if buffer in BUFFERS:
            raise ValueError(
                f"Data model name '{name}' gives the buffer {buffer}, which the backend already holds"
            )
        if buffer in carried:
            raise ValueError(
                f"Data model names '{carried[buffer]}' and '{name}' both give the buffer {buffer}"
            )
        carried[buffer] = name


def check_names(instructions, data_models):
    """Refuse a spec whose names the generated backend cannot carry"""
    check_instruction_names(instructions)
    for instruction in instructions:
        check_attribute_names(instruction)
    check_data_model_names(data_models)


class SemanticsChecker(OperandVisitor):
    """Refuses each line of a semantics the selector cannot write into a rule"""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.values = []
        self.read = set()

    def visitInstruction(self, ctx):
        word = ctx.OPERATION().getText()
        value = ctx.IDENTIFIER().getText()
        operands = self.visit(ctx.operands()) if ctx.operands() else []
        if word == 'parameter':
            # An input, read by its number
            if [kind for _, kind in operands] != ['INT']:
                given = ctx.operands().getText() if ctx.operands() else ''
                raise ValueError(
                    f"{self.name}: its semantics reads parameter({given}), not one input's number"
                )
        else:
            if word not in OPERATIONS:
                raise ValueError(
                    f"{self.name}: its semantics uses the operation '{word}', "
                    'which the selector does not have'
                )
            # The operands that name a value are what the operator takes
            named = [text for text, kind in operands if kind == 'IDENTIFIER']
            _, count = OPERATIONS[word]
            if len(named) != count:
                raise ValueError(
                    f"{self.name}: its semantics gives the operation '{word}' {len(named)} values, "
                    f"where the selector's takes {count}"
                )
            for operand in named:
                if operand not in self.values:
                    raise ValueError(
                        f"{self.name}: its semantics reads '{operand}', which no line above defines"
                    )
            self.read.update(named)
        if value in self.values:
            raise ValueError(f"{self.name}: its semantics defines '{value}' twice")
        self.values.append(value)


def check_semantics(instructions):
    """Refuse a semantics the selector cannot write a rule from"""
    for instruction in instructions:
        name = instruction.instruction
        checker = SemanticsChecker(name)
        checker.visit(instruction.instr_semantics_ast)
        # The rule's pattern is the last line's, and holds what that line reads
        for value in checker.values[:-1]:
            if value not in checker.read:
                raise ValueError(
                    f"{name}: its semantics defines '{value}', which no line below reads"
                )
