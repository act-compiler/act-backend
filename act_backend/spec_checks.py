"""What a backend cannot be generated for: each refused by name, before anything is written"""

import re

from antlr4.tree.Tree import TerminalNode
from antlr4.tree.Trees import Trees
from taidl.antlr4 import IDLV2ExprParser

from .instruction_info import get_buffer_name, get_kebab_name, to_camel_case
from .isel_ir2isa_rewrites_rs_generator import (
    ELEMENT_TYPES,
    RustGeneratorVisitor,
    attribute_factor,
)
from .isel_ir2isa_rewrites_txt_generator import OPERATIONS
from .semantics_visitor import OperandVisitor

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

RUST_KEYWORDS = {
    'abstract',
    'as',
    'async',
    'await',
    'become',
    'box',
    'break',
    'const',
    'continue',
    'crate',
    'do',
    'dyn',
    'else',
    'enum',
    'extern',
    'false',
    'final',
    'fn',
    'for',
    'if',
    'impl',
    'in',
    'let',
    'loop',
    'macro',
    'match',
    'mod',
    'move',
    'mut',
    'override',
    'priv',
    'pub',
    'ref',
    'return',
    'self',
    'Self',
    'static',
    'struct',
    'super',
    'trait',
    'true',
    'try',
    'type',
    'typeof',
    'unsafe',
    'unsized',
    'use',
    'virtual',
    'where',
    'while',
    'yield',
}

CPP_KEYWORDS = {
    'alignas',
    'alignof',
    'and',
    'and_eq',
    'asm',
    'auto',
    'bitand',
    'bitor',
    'bool',
    'break',
    'case',
    'catch',
    'char',
    'char16_t',
    'char32_t',
    'class',
    'compl',
    'const',
    'const_cast',
    'constexpr',
    'continue',
    'decltype',
    'default',
    'delete',
    'do',
    'double',
    'dynamic_cast',
    'else',
    'enum',
    'explicit',
    'export',
    'extern',
    'false',
    'float',
    'for',
    'friend',
    'goto',
    'if',
    'inline',
    'int',
    'long',
    'mutable',
    'namespace',
    'new',
    'noexcept',
    'not',
    'not_eq',
    'nullptr',
    'operator',
    'or',
    'or_eq',
    'private',
    'protected',
    'public',
    'register',
    'reinterpret_cast',
    'return',
    'short',
    'signed',
    'sizeof',
    'static',
    'static_assert',
    'static_cast',
    'struct',
    'switch',
    'template',
    'this',
    'thread_local',
    'throw',
    'true',
    'try',
    'typedef',
    'typeid',
    'typename',
    'union',
    'unsigned',
    'using',
    'virtual',
    'void',
    'volatile',
    'wchar_t',
    'while',
    'xor',
    'xor_eq',
}

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


def check_instruction_names(instructions):
    """Refuse an instruction name the selector or the allocator cannot carry"""
    carried = {}
    for instruction in instructions:
        name = instruction.instruction
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


def check_data_model_names(data_models):
    """Refuse a data model name that gives no buffer of its own"""
    carried = {}
    for data_model in data_models:
        name = data_model.var_name
        # d0 is the HBM, which the backend holds itself
        if name == 'd0':
            continue
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


def check_outputs(instructions):
    """Refuse an instruction that does not have one output, which the allocator places"""
    for instruction in instructions:
        if len(instruction.instr_outputs) != 1:
            raise ValueError(
                f'{instruction.instruction}: the instruction has '
                f'{len(instruction.instr_outputs)} outputs, and the allocator places one'
            )


class SemanticsChecker(OperandVisitor):
    """Refuses each line of a semantics the selector cannot write into a rule"""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.values = []
        self.read = set()
        self.roots = []
        self.inputs = []

    def visitInstruction(self, ctx):
        word = ctx.OPERATION().getText()
        value = ctx.IDENTIFIER().getText()
        element_type = ctx.result_type().TYPE().getText()
        if element_type not in ELEMENT_TYPES:
            raise ValueError(
                f"{self.name}: its semantics uses the element type '{element_type}', "
                'which the selector does not have'
            )
        operands = self.visit(ctx.operands()) if ctx.operands() else []
        if word == 'parameter':
            # An input, read by its number
            self.inputs.append(int(operands[0][0]))
        else:
            if word not in OPERATIONS:
                raise ValueError(
                    f"{self.name}: its semantics uses the operation '{word}', "
                    'which the selector does not have'
                )
            # The operands that name a value are what the operator takes
            named = [text for text, kind in operands if kind == 'VALUE']
            _, count = OPERATIONS[word]
            if len(named) != count:
                raise ValueError(
                    f"{self.name}: its semantics gives the operation '{word}' {len(named)} values, "
                    f"where the selector's takes {count}"
                )
            self.read.update(named)
        self.values.append(value)
        if ctx.ROOT():
            self.roots.append(value)


# Each operator as the language reads it, `/` rounding down
ARITHMETIC = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a // b,
    '%': lambda a, b: a % b,
}


def integer(value):
    """The integer a parsed value of integers alone holds, else None"""
    if isinstance(value, IDLV2ExprParser.AtomContext):
        if value.INT() is not None:
            return int(value.getText())
        return None if value.reference() is not None else integer(value.expression())
    if isinstance(value, IDLV2ExprParser.UnaryContext):
        if value.MINUS() is None:
            return integer(value.atom())
        negated = integer(value.unary())
        return None if negated is None else -negated
    # A sum or a product, its operands in order
    parts = list(value.getChildren())
    result = integer(parts[0])
    for operator, operand in zip(parts[1::2], parts[2::2]):
        right = integer(operand)
        if result is None or right is None or (right == 0 and operator.getText() in ('/', '%')):
            return None
        result = ARITHMETIC[operator.getText()](result, right)
    return result


def divides_by_zero(expression):
    """Whether a parsed value divides, or takes a remainder, by a constant zero"""
    for node in Trees.descendants(expression):
        if isinstance(node, IDLV2ExprParser.MultiplicativeContext):
            for position, operand in enumerate(node.unary()[1:]):
                operator = node.getChild(2 * position + 1).getText()
                if operator in ('/', '%') and integer(operand) == 0:
                    return True
    return False


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
            if value in checker.roots:
                raise ValueError(f"{name}: its semantics marks '{value}' ROOT above its last line")
        # The rule is matched over the inputs, each read by one parameter
        if len(checker.inputs) != len(instruction.instr_inputs):
            raise ValueError(
                f'{name}: its semantics reads {len(checker.inputs)} inputs by parameter, '
                f'and the instruction has {len(instruction.instr_inputs)}'
            )
        if not checker.inputs:
            raise ValueError(
                f'{name}: the instruction has no inputs, which the selector cannot match'
            )
        # An address is set after selection: no line reads one
        for expression in instruction.instr_expression_asts.values():
            placed = Trees.findAllTokenNodes(expression, IDLV2ExprParser.ADDRESSING)
            if placed:
                raise ValueError(
                    f"{name}: its semantics reads the address attribute '{placed[0].getText()}', "
                    'which is not known when an instruction is selected'
                )
        # A value whose generated code divides by zero
        for written, expression in instruction.instr_expression_asts.items():
            if divides_by_zero(expression):
                raise ValueError(
                    f"{name}: its semantics divides by zero in '{written.getText()[1:-1]}'"
                )
        # A computational attribute is read from a shape naming it
        visitor = RustGeneratorVisitor(instruction.instr_expression_asts)
        lines = visitor.visit(instruction.instr_semantics_ast)
        root = lines['root_assignment']
        for attribute in instruction.comp_attr:
            if attribute not in lines['comp_attributes']:
                raise ValueError(
                    f'{name}: no shape of its semantics names the computational attribute '
                    f"'{attribute}'"
                )
            # Its value comes from the result's first dimension naming it
            if root['first'] is None or attribute_factor(root['first'], attribute):
                continue
            references = Trees.findAllTokenNodes(root['first'], IDLV2ExprParser.COMPUTATIONAL)
            if f'@c.{attribute}' in [reference.getText() for reference in references]:
                dimension = root['shape'].split(',')[0].strip('`').strip()
                raise ValueError(
                    f"{name}: its result's first dimension '{dimension}' is not the computational "
                    f"attribute '{attribute}' alone or times a positive integer"
                )


def address_attribute(start):
    """Return the address attribute a start is, alone, or None"""
    token = start.start
    if token is start.stop and token.type == IDLV2ExprParser.ADDRESSING:
        return token.text[3:]
    return None


# Tokens a size may not hold, beside a negation
SIZE_TOKENS = (IDLV2ExprParser.PERCENT, IDLV2ExprParser.ADDRESSING, IDLV2ExprParser.STATE)


def beyond_allocator(size):
    """Whether a size holds a remainder, a negation, an address attribute or a state variable"""
    if isinstance(size, TerminalNode):
        return size.getSymbol().type in SIZE_TOKENS
    if isinstance(size, IDLV2ExprParser.UnaryContext) and size.MINUS():
        return True
    return any(beyond_allocator(part) for part in size.getChildren())


def check_size(what, text, size):
    """Refuse a size the allocator's C++ cannot take as it is written"""
    if beyond_allocator(size):
        raise ValueError(
            f"{what} has the size '{text}', which is not integers and computational attributes "
            "joined by '*', '/', '+' and '-'"
        )
    if divides_by_zero(size):
        raise ValueError(f"{what} has the size '{text}', which divides by zero")


def check_operands(instructions, data_models):
    """Refuse an operand the allocator cannot address"""
    access_dimensions = {model.var_name: len(model.access_dim) for model in data_models}
    # The HBM is addressed along one dimension, its bytes
    access_dimensions['d0'] = 1
    for instruction in instructions:
        name = instruction.instruction
        read = set()
        # Each operand as written and as taidl parsed it
        written = [*instruction.instr_outputs, *instruction.instr_inputs]
        parsed = [*instruction.instr_output_asts, *instruction.instr_input_asts]
        for (buffer, addresses, sizes), (_, starts, extents) in zip(written, parsed):
            what = f"{name}: its operand on '{buffer}'"
            if access_dimensions[buffer] != 1:
                raise ValueError(
                    f'{what} is addressed along {access_dimensions[buffer]} dimensions, '
                    'and the allocator addresses one'
                )
            address = address_attribute(starts[0])
            if address is None:
                raise ValueError(
                    f"{what} has the address '{addresses[0]}', "
                    'which is not one address attribute of the instruction'
                )
            check_size(what, sizes[0], extents[0])
            read.add(address)
        for attribute in instruction.parameters:
            if attribute not in read:
                raise ValueError(f"{name}: no operand reads the address attribute '{attribute}'")


def check_inplace(instructions):
    """Refuse an input marked in place: the allocator keeps every input"""
    for instruction in instructions:
        for number, marked in enumerate(instruction.inplace):
            if marked:
                raise ValueError(
                    f'{instruction.instruction}: its input {number} is marked in place, '
                    'which the backend does not read'
                )


def check_constraints(instructions):
    """Refuse an instruction that has a constraint: the backend would emit calls outside it"""
    for instruction in instructions:
        if not instruction.constraints:
            continue
        text, tree = instruction.constraints[0], instruction.constraint_asts[0]
        # A placement mark's constraint is named by its mark
        marks = [
            method
            for method, placed in instruction.placed.items()
            if any(tree is constraint for constraint in placed)
        ]
        origin = f'{marks[0]} gives it' if marks else 'it declares'
        raise ValueError(
            f"{instruction.instruction}: {origin} the constraint '{text}', "
            'which the backend does not read'
        )


def check_state(instructions):
    """Refuse an instruction that reads or updates machine state"""
    for instruction in instructions:
        name = instruction.instruction
        if instruction.update:
            raise ValueError(
                f"{name}: it updates machine state by '{instruction.update[0]}', "
                'which the backend does not model'
            )
        for expression in instruction.instr_expression_asts.values():
            read = Trees.findAllTokenNodes(expression, IDLV2ExprParser.STATE)
            if read:
                raise ValueError(
                    f"{name}: its semantics reads the state variable '{read[0].getText()}', "
                    'which the backend does not model'
                )
