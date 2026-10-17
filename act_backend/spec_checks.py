"""What a backend cannot be generated for: each refused by name, before anything is written"""

from antlr4.tree.Tree import TerminalNode
from antlr4.tree.Trees import Trees
from taidl.antlr4 import IDLV2ExprParser

from .isel_ir2isa_rewrites_rs_generator import (
    ELEMENT_TYPES,
    RustGeneratorVisitor,
    attribute_reading,
)
from .isel_ir2isa_rewrites_txt_generator import OPERATIONS
from .semantics_visitor import OperandVisitor


def check_outputs(instructions):
    """Refuse an instruction with other than one output: the allocator places one"""
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
                'which the selector does not support'
            )
        operands = self.visit(ctx.operands()) if ctx.operands() else []
        if word == 'parameter':
            # An input, read by its number
            self.inputs.append(int(operands[0][0]))
        else:
            if word not in OPERATIONS:
                raise ValueError(
                    f"{self.name}: its semantics uses the operation '{word}', "
                    'which the selector does not support'
                )
            # The operator takes the operands that name a value
            named = [text for text, kind in operands if kind == 'VALUE']
            _, count = OPERATIONS[word]
            if len(named) != count:
                raise ValueError(
                    f"{self.name}: its semantics gives the operation '{word}' {len(named)} values, "
                    f'where the selector takes {count}'
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
        # The pattern is the last line's, holding what it reads
        for value in checker.values[:-1]:
            if value not in checker.read:
                raise ValueError(
                    f"{name}: its semantics defines '{value}', which no line below reads"
                )
            if value in checker.roots:
                raise ValueError(f"{name}: its semantics marks '{value}' ROOT above its last line")
        # The rewrite matches over the inputs, one parameter each
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
        for attribute in instruction.comp_attr:
            if attribute not in lines['comp_attributes']:
                raise ValueError(
                    f'{name}: no shape of its semantics names the computational attribute '
                    f"'{attribute}'"
                )
            # Some dimension holds it alone or times a positive integer
            if attribute_reading(lines['assignments'], attribute) is None:
                raise ValueError(
                    f'{name}: no dimension of its semantics holds the computational attribute '
                    f"'{attribute}' alone or times a positive integer"
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
                    'which the backend does not support'
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
            'which the backend does not support'
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
