"""Rust and C++ for a value or a comparison taidl parsed, each attribute by its generated name"""

from antlr4.tree.Trees import Trees
from taidl.antlr4 import IDLV2ExprParser, IDLV2ExprVisitor

from .instruction_info import get_attribute_name
from .template_loader import CALL_WIDTH, fits

# The allocator's call for each operator that divides, rounding down as the oracle does
ROUNDED_CALLS = {'/': 'floor_div', '%': 'floor_mod'}


class ExpressionWriter(IDLV2ExprVisitor):
    """A value as one expression, its operators spaced and its brackets the text's own"""

    def __init__(self, cpp=False):
        super().__init__()
        self.cpp = cpp

    def visitExpression(self, ctx):
        return self.visit(ctx.additive())

    def chain(self, ctx, operands):
        """A run of operands, the operators between them, a C++ quotient or remainder a call"""
        source = self.visit(operands[0])
        for k, operand in enumerate(operands[1:]):
            operator = ctx.getChild(2 * k + 1).getText()
            if self.cpp and operator in ROUNDED_CALLS:
                source = f'{ROUNDED_CALLS[operator]}({source}, {self.visit(operand)})'
            else:
                source += f' {operator} {self.visit(operand)}'
        return source

    def visitAdditive(self, ctx):
        return self.chain(ctx, ctx.multiplicative())

    def visitMultiplicative(self, ctx):
        return self.chain(ctx, ctx.unary())

    def visitUnary(self, ctx):
        if ctx.MINUS() is None:
            return self.visit(ctx.atom())
        negated = self.visit(ctx.unary())
        # Two minuses in C++ would decrement
        return f'-({negated})' if self.cpp and negated.startswith('-') else f'-{negated}'

    def visitAtom(self, ctx):
        if ctx.INT() is not None:
            # Each an int64 in C++, so literals never compute in int
            return f'int64{{{ctx.INT().getText()}}}' if self.cpp else ctx.INT().getText()
        if ctx.reference() is not None:
            # An attribute or state variable, by its generated name
            return get_attribute_name(ctx.reference().getText()[3:])
        return f'({self.visit(ctx.expression())})'


def source(tree, cpp=False):
    """The Rust, or with cpp the C++, that computes a value"""
    return ExpressionWriter(cpp).visit(tree)


class TypedWriter(ExpressionWriter):
    """A value as Rust, each literal an i64"""

    def visitAtom(self, ctx):
        if ctx.INT() is not None:
            return f'{ctx.INT().getText()}i64'
        return super().visitAtom(ctx)


# The checked call of each arithmetic operator
CHECKED_OPERATORS = {'+': 'checked_add', '-': 'checked_sub', '*': 'checked_mul'}
# The widest operand, so two and what joins them keep a call on its line
OPERAND_WIDTH = (CALL_WIDTH - len(' <= ')) // 2


class CheckedWriter(TypedWriter):
    """A value as Rust lets of a closure, an operator to a let, each operator checked"""

    def __init__(self):
        super().__init__()
        self.lets = []

    def local(self, value):
        """A value in a let of its own, as rustfmt lays it out; return the let's name"""
        name = f'value_{len(self.lets)}'
        line = f'    let {name} = {value};'
        self.lets.append(line if fits(line) else f'    let {name} =\n      {value};')
        return name

    def operation(self, operator, left, right):
        """The checked Rust of one operator on two operands, None past i64"""
        return f'{left}.{CHECKED_OPERATORS[operator]}({right})?'

    def chain(self, ctx, operands):
        # Each operator a let on the value so far
        value = self.visit(operands[0])
        for k, operand in enumerate(operands[1:]):
            operator = ctx.getChild(2 * k + 1).getText()
            value = self.local(self.operation(operator, value, self.visit(operand)))
        return value

    def visitUnary(self, ctx):
        if ctx.MINUS() is None:
            return self.visit(ctx.atom())
        return self.local(f'{self.visit(ctx.unary())}.checked_neg()?')

    def visitAtom(self, ctx):
        # A bracket nests as a let does
        if ctx.expression() is not None:
            return self.visit(ctx.expression())
        operand = super().visitAtom(ctx)
        # A wide attribute in a let of its own
        return self.local(operand) if len(operand) > OPERAND_WIDTH else operand

    def computed(self, name, value):
        """A local of what the lets compute, None where an operator leaves i64"""
        lets = '\n'.join(self.lets)
        return f'  let {name} = computed(|| {{\n{lets}\n    Some({value})\n  }});\n'


def computes(tree):
    """Whether a parse tree holds an arithmetic operator"""
    tokens = (
        IDLV2ExprParser.PLUS,
        IDLV2ExprParser.MINUS,
        IDLV2ExprParser.STAR,
        IDLV2ExprParser.SLASH,
        IDLV2ExprParser.PERCENT,
    )
    return any(Trees.findAllTokenNodes(tree, token) for token in tokens)


def chain(tree):
    """A value's outermost run: its operands' C++ and the operators between them"""
    additive = tree.additive()
    product = additive.multiplicative(0)
    # A product that divides is one call, never broken at its operators
    divides = {part.getText() for part in product.getChildren()} & ROUNDED_CALLS.keys()
    run = additive if len(additive.multiplicative()) > 1 or divides else product
    operands = run.multiplicative() if run is additive else run.unary()
    operators = [run.getChild(2 * k + 1).getText() for k in range(len(operands) - 1)]
    writer = ExpressionWriter(cpp=True)
    return [writer.visit(operand) for operand in operands], operators


def checked_comparison(name, tree):
    """The Rust that leaves in a local whether a comparison holds, None past i64"""
    writer = CheckedWriter()
    left, right = (writer.visit(side) for side in tree.additive())
    return writer.computed(name, f'{left} {tree.compare().getText()} {right}')


# Each comparison's negation, exact over integers
NEGATED = {'<=': '>', '<': '>=', '==': '!=', '>=': '<', '>': '<=', '!=': '=='}


def comparison(tree):
    """The Rust that holds where one comparison of a constraint does not"""
    # Literals alone would compute in i32, so each is i64
    named = Trees.findAllTokenNodes(tree, IDLV2ExprParser.COMPUTATIONAL)
    writer = ExpressionWriter() if named else TypedWriter()
    left, right = tree.additive()
    return f'{writer.visit(left)} {NEGATED[tree.compare().getText()]} {writer.visit(right)}'
