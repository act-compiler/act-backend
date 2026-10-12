"""Rust and C++ for a value taidl has parsed, each attribute by its own name"""

from taidl.antlr4 import IDLV2ExprVisitor


class ExpressionWriter(IDLV2ExprVisitor):
    """A value as one expression, its operators spaced and its brackets the text's own"""

    def visitExpression(self, ctx):
        return self.visit(ctx.additive())

    def chain(self, ctx, operands):
        """A run of operands, the operators between them"""
        source = self.visit(operands[0])
        for k, operand in enumerate(operands[1:]):
            source += f' {ctx.getChild(2 * k + 1).getText()} {self.visit(operand)}'
        return source

    def visitAdditive(self, ctx):
        return self.chain(ctx, ctx.multiplicative())

    def visitMultiplicative(self, ctx):
        return self.chain(ctx, ctx.unary())

    def visitUnary(self, ctx):
        if ctx.MINUS() is None:
            return self.visit(ctx.atom())
        return f'-{self.visit(ctx.unary())}'

    def visitAtom(self, ctx):
        if ctx.INT() is not None:
            return ctx.INT().getText()
        if ctx.reference() is not None:
            # An attribute or state variable, its prefix dropped
            return ctx.reference().getText()[3:]
        return f'({self.visit(ctx.expression())})'


def source(tree):
    """The Rust or C++ that computes a value"""
    return ExpressionWriter().visit(tree)
