"""The visitor methods every reader of an instruction's semantics shares"""

from taidl.antlr4 import IDLV2Visitor


class OperandVisitor(IDLV2Visitor):
    def visitOperands(self, ctx):
        operand_list = []
        for operand_ctx in ctx.operand():
            operand_info = self.visit(operand_ctx)
            operand_list.append(operand_info)
        return operand_list

    def visitOperand(self, ctx):
        value_ctx = ctx.value()
        value_text = value_ctx.getText()

        if value_ctx.INT():
            operand_type = 'INT'
        elif value_ctx.IDENTIFIER():
            operand_type = 'IDENTIFIER'
        else:
            operand_type = 'EXPRESSION'

        return (value_text, operand_type)
