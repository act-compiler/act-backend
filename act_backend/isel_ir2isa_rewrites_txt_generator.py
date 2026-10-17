import os
import re

from taidl.antlr4 import IDLV2Parser

from .instruction_info import get_rewrite_word
from .semantics_visitor import OperandVisitor
from .template_loader import fill_placeholders, get_backend_template_loader

# Each e-graph operation: its rewrite word and value count
OPERATIONS = {
    'add': ('add', 2),
    'bitcast_convert': ('bitcvt', 1),
    'broadcast': ('broadcast_?', 1),
    'concatenate': ('concat_?', 2),
    'constant': ('constant_?', 0),
    'convert': ('convert_?', 1),
    'copy': ('copy', 1),
    'divide': ('divide', 2),
    'dot': ('dot', 2),
    'exponential': ('exponential', 1),
    'or': ('or', 2),
    'reduce_add': ('reduce_?', 1),
    'reshape': ('reshape_?', 1),
    'shift_left': ('shift-left', 2),
    'shift_right_logical': ('shift-right-logical', 2),
    'slice': ('slice_?', 1),
    'transpose': ('transpose_?', 1),
    'xor': ('xor', 2),
}

# A pattern variable, after a start, space or bracket
VARIABLE = re.compile(r'(?<![^\s(])\?[^\s()]+')
# A variable, or an operation that takes no value
LEAF = re.compile(rf'{VARIABLE.pattern}|\([^\s()]+\)')


def repeated_leaf(pattern):
    """Return a leaf a rewrite's left side holds twice, or None"""
    leaves = LEAF.findall(pattern)
    return next((leaf for leaf in leaves if leaves.count(leaf) > 1), None)


class RewriteRuleVisitor(OperandVisitor):
    def __init__(self):
        super().__init__()
        self.variable_counter = 0
        self.parameter_variables = {}
        self.variable_definitions = {}
        self.root_variable = None
        self.used_values = set()

    def get_next_variable(self):
        var_name = f'?{chr(ord("a") + self.variable_counter)}'
        self.variable_counter += 1
        return var_name

    def substitute_variables(self, pattern, substitutions):
        result = pattern
        for old_var, new_var in substitutions.items():
            result = result.replace(old_var, new_var)
        return result

    def visitModule(self, ctx: IDLV2Parser.ModuleContext):
        self.visitChildren(ctx)
        # The pattern of the one line that carries ROOT
        return self.variable_definitions[self.root_variable]

    def get_parameter_variables(self):
        sorted_params = sorted(self.parameter_variables.items(), key=lambda x: x[0])
        return [var for param_num, var in sorted_params]

    def visitInstruction(self, ctx: IDLV2Parser.InstructionContext):
        lhs_name = ctx.IDENTIFIER().getText()
        op_name = ctx.OPERATION().getText()

        is_root = ctx.ROOT() is not None
        if is_root:
            self.root_variable = lhs_name

        if op_name == 'parameter':
            operand_info = self.visit(ctx.operands())
            param_num = int(operand_info[0][0])
            var = self.get_next_variable()
            self.parameter_variables[param_num] = var
            self.variable_definitions[lhs_name] = var
            return var

        operand_patterns = []
        if ctx.operands():
            operand_info = self.visit(ctx.operands())
            for operand_value, operand_type in operand_info:
                if operand_type == 'VALUE':
                    pattern = self.variable_definitions[operand_value]

                    if operand_value in self.used_values:
                        # A copy shares no variable with what it copies
                        substitutions = {
                            var: self.get_next_variable()
                            for var in dict.fromkeys(VARIABLE.findall(pattern))
                        }
                        pattern = self.substitute_variables(pattern, substitutions)
                    else:
                        self.used_values.add(operand_value)

                    operand_patterns.append(pattern)

        rewrite_op_name, _ = OPERATIONS[op_name]
        pattern = (
            f'({rewrite_op_name} {" ".join(operand_patterns)})'
            if operand_patterns
            else f'({rewrite_op_name})'
        )

        self.variable_definitions[lhs_name] = pattern

        return pattern


def generate_rewrite_rule(metadata, templates):
    """Generate one rewrite from its template"""
    visitor = RewriteRuleVisitor()
    lhs_pattern = visitor.visit(metadata.semantics_ast)
    parameter_vars = visitor.get_parameter_variables()
    rewrite_word = get_rewrite_word(metadata.name)

    # Generate RHS with or without _? suffix
    rhs = (
        f'({rewrite_word}_? {" ".join(parameter_vars)})'
        if metadata.attributes
        else f'({rewrite_word} {" ".join(parameter_vars)})'
    )

    # Calculate padding to align the '=>' at column 20
    padding = ' ' * (19 - len(rewrite_word))

    return templates.render(
        'ir2isa_rewrites.txt.IR2ISA_REWRITE_RULES.rule.txt',
        rewrite_word=rewrite_word,
        padding=padding,
        lhs_pattern=lhs_pattern,
        rhs=rhs,
    )


def generate_ir2isa_rewrites_txt_file(backend_gen_dir: str, instruction_metadata_list):
    """Template ir2isa_rewrites.txt from generic file"""
    templates = get_backend_template_loader()

    # Read generic file
    rewrites_file = os.path.join(backend_gen_dir, 'src', 'isel', 'rewrites', 'ir2isa_rewrites.txt')
    with open(rewrites_file, 'r') as f:
        content = f.read()

    # Generate every rewrite
    rules = []
    for metadata in instruction_metadata_list:
        rule = generate_rewrite_rule(metadata, templates)
        rules.append(rule)

    # Join rules with newlines
    all_rules = '\n'.join(rules)

    # Replace placeholder
    content = fill_placeholders(content, {'IR2ISA_REWRITE_RULES': all_rules}, rewrites_file)

    # Write back
    with open(rewrites_file, 'w') as f:
        f.write(content)
