import os
from taidl.antlr4 import IDLV2Parser, IDLV2Visitor
from .instruction_info import get_kebab_name
from .template_loader import fill_placeholders, get_backend_template_loader


class RewriteRuleVisitor(IDLV2Visitor):
    OP_MAP = {
        'reshape': 'reshape_?',
        'convert': 'convert_?',
        'slice': 'slice_?',
        'transpose': 'transpose_?',
        'broadcast': 'broadcast_?',
        'concat': 'concat_?',
        'constant': 'constant_?',
        'bitcast_convert': 'bitcvt',
        'reduce_add': 'reduce_?',
        'shift_left': 'shift-left',
        'shift_right_logical': 'shift-right-logical',
    }

    def __init__(self):
        super().__init__()
        self.pattern_stack = []
        self.variable_counter = 0
        self.parameter_variables = {}
        self.variable_definitions = {}
        self.root_variable = None
        self.used_values = set()

    def get_next_variable(self):
        var_name = f'?{chr(ord("a") + self.variable_counter)}'
        self.variable_counter += 1
        return var_name

    def substitute_params(self, pattern, substitutions):
        result = pattern
        for old_var, new_var in substitutions.items():
            result = result.replace(old_var, new_var)
        return result

    def visitModule(self, ctx: IDLV2Parser.ModuleContext):
        self.visitChildren(ctx)
        if self.root_variable and self.root_variable in self.variable_definitions:
            return self.variable_definitions[self.root_variable]
        elif self.pattern_stack:
            return self.pattern_stack[-1]
        return None

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
            if ctx.operands():
                operand_info = self.visit(ctx.operands())
                if operand_info:
                    param_num = int(operand_info[0][0])
                    var = self.get_next_variable()
                    self.parameter_variables[param_num] = var
                    self.variable_definitions[lhs_name] = var
                    return var

        operand_patterns = []
        if ctx.operands():
            operand_info = self.visit(ctx.operands())
            for operand_value, operand_type in operand_info:
                if operand_type == 'IDENTIFIER':
                    if operand_value in self.variable_definitions:
                        pattern = self.variable_definitions[operand_value]

                        if operand_value in self.used_values:
                            substitutions = {}
                            for param_var in self.parameter_variables.values():
                                if param_var in pattern:
                                    fresh_var = self.get_next_variable()
                                    substitutions[param_var] = fresh_var
                            if substitutions:
                                pattern = self.substitute_params(pattern, substitutions)
                        else:
                            self.used_values.add(operand_value)

                        operand_patterns.append(pattern)
                    else:
                        var = self.get_next_variable()
                        operand_patterns.append(var)

        rewrite_op_name = self.OP_MAP.get(op_name, op_name)
        pattern = (
            f'({rewrite_op_name} {" ".join(operand_patterns)})'
            if operand_patterns
            else f'({rewrite_op_name})'
        )

        self.variable_definitions[lhs_name] = pattern
        self.pattern_stack.append(pattern)

        return pattern

    def visitOperands(self, ctx: IDLV2Parser.OperandsContext):
        operand_list = []
        for operand_ctx in ctx.operand():
            operand_info = self.visit(operand_ctx)
            operand_list.append(operand_info)
        return operand_list

    def visitOperand(self, ctx: IDLV2Parser.OperandContext):
        value_ctx = ctx.value()
        value_text = value_ctx.getText()

        if value_ctx.INT():
            operand_type = 'INT'
        elif value_ctx.IDENTIFIER():
            operand_type = 'IDENTIFIER'
        else:
            operand_type = 'EXPRESSION'

        return (value_text, operand_type)


def generate_rewrite_rule(metadata, templates):
    """Generate a single rewrite rule using template"""
    visitor = RewriteRuleVisitor()
    lhs_pattern = visitor.visit(metadata.semantics_ast)
    parameter_vars = visitor.get_parameter_variables()
    kebab_name = get_kebab_name(metadata.name)

    # No pattern or no parameter: a rule over a lone variable
    if not (lhs_pattern and parameter_vars):
        # Align the '=>' at column 20, as for every rule
        padding = ' ' * (19 - len(kebab_name))
        return templates.render(
            'ir2isa_rewrites.txt.IR2ISA_REWRITE_RULES.rule.txt',
            kebab_name=kebab_name,
            padding=padding,
            lhs_pattern='(?fallback)',
            rhs=f'({kebab_name} ?args)',
        )

    # Generate RHS with or without _? suffix
    rhs = (
        f'({kebab_name}_? {" ".join(parameter_vars)})'
        if metadata.has_comp_attrs
        else f'({kebab_name} {" ".join(parameter_vars)})'
    )

    # Calculate padding to align the '=>' at column 20
    padding = ' ' * (19 - len(kebab_name))

    return templates.render(
        'ir2isa_rewrites.txt.IR2ISA_REWRITE_RULES.rule.txt',
        kebab_name=kebab_name,
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

    # Generate all rewrite rules
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
