import os
import re

import taidl.operations
from antlr4.tree.Tree import TerminalNode
from antlr4.tree.Trees import Trees
from taidl.antlr4 import IDLV2ExprParser

from .expression_writer import source
from .instruction_info import get_attribute_name
from .semantics_visitor import OperandVisitor
from .template_loader import fill_placeholders, fits, get_backend_template_loader

# The e-graph's element types, by their semantics word
ELEMENT_TYPES = {
    'pred': 'Dtype::Pred',
    's8': 'Dtype::I8',
    's16': 'Dtype::I16',
    's32': 'Dtype::I32',
    's64': 'Dtype::I64',
    'u8': 'Dtype::U8',
    'u16': 'Dtype::U16',
    'u32': 'Dtype::U32',
    'u64': 'Dtype::U64',
    'f16': 'Dtype::FP16',
    'bf16': 'Dtype::BF16',
    'f32': 'Dtype::FP32',
    'f64': 'Dtype::FP64',
}

# The operations whose payload holds their attributes' integers, in order
COMPARED = {'concatenate', 'reduce_add', 'slice', 'transpose'}
# The operations whose payload holds none of their attributes
UNCOMPARED = {'broadcast', 'dot'}

# rustfmt's fn_call_width: the widest a call's arguments stay on one line
CALL_WIDTH = 60

# Each matched value's shape and type, by its position
LHS_METADATA = (
    '  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses\n'
    '    .iter()\n'
    '    .map(|id| egraph[*id].data.clone())\n'
    '    .collect();\n'
)


class RustGeneratorVisitor(OperandVisitor):
    def __init__(self, expressions):
        super().__init__()
        # Each backtick expression's parsed tree, by its node
        self.expressions = expressions
        self.assignments = []
        self.root_assignment = None
        self.comp_attributes = set()
        self.variable_definitions = {}

    def visitModule(self, ctx):
        self.visitChildren(ctx)

        # The values in the pattern order a match walks
        self.assignments = self.pattern_order(self.root_assignment['lhs'])
        self.root_assignment = self.assignments[-1]

        # Each later occurrence of a value, beside its first
        firsts = {}
        copies = []
        for position, assignment in enumerate(self.assignments):
            first = firsts.setdefault(assignment['lhs'], position)
            if first != position:
                copies.append((position, first))

        return {
            'assignments': self.assignments,
            'root_assignment': self.root_assignment,
            'comp_attributes': list(self.comp_attributes),
            'copies': copies,
        }

    def visitInstruction(self, ctx):
        lhs_name = ctx.IDENTIFIER().getText()
        lhs_type, lhs_shape, lhs_dims, lhs_trees = self.visit(ctx.result_type())
        attributes = ctx.attributes().attribute() if ctx.attributes() else []

        is_root = ctx.ROOT() is not None

        self.extract_comp_attrs_from_text(lhs_shape)

        operand_names = []
        if ctx.operands():
            operands = self.visit(ctx.operands())
            for operand_text, operand_type in operands:
                if operand_type == 'VALUE':
                    operand_names.append(operand_text)

        assignment = {
            'lhs': lhs_name,
            'op': ctx.OPERATION().getText(),
            # Each attribute's value, by its name
            'attributes': {
                attribute.name().getText(): attribute.attribute_value() for attribute in attributes
            },
            'dtype': lhs_type,
            'shape': lhs_shape,
            'dims': lhs_dims,
            'trees': lhs_trees,
            'operands': operand_names,
        }

        self.variable_definitions[lhs_name] = len(self.assignments)
        self.assignments.append(assignment)

        if is_root:
            self.root_assignment = assignment

        return assignment

    def pattern_order(self, name):
        """A value after the values it reads, left to right, one entry per occurrence"""
        assignment = self.assignments[self.variable_definitions[name]]
        ordered = []
        for operand in assignment['operands']:
            ordered += self.pattern_order(operand)
        return [*ordered, assignment]

    def extract_comp_attrs_from_text(self, text):
        pattern = r'@c\.(\w+)'
        matches = re.findall(pattern, str(text))
        self.comp_attributes.update(matches)

    def visitResult_type(self, ctx):
        # A rank-0 result is one element, like a kernel's
        if ctx.shape() is None:
            return (ctx.TYPE().getText(), '', ['1'], [None])
        # The type, the written shape, and each dimension in Rust
        dims = [self.visit(dim) for dim in ctx.shape().dim()]
        # Each dimension's tree, where it is an expression
        trees = [self.expressions.get(dim.EXPRESSION()) for dim in ctx.shape().dim()]
        return (ctx.TYPE().getText(), ctx.shape().getText(), dims, trees)

    def visitDim(self, ctx):
        if ctx.INT():
            return ctx.INT().getText()
        return source(self.expressions[ctx.EXPRESSION()])


def slice_ranges(ranges):
    """Return each range of a slice list as its values: start, limit and any stride"""
    groups = []
    for child in ranges.getChildren():
        if isinstance(child, TerminalNode) and child.getText() == '[':
            groups.append([])
        elif not isinstance(child, TerminalNode):
            groups[-1].append(child)
    return groups


def attribute_factor(dimension, attribute):
    """Return the positive integer a dimension multiplies an attribute by, 1 where it stands alone

    None for a dimension that holds the attribute in another form, or not at all.
    """
    if dimension is None:
        return None
    # The dimension's tokens, a bracket around a term left out
    leaves = [
        node.getText() for node in Trees.descendants(dimension) if isinstance(node, TerminalNode)
    ]
    tokens = [token for token in leaves if token not in ('(', ')')]
    reference = f'@c.{attribute}'
    if tokens == [reference]:
        return 1
    if len(tokens) != 3 or tokens[1] != '*' or reference not in (tokens[0], tokens[2]):
        return None
    # The integer on either side, never zero, the rewrite's divisor
    factor = tokens[2] if tokens[0] == reference else tokens[0]
    if not factor.isdigit():
        return None
    return int(factor) or None


def attribute_reading(assignments, attribute):
    """Return where a rewrite reads an attribute: a value's position, its dimension and the factor

    The first dimension in pattern order that holds the attribute alone or times a positive integer,
    or None where none does.
    """
    for position, assignment in enumerate(assignments):
        for dimension, tree in enumerate(assignment['trees']):
            factor = attribute_factor(tree, attribute)
            if factor:
                return position, dimension, factor
    return None


def payload_reading(assignments, attribute, expressions):
    """Return where a rewrite reads an attribute from a payload: a position, an index and the factor

    The first compared value in pattern order that holds the attribute alone or times a positive
    integer, or None where none does.
    """
    for position, assignment in enumerate(assignments):
        if assignment['op'] not in COMPARED:
            continue
        for index, value in enumerate(compared_values(assignment)):
            factor = attribute_factor(expressions.get(value.EXPRESSION()), attribute)
            if factor:
                return position, index, factor
    return None


def generate_comp_attr_extraction(assignments, attributes, checked, expressions):
    """The Rust that reads each attribute, the precondition's first checking the shape holds it

    An attribute no shape names is read from the payload of a matched e-node.
    """
    code = ''
    for attribute in attributes:
        name = get_attribute_name(attribute)
        if attribute_reading(assignments, attribute) is None:
            position, index, factor = payload_reading(assignments, attribute, expressions)
            read = f'TensorOp::payload_int(&lhs_enodes[{position}], {index})'
            division = f' / {factor}' if factor != 1 else ''
            if not checked:
                code += let_line(name, f'{read}.unwrap(){division}')
                continue
            # Read into `value`, so a long name never widens the let-else
            code += f'  let Some(value) = {read} else {{\n    return false;\n  }};\n'
            code += let_line(name, f'value{division}')
            continue
        position, dimension, factor = attribute_reading(assignments, attribute)
        shape = f'lhs_metadata[{position}].shape'
        if checked and dimension:
            # A matched value may have fewer dimensions
            code += f'  if {shape}.len() <= {dimension} {{\n    return false;\n  }}\n'
        division = f' / {factor}' if factor != 1 else ''
        code += f'  let {name} = {shape}[{dimension}]{division};\n'
    return code


def compared_values(assignment):
    """Return the values a compared operation's attributes give, in its payload's order"""
    values = []
    for name in taidl.operations.OPERATIONS[assignment['op']][1]:
        given = assignment['attributes'][name]
        if given.slice_list():
            # Each range's start and limit
            values += [bound for bounds in slice_ranges(given.slice_list()) for bound in bounds[:2]]
        else:
            values += given.brace_list().value()
    return values


def let_line(name, value):
    """A let as rustfmt lays it out, the value on a line of its own where the let is too wide"""
    line = f'  let {name} = {value};'
    if fits(line):
        return f'{line}\n'
    return f'  let {name} =\n    {value};\n'


def value_lets(name, tree):
    """The lets that leave a value in a local, one too wide for a line built an operand at a time"""
    text = source(tree)
    if fits(f'    {text};'):
        return let_line(name, text)
    # A bracket or a lone operand stands for what it holds, a negation negated after
    if isinstance(tree, IDLV2ExprParser.ExpressionContext):
        return value_lets(name, tree.additive())
    if isinstance(tree, IDLV2ExprParser.AtomContext):
        if tree.expression() is None:
            return let_line(name, text)
        return value_lets(name, tree.expression())
    if isinstance(tree, IDLV2ExprParser.UnaryContext):
        if tree.MINUS() is None:
            return value_lets(name, tree.atom())
        return value_lets(name, tree.unary()) + let_line(name, f'-{name}')
    # A sum or a product: its first operand, then each with its operator
    parts = list(tree.getChildren())
    lines = value_lets(name, parts[0])
    for position, (operator, operand) in enumerate(zip(parts[1::2], parts[2::2]), 1):
        step = f'{name} {operator.getText()} {source(operand)}'
        if not fits(f'    {step};'):
            # A wide operand in a local of its own
            lines += value_lets(f'{name}_{position}', operand)
            step = f'{name} {operator.getText()} {name}_{position}'
        lines += let_line(name, step)
    return lines


def generate_attr_checks(assignments, expressions, templates):
    """The Rust that compares each compared operation's attributes with its matched e-node"""
    checks = ''
    for index, assignment in enumerate(assignments):
        if assignment['op'] not in COMPARED:
            continue
        values = compared_values(assignment)
        # Each value an integer, or an expression over the attributes
        ints = [
            source(expressions[value.EXPRESSION()]) if value.EXPRESSION() else value.getText()
            for value in values
        ]
        if len(f'&lhs_enodes[{index}], &[{", ".join(ints)}]') <= CALL_WIDTH:
            checks += templates.render_lines(
                'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.attr_check.txt',
                index=str(index),
                ints=', '.join(ints),
            )
            continue
        # Too many for one line: the count, then each integer
        checks += templates.render_lines(
            'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.attr_check_len.txt',
            index=str(index),
            count=str(len(ints)),
        )
        for position, (value, text) in enumerate(zip(values, ints)):
            check = templates.render_lines(
                'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.attr_check_int.txt',
                index=str(index),
                position=str(position),
                value=text,
            )
            if not fits(check):
                # A value too wide for its check, computed first
                checks += value_lets('value', expressions[value.EXPRESSION()])
                check = templates.render_lines(
                    'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.attr_check_int.txt',
                    index=str(index),
                    position=str(position),
                    value='value',
                )
            checks += check
    return checks


def generate_precond_function(instruction_name, semantics_data, attributes, expressions, templates):
    assignments = semantics_data['assignments']

    num_assignments = len(assignments)

    # Generate comp attribute extraction
    comp_attr_code = generate_comp_attr_extraction(assignments, attributes, True, expressions)

    # Each copy of a value is its first occurrence's e-class
    shape_checks = [
        templates.render(
            'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.copy_check.txt',
            copy=str(copy),
            first=str(first),
        )
        for copy, first in semantics_data['copies']
    ]

    # Generate shape/dtype checks for all assignments
    for i, assignment in enumerate(assignments):
        dtype_rust = ELEMENT_TYPES[assignment['dtype']]
        shape_vec = f'vec![{", ".join(assignment["dims"])}]'

        shape_check = templates.render(
            'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.shape_check.txt',
            index=str(i),
            shape_vec=shape_vec,
            dtype=dtype_rust,
        )
        shape_checks.append(shape_check)

    shape_checks_code = '\n'.join(shape_checks)

    # Attributes are compared once the shapes hold
    attr_checks = generate_attr_checks(assignments, expressions, templates)

    return templates.render_lines(
        'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.precond.txt',
        instruction_name=instruction_name,
        enodes='lhs_enodes' if attr_checks else '_',
        num_assignments=str(num_assignments),
        comp_attr_code=comp_attr_code,
        shape_checks=shape_checks_code,
        attr_checks=attr_checks,
    )


def generate_metadata_function(
    instruction_name, semantics_data, attributes, expressions, rhs_size, templates
):
    assignments = semantics_data['assignments']

    num_assignments = len(assignments)

    if attributes:
        # The attributes' values, as the precondition reads them
        shaped = [
            attribute for attribute in attributes if attribute_reading(assignments, attribute)
        ]
        comp_attr_extraction = generate_comp_attr_extraction(
            assignments, attributes, False, expressions
        )

        # The attributes as one string, in declared order
        names = [get_attribute_name(attribute) for attribute in attributes]
        if len(names) == 1:
            value = f'{names[0]}.to_string()'
        else:
            pieces = ','.join(['{}'] * len(names))
            value = f'format!("{pieces}", {", ".join(names)})'

        return templates.render_lines(
            'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.metadata_with_attr.txt',
            instruction_name=instruction_name,
            egraph='egraph' if shaped else '_',
            enodes='_' if len(shaped) == len(attributes) else 'lhs_enodes',
            num_assignments=str(num_assignments),
            attributes=value,
            comp_attr_extraction=(LHS_METADATA if shaped else '') + comp_attr_extraction,
            rhs_size=str(rhs_size),
        )
    else:
        return templates.render_lines(
            'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.metadata_no_attr.txt',
            instruction_name=instruction_name,
            num_assignments=str(num_assignments),
            rhs_size=str(rhs_size),
        )


def generate_set_shapes_function(
    instruction_name, semantics_data, attributes, expressions, rhs_size, templates
):
    assignments = semantics_data['assignments']
    root_assignment = semantics_data['root_assignment']

    num_assignments = len(assignments)

    # The value of each attribute a shape holds, as the precondition reads it
    shaped = [attribute for attribute in attributes if attribute_reading(assignments, attribute)]
    metadata_code = ''
    if shaped:
        metadata_code = LHS_METADATA + generate_comp_attr_extraction(
            assignments, shaped, False, expressions
        )

    # Generate shape setting
    dtype_rust = ELEMENT_TYPES[root_assignment['dtype']]
    shape_vec = f'vec![{", ".join(root_assignment["dims"])}]'

    shape_setting = templates.render_lines(
        'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.shape_setting.txt',
        shape_vec=shape_vec,
        dtype=dtype_rust,
    )

    return templates.render_lines(
        'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.set_shapes.txt',
        instruction_name=instruction_name,
        num_assignments=str(num_assignments),
        rhs_size=str(rhs_size),
        metadata_code=metadata_code,
        shape_setting=shape_setting,
    )


def generate_ir2isa_rust_functions(metadata, templates):
    visitor = RustGeneratorVisitor(metadata.expressions)
    semantics_data = visitor.visit(metadata.semantics_ast)

    attributes = metadata.attributes
    precond = generate_precond_function(
        metadata.name, semantics_data, attributes, metadata.expressions, templates
    )
    metadata_func = generate_metadata_function(
        metadata.name,
        semantics_data,
        attributes,
        metadata.expressions,
        metadata.rhs_size,
        templates,
    )
    set_shapes = generate_set_shapes_function(
        metadata.name,
        semantics_data,
        attributes,
        metadata.expressions,
        metadata.rhs_size,
        templates,
    )

    return f'{precond}\n{metadata_func}\n{set_shapes}'


def generate_ir2isa_rewrites_rs_file(backend_gen_dir: str, instruction_metadata_list):
    """Template ir2isa_rewrites.rs from generic file"""
    templates = get_backend_template_loader()

    # Read generic file
    rust_file = os.path.join(backend_gen_dir, 'src', 'isel', 'rewrites', 'ir2isa_rewrites.rs')
    with open(rust_file, 'r') as f:
        content = f.read()

    # Generate all rewrite functions
    functions = []
    for metadata in instruction_metadata_list:
        func_code = generate_ir2isa_rust_functions(metadata, templates)
        functions.append(func_code)

    # Functions one blank line apart, none at the end
    all_functions = '\n'.join(functions).rstrip('\n')

    # Replace placeholder
    content = fill_placeholders(content, {'IR2ISA_REWRITE_FUNCTIONS': all_functions}, rust_file)

    # Write back
    with open(rust_file, 'w') as f:
        f.write(content)
