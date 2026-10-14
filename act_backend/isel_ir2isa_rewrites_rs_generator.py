import os
import re

from antlr4.tree.Tree import TerminalNode
from antlr4.tree.Trees import Trees

from .expression_writer import source
from .semantics_visitor import OperandVisitor
from .template_loader import fill_placeholders, get_backend_template_loader

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


def generate_comp_attr_extraction(assignments, attributes, checked):
    """The Rust that reads each attribute, the precondition first checking its dimension exists"""
    code = ''
    for attribute in attributes:
        position, dimension, factor = attribute_reading(assignments, attribute)
        shape = f'lhs_metadata[{position}].shape'
        if checked and dimension:
            # A matched value may have fewer dimensions
            code += f'  if {shape}.len() <= {dimension} {{\n    return false;\n  }}\n'
        division = f' / {factor}' if factor != 1 else ''
        code += f'  let {attribute} = {shape}[{dimension}]{division};\n'
    return code


def generate_precond_function(instruction_name, semantics_data, attributes, templates):
    assignments = semantics_data['assignments']

    num_assignments = len(assignments)

    # Generate comp attribute extraction
    comp_attr_code = generate_comp_attr_extraction(assignments, attributes, True)

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

    return templates.render_lines(
        'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.precond.txt',
        instruction_name=instruction_name,
        num_assignments=str(num_assignments),
        comp_attr_code=comp_attr_code,
        shape_checks=shape_checks_code,
    )


def generate_metadata_function(instruction_name, semantics_data, attributes, rhs_size, templates):
    assignments = semantics_data['assignments']

    num_assignments = len(assignments)

    if attributes:
        # The attributes' values, as the precondition reads them
        comp_attr_extraction = generate_comp_attr_extraction(assignments, attributes, False)

        # The attributes as one string, in declared order
        if len(attributes) == 1:
            value = f'{attributes[0]}.to_string()'
        else:
            pieces = ','.join(['{}'] * len(attributes))
            value = f'format!("{pieces}", {", ".join(attributes)})'

        return templates.render_lines(
            'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.metadata_with_attr.txt',
            instruction_name=instruction_name,
            num_assignments=str(num_assignments),
            attributes=value,
            comp_attr_extraction=comp_attr_extraction.strip(),
            rhs_size=str(rhs_size),
        )
    else:
        return templates.render_lines(
            'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.metadata_no_attr.txt',
            instruction_name=instruction_name,
            num_assignments=str(num_assignments),
            rhs_size=str(rhs_size),
        )


def generate_set_shapes_function(instruction_name, semantics_data, attributes, rhs_size, templates):
    assignments = semantics_data['assignments']
    root_assignment = semantics_data['root_assignment']

    num_assignments = len(assignments)

    # The attributes' values, as the precondition reads them
    metadata_code = ''
    if attributes:
        metadata_code = (
            '  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses\n'
            '    .iter()\n'
            '    .map(|id| egraph[*id].data.clone())\n'
            '    .collect();\n' + generate_comp_attr_extraction(assignments, attributes, False)
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
    precond = generate_precond_function(metadata.name, semantics_data, attributes, templates)
    metadata_func = generate_metadata_function(
        metadata.name, semantics_data, attributes, metadata.rhs_size, templates
    )
    set_shapes = generate_set_shapes_function(
        metadata.name, semantics_data, attributes, metadata.rhs_size, templates
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
