import os
import re

from .expression_writer import source
from .semantics_visitor import OperandVisitor
from .template_loader import fill_placeholders, get_backend_template_loader


class RustGeneratorVisitor(OperandVisitor):
    def __init__(self, expressions):
        super().__init__()
        # Each backtick expression's parsed tree, by its node
        self.expressions = expressions
        self.assignments = []
        self.root_assignment = None
        self.comp_attributes = set()
        self.variable_definitions = {}
        self.variable_usage = set()

    def visitModule(self, ctx):
        self.visitChildren(ctx)

        duplicates = [a for a in self.assignments if a['lhs'].endswith('_dup')]
        non_duplicates = [a for a in self.assignments if not a['lhs'].endswith('_dup')]
        self.assignments = duplicates + non_duplicates

        for assignment in self.assignments:
            if assignment['lhs'] == self.root_assignment['lhs']:
                self.root_assignment = assignment
                break

        return {
            'assignments': self.assignments,
            'root_assignment': self.root_assignment,
            'comp_attributes': list(self.comp_attributes),
        }

    def visitInstruction(self, ctx):
        lhs_name = ctx.IDENTIFIER().getText()
        lhs_type, lhs_shape, lhs_dims = self.visit(ctx.result_type())

        is_root = ctx.ROOT() is not None

        self.extract_comp_attrs_from_text(lhs_shape)

        operand_names = []
        if ctx.operands():
            operands = self.visit(ctx.operands())
            for operand_text, operand_type in operands:
                self.extract_comp_attrs_from_text(operand_text)
                if operand_type == 'IDENTIFIER':
                    operand_names.append(operand_text)

        for operand_text in operand_names:
            if operand_text in self.variable_definitions:
                if operand_text in self.variable_usage:
                    self.duplicate_variable_recursive(operand_text)
                else:
                    self.variable_usage.add(operand_text)

        assignment = {
            'lhs': lhs_name,
            'dtype': lhs_type,
            'shape': lhs_shape,
            'dims': lhs_dims,
            'operands': operand_names,
        }

        self.variable_definitions[lhs_name] = len(self.assignments)
        self.assignments.append(assignment)

        if is_root:
            self.root_assignment = assignment

        return assignment

    def duplicate_variable_recursive(self, var_name):
        dup_name = f'{var_name}_dup'

        if dup_name in self.variable_definitions:
            return

        original = self.assignments[self.variable_definitions[var_name]]

        for operand in original['operands']:
            if operand in self.variable_definitions:
                self.duplicate_variable_recursive(operand)

        duplicate = {
            'lhs': dup_name,
            'dtype': original['dtype'],
            'shape': original['shape'],
            'dims': original['dims'],
            'operands': original['operands'],
        }
        self.variable_definitions[dup_name] = len(self.assignments)
        self.assignments.append(duplicate)

    def extract_comp_attrs_from_text(self, text):
        pattern = r'@c\.(\w+)'
        matches = re.findall(pattern, str(text))
        self.comp_attributes.update(matches)

    def visitResult_type(self, ctx):
        # A rank-0 result is one element, like a kernel's
        if ctx.shape() is None:
            return (ctx.TYPE().getText(), '', ['1'])
        # The type, the written shape, and each dimension in Rust
        dims = [self.visit(dim) for dim in ctx.shape().dim()]
        return (ctx.TYPE().getText(), ctx.shape().getText(), dims)

    def visitDim(self, ctx):
        if ctx.INT():
            return ctx.INT().getText()
        return source(self.expressions[ctx.EXPRESSION()])


def map_dtype_to_rust(dtype):
    dtype_map = {
        's8': 'Dtype::I8',
        'u8': 'Dtype::U8',
        's32': 'Dtype::I32',
        'u32': 'Dtype::U32',
        'bf16': 'Dtype::BF16',
    }

    rust_type = dtype_map.get(dtype)
    assert rust_type != None
    return rust_type


def extract_comp_attr_from_shape(shape_text, comp_attrs):
    # Find the computational attribute that appears in the shape
    for attr in comp_attrs:
        if f'@c.{attr}' in shape_text:
            return attr
    return comp_attrs[0] if comp_attrs else None


def root_comp_attr(semantics_data):
    # The attribute the root's shape names, if the semantics has one
    root_assignment = semantics_data['root_assignment']
    comp_attrs = semantics_data['comp_attributes']
    if comp_attrs:
        return extract_comp_attr_from_shape(root_assignment['shape'], comp_attrs)
    return None


def parse_comp_attr_expression(shape_expr, comp_attr):
    # Only works with @c.attr or @c.attr * constant
    # Very fragile but should work
    if not comp_attr or f'@c.{comp_attr}' not in shape_expr:
        return comp_attr, []

    expr = shape_expr.replace('`', '').replace(' ', '')
    pattern = f'@c.{comp_attr}'

    if f'{pattern}*' in expr:
        start = expr.index(pattern) + len(pattern) + 1
        factor_str = ''
        for char in expr[start:]:
            if char.isdigit():
                factor_str += char
            else:
                break
        return comp_attr, [int(factor_str)]

    return comp_attr, []


def generate_comp_attr_extraction(root_assignment, comp_attr, num_assignments):
    # Generate computational attribute extraction code from ROOT assignment
    if not comp_attr:
        return ''

    comp_attr, factors = parse_comp_attr_expression(root_assignment['shape'], comp_attr)
    root_index = num_assignments - 1  # ROOT is always the last assignment

    if factors:
        # Calculate division factor as product of all factors
        division_factor = 1
        for factor in factors:
            division_factor *= factor
        return f'  let {comp_attr} = lhs_metadata[{root_index}].shape[0] / {division_factor};\n'
    else:
        return f'  let {comp_attr} = lhs_metadata[{root_index}].shape[0];\n'


def generate_precond_function(instruction_name, semantics_data, templates):
    assignments = semantics_data['assignments']
    root_assignment = semantics_data['root_assignment']

    num_assignments = len(assignments)
    comp_attr = root_comp_attr(semantics_data)

    # Generate comp attribute extraction
    comp_attr_code = generate_comp_attr_extraction(root_assignment, comp_attr, num_assignments)

    # Generate shape/dtype checks for all assignments
    shape_checks = []
    for i, assignment in enumerate(assignments):
        dtype_rust = map_dtype_to_rust(assignment['dtype'])
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


def generate_metadata_function(instruction_name, semantics_data, rhs_size, templates):
    assignments = semantics_data['assignments']
    root_assignment = semantics_data['root_assignment']

    num_assignments = len(assignments)
    comp_attr = root_comp_attr(semantics_data)

    if comp_attr:
        # Use unified comp_attr extraction logic - same as precond function
        comp_attr_extraction = generate_comp_attr_extraction(
            root_assignment, comp_attr, num_assignments
        )

        return templates.render_lines(
            'ir2isa_rewrites.rs.IR2ISA_REWRITE_FUNCTIONS.metadata_with_attr.txt',
            instruction_name=instruction_name,
            num_assignments=str(num_assignments),
            comp_attr=comp_attr,
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


def generate_set_shapes_function(instruction_name, semantics_data, rhs_size, templates):
    assignments = semantics_data['assignments']
    root_assignment = semantics_data['root_assignment']

    num_assignments = len(assignments)
    comp_attr = root_comp_attr(semantics_data)

    # Generate metadata extraction - use unified logic same as other functions
    metadata_code = ''
    if comp_attr:
        metadata_code = (
            '  let lhs_metadata: Vec<TensorInfo> = lhs_eclasses\n'
            '    .iter()\n'
            '    .map(|id| egraph[*id].data.clone())\n'
            '    .collect();\n'
            + generate_comp_attr_extraction(root_assignment, comp_attr, num_assignments)
        )

    # Generate shape setting
    dtype_rust = map_dtype_to_rust(root_assignment['dtype'])
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

    precond = generate_precond_function(metadata.name, semantics_data, templates)
    metadata_func = generate_metadata_function(
        metadata.name, semantics_data, metadata.rhs_size, templates
    )
    set_shapes = generate_set_shapes_function(
        metadata.name, semantics_data, metadata.rhs_size, templates
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

    # Join functions with one blank line between, none at the end
    all_functions = '\n'.join(functions).rstrip('\n')

    # Replace placeholder
    content = fill_placeholders(content, {'IR2ISA_REWRITE_FUNCTIONS': all_functions}, rust_file)

    # Write back
    with open(rust_file, 'w') as f:
        f.write(content)
