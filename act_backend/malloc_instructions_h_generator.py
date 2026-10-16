"""Generator for instructions.h - templates the generic file with instruction classes"""

import os

from .expression_writer import source
from .instruction_info import to_camel_case
from .template_loader import fill_placeholders, get_backend_template_loader


def generate_get_h_method(index, addr_var, size_expr, templates):
    range_pair = f'{addr_var}, MAKE_SUM({addr_var}, {size_expr})'
    return templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.get_h_method.txt', index=index, ranges=range_pair
    )


def generate_instruction_class(instruction, templates):
    """Generate C++ instruction class from instruction definition"""
    class_name = to_camel_case(instruction.instruction)
    op_name = instruction.instruction
    comp_attrs = instruction.comp_attr
    params = instruction.parameters

    # Generate members
    comp_attr_members = ''
    for attr in comp_attrs:
        comp_attr_members += f'  int64 {attr};\n'

    # A blank line parts the two kinds of member
    addr_var_members = '\n' if comp_attrs else ''
    for param in params:
        addr_var_members += f'  IntVar *{param} = MAKE_INT_VAR(INT64_MIN, INT64_MAX);\n'

    # Generate get_h methods: output first (get_h0), then inputs (get_h1, get_h2, ...)
    get_h_methods = ''
    get_h_cases = ''

    # The output's range first, then each input's
    operands = [*instruction.instr_output_asts, *instruction.instr_input_asts]
    for h_index, (_, starts, sizes) in enumerate(operands):
        addr_var = source(starts[0])
        size_expr = source(sizes[0])

        get_h_methods += generate_get_h_method(h_index, addr_var, size_expr, templates)
        get_h_cases += templates.render_lines(
            'instructions.h.INSTRUCTION_CLASSES.get_h_case.txt', index=h_index
        )

    # Generate constructor parameters
    comp_attr_params = ''
    if comp_attrs:
        comp_attr_params = ', ' + ', '.join(f'int64 {attr}' for attr in comp_attrs)

    # Generate constructor initialization
    comp_attr_init = ''
    if comp_attrs:
        comp_attr_init = ', ' + ', '.join(f'{attr}({attr})' for attr in comp_attrs)

    # The text names each attribute's value, a line each
    values = [(attr, attr) for attr in comp_attrs]
    values += [(param, f'{param}->Value()') for param in params]
    str_pieces = ''
    for index, (attr, value) in enumerate(values):
        str_pieces += templates.render_lines(
            'instructions.h.INSTRUCTION_CLASSES.str_piece.txt',
            label=(', ' if index else '') + f'{attr} = ',
            value=value,
        )

    # Generate get_int_var list
    get_int_var_list = ', '.join(params)

    # Generate complete class using template
    return templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.class.txt',
        CLASS_NAME=class_name,
        OP_NAME=op_name,
        COMP_ATTR_MEMBERS=comp_attr_members,
        ADDR_VAR_MEMBERS=addr_var_members,
        GET_H_METHODS=get_h_methods,
        COMP_ATTR_PARAMS=comp_attr_params,
        COMP_ATTR_INIT=comp_attr_init,
        STR_PIECES=str_pieces,
        GET_H_CASES=get_h_cases,
        GET_INT_VAR_LIST=get_int_var_list,
    )


def generate_instructions_file(backend_gen_dir, instructions):
    """Template instructions.h from generic file"""
    templates = get_backend_template_loader()

    # Read generic template file
    generic_file = os.path.join(backend_gen_dir, 'cpp', 'malloc', 'include', 'instructions.h')
    with open(generic_file, 'r') as f:
        content = f.read()

    # Generate all instruction classes, a blank line between two
    classes = '\n'.join(
        generate_instruction_class(instruction, templates) for instruction in instructions
    ).rstrip('\n')

    # Replace placeholder
    content = fill_placeholders(content, {'INSTRUCTION_CLASSES': classes}, generic_file)

    # Write back
    with open(generic_file, 'w') as f:
        f.write(content)
