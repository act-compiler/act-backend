"""Generator for instructions.h - templates the generic file with instruction classes"""

import os

from .expression_writer import chain, source
from .instruction_info import get_attribute_name, get_type_names
from .template_loader import call_lines, fill_placeholders, fits, get_backend_template_loader


def generate_size(size):
    """A size's local as clang-format lays it out: on one line, after its =, or after operators"""
    value = source(size)
    for local in (f'    int64 size = {value};', f'    int64 size =\n        {value};'):
        if fits(local):
            return local
    # Each operand on the line it fits, an operator ending a line
    operands, operators = chain(size)
    afters = [*(f' {operator}' for operator in operators), ';']
    first = f'    int64 size = {operands[0]}{afters[0]}'
    if fits(first):
        # The rest aligned after the =
        lines, indent = [first], ' ' * 17
    else:
        # All at the continuation indent below it
        lines, indent = ['    int64 size =', f'        {operands[0]}{afters[0]}'], ' ' * 8
    for operand, after in zip(operands[1:], afters[1:]):
        if fits(f'{lines[-1]} {operand}{after}'):
            lines[-1] += f' {operand}{after}'
        else:
            lines.append(f'{indent}{operand}{after}')
    return '\n'.join(lines)


def generate_range(address):
    """A range's return as clang-format lays it out, each layout tried until one fits"""
    for lines in (
        [f'    return {{{{{address}, MAKE_SUM({address}, size)}}}};'],
        ['    return {', f'        {{{address}, MAKE_SUM({address}, size)}}}};'],
        [f'    return {{{{{address},', f'             MAKE_SUM({address}, size)}}}};'],
        ['    return {', f'        {{{address},', f'         MAKE_SUM({address}, size)}}}};'],
        [
            f'    return {{{{{address},',
            f'             MAKE_SUM({address},',
            '                      size)}};',
        ],
    ):
        if fits('\n'.join(lines)):
            break
    return '\n'.join(lines)


def generate_get_h_method(index, addr_var, size, templates):
    range_pair = f'{addr_var}, MAKE_SUM({addr_var}, {source(size)})'
    method = templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.get_h_method.txt', index=index, ranges=range_pair
    )
    if fits(method):
        return method
    # The size first, as a local of its own
    return templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.get_h_method_wrapped.txt',
        index=index,
        size=generate_size(size),
        range=generate_range(addr_var),
    )


def generate_constructor_packed(class_name, op_name, attributes):
    """A constructor too wide for its two templates, packed as clang-format packs it"""
    parameters = ['Tensor *y', 'std::vector<Tensor *> X', *(f'int64 {a}' for a in attributes)]
    declarator = call_lines(f'  {class_name}(', parameters, ')', 2, declaration=True)
    # Initializers as many to a line as fit, one too wide broken after its parenthesis
    pieces = [('Instruction(', f'"{op_name}", y, X)'), *((f'{a}(', f'{a})') for a in attributes)]
    lines = ['      :']
    for k, (head, value) in enumerate(pieces):
        value += ' {' if k == len(pieces) - 1 else ','
        column = len(lines[-1]) + 5
        if fits(f'{lines[-1]} {head}{value}'):
            lines[-1] += f' {head}{value}'
        elif fits(f'        {head}{value}'):
            lines.append(f'        {head}{value}')
        elif fits(f'{lines[-1]} {head}') and fits(' ' * column + value):
            lines[-1] += f' {head}'
            lines.append(' ' * column + value)
        else:
            lines += [f'        {head}', f'            {value}']
    return '\n'.join([declarator, *lines])


def generate_instruction_class(instruction, class_name, templates):
    """Generate C++ instruction class from instruction definition"""
    op_name = instruction.instruction
    comp_attrs = instruction.comp_attr
    params = instruction.parameters
    # Each attribute's generated name; the assembly prints the spec's
    names = {attr: get_attribute_name(attr) for attr in [*comp_attrs, *params]}

    # Generate members
    comp_attr_members = ''
    for attr in comp_attrs:
        comp_attr_members += f'  int64 {names[attr]};\n'

    # A blank line parts the two kinds of member
    addr_var_members = '\n' if comp_attrs else ''
    for param in params:
        member = f'  IntVar *{names[param]} = MAKE_INT_VAR(INT64_MIN, INT64_MAX);'
        if not fits(member):
            # Broken after its =
            member = f'  IntVar *{names[param]} =\n      MAKE_INT_VAR(INT64_MIN, INT64_MAX);'
        addr_var_members += member + '\n'

    # get_h methods: the output's first (get_h0), then the inputs'
    get_h_methods = ''
    get_h_cases = ''

    # The output's range first, then each input's
    operands = [*instruction.instr_output_asts, *instruction.instr_input_asts]
    for h_index, (_, starts, sizes) in enumerate(operands):
        addr_var = source(starts[0])

        get_h_methods += generate_get_h_method(h_index, addr_var, sizes[0], templates)
        get_h_cases += templates.render_lines(
            'instructions.h.INSTRUCTION_CLASSES.get_h_case.txt', index=h_index
        )

    # Generate constructor parameters
    comp_attr_params = ''
    if comp_attrs:
        comp_attr_params = ', ' + ', '.join(f'int64 {names[attr]}' for attr in comp_attrs)

    # Generate constructor initialization
    comp_attr_init = ''
    if comp_attrs:
        comp_attr_init = ', ' + ', '.join(f'{names[attr]}({names[attr]})' for attr in comp_attrs)

    # The constructor on one line, else broken before its initializers
    constructor = templates.render_fitting(
        'instructions.h.INSTRUCTION_CLASSES.constructor.txt',
        class_name=class_name,
        op_name=op_name,
        parameters=comp_attr_params,
        initializers=comp_attr_init,
    )
    if not fits(constructor):
        constructor = generate_constructor_packed(
            class_name, op_name, [names[attr] for attr in comp_attrs]
        )

    # The text names each attribute's value, a line each
    values = [(attr, names[attr]) for attr in comp_attrs]
    values += [(param, f'{names[param]}->Value()') for param in params]
    str_pieces = ''
    for index, (attr, value) in enumerate(values):
        str_pieces += templates.render_fitting(
            'instructions.h.INSTRUCTION_CLASSES.str_piece.txt',
            label=(', ' if index else '') + f'{attr} = ',
            value=value,
        )
        str_pieces += '\n'

    # Address variables on one line, else pushed a line each
    variables = [names[param] for param in params]
    get_int_var = templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.get_int_var.txt', variables=', '.join(variables)
    )
    if not fits(get_int_var):
        pushes = ''.join(
            templates.render_lines(
                'instructions.h.INSTRUCTION_CLASSES.get_int_var_push.txt', variable=variable
            )
            for variable in variables
        )
        get_int_var = templates.render_lines(
            'instructions.h.INSTRUCTION_CLASSES.get_int_var_pushed.txt', pushes=pushes
        )

    # Generate complete class using template
    return templates.render_lines(
        'instructions.h.INSTRUCTION_CLASSES.class.txt',
        CLASS_NAME=class_name,
        COMP_ATTR_MEMBERS=comp_attr_members,
        ADDR_VAR_MEMBERS=addr_var_members,
        GET_H_METHODS=get_h_methods,
        CONSTRUCTOR=constructor,
        STR_PIECES=str_pieces,
        GET_H_CASES=get_h_cases,
        GET_INT_VAR=get_int_var,
    )


def generate_instructions_file(backend_gen_dir, instructions):
    """Template instructions.h from generic file"""
    templates = get_backend_template_loader()

    # Read generic template file
    generic_file = os.path.join(backend_gen_dir, 'cpp', 'malloc', 'include', 'instructions.h')
    with open(generic_file, 'r') as f:
        content = f.read()

    # Generate all instruction classes, a blank line between two
    class_names = get_type_names([instruction.instruction for instruction in instructions])
    classes = '\n'.join(
        generate_instruction_class(instruction, class_names[instruction.instruction], templates)
        for instruction in instructions
    ).rstrip('\n')

    # Replace placeholder
    content = fill_placeholders(content, {'INSTRUCTION_CLASSES': classes}, generic_file)

    # Write back
    with open(generic_file, 'w') as f:
        f.write(content)
