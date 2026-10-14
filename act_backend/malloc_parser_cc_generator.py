import os

from .instruction_info import get_attribute_name, get_graph_name, get_type_names
from .template_loader import fill_placeholders, get_backend_template_loader


def generate_parser_instruction(instruction, class_name, templates):
    op_name = get_graph_name(instruction.instruction)
    comp_attrs = instruction.comp_attr

    if comp_attrs:
        # Each attribute from the one string they travel in
        parses = '\n'.join(
            templates.render(
                'parser.cc.INSTRUCTION_CASES.attribute.txt',
                variable=get_attribute_name(attr_name),
            )
            for attr_name in comp_attrs
        )
        return templates.render(
            'parser.cc.INSTRUCTION_CASES.case_with_attr.txt',
            op_name=op_name,
            ClassName=class_name,
            attribute_parses=parses,
            attributes=', '.join(get_attribute_name(attr_name) for attr_name in comp_attrs),
        )
    else:
        return templates.render(
            'parser.cc.INSTRUCTION_CASES.case_no_attr.txt', op_name=op_name, ClassName=class_name
        )


def generate_parser_instructions(instructions, templates):
    cases = []
    class_names = get_type_names([instruction.instruction for instruction in instructions])
    for instruction in instructions:
        class_name = class_names[instruction.instruction]
        cases.append(generate_parser_instruction(instruction, class_name, templates))

    # Each case closes the one above it
    return '\n'.join(cases)


def generate_parser_file(act_dest_dir, instructions):
    parser_file = os.path.join(act_dest_dir, 'cpp', 'malloc', 'src', 'parser.cc')

    templates = get_backend_template_loader()

    with open(parser_file, 'r') as f:
        content = f.read()

    instruction_cases = generate_parser_instructions(instructions, templates)

    content = fill_placeholders(content, {'INSTRUCTION_CASES': instruction_cases}, parser_file)

    with open(parser_file, 'w') as f:
        f.write(content)
