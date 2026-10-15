import os

from .instruction_info import (
    InstructionInfo,
    extract_instruction_info,
    get_buffer_name,
    get_rust_variant_name,
)
from .template_loader import fill_placeholders, get_backend_template_loader


def generate_buffer_variants(data_models, templates):
    """Generate buffer variants using templates"""
    variants = []
    display_arms = []

    for data_model in data_models:
        if data_model.var_name != 'd0':
            buffer_name = get_buffer_name(data_model.var_name)
            variants.append(
                templates.render('buffer.rs.BUFFER_VARIANTS.variant.txt', buffer_name=buffer_name)
            )
            display_arms.append(
                templates.render(
                    'buffer.rs.BUFFER_DISPLAY_MATCH_ARMS.arm.txt', buffer_name=buffer_name
                )
            )

    return variants, display_arms


def generate_buffer_assignment_arms(instructions_info: list[InstructionInfo], templates):
    """Generate buffer assignment arms using templates"""
    buffer_assignment_arms = []

    for info in instructions_info:
        variant_name = get_rust_variant_name(info.name)
        pattern = '(_, _)' if info.has_metadata else '(_)'
        buffer_assignment_arms.append(
            templates.render(
                'buffer.rs.ISA_BUFFER_ASSIGNMENT_MATCH_ARMS.arm.txt',
                variant_name=variant_name,
                pattern=pattern,
                buffer_assignment=info.buffer_assignment,
            )
        )

    return buffer_assignment_arms


def generate_buffer_rs_file(act_dest_dir, instructions, data_models):
    """Template buffer.rs from generic file"""
    templates = get_backend_template_loader()
    buffer_file = os.path.join(act_dest_dir, 'src', 'ir', 'buffer.rs')

    with open(buffer_file, 'r') as f:
        content = f.read()

    buffer_variants, buffer_display_arms = generate_buffer_variants(data_models, templates)
    instructions_info = extract_instruction_info(instructions)
    buffer_assignment_arms = generate_buffer_assignment_arms(instructions_info, templates)

    replacements = {
        'BUFFER_VARIANTS': '\n'.join(buffer_variants),
        'BUFFER_DISPLAY_MATCH_ARMS': '\n'.join(buffer_display_arms),
        'ISA_BUFFER_ASSIGNMENT_MATCH_ARMS': '\n'.join(buffer_assignment_arms),
    }

    content = fill_placeholders(content, replacements, buffer_file)

    with open(buffer_file, 'w') as f:
        f.write(content)
