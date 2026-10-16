import os

from .instruction_info import (
    InstructionInfo,
    extract_instruction_info,
    get_buffer_name,
)
from .template_loader import CALL_WIDTH, fill_placeholders, fits, get_backend_template_loader


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
            # write!'s arguments too wide for a line go a line each
            arm = 'buffer.rs.BUFFER_DISPLAY_MATCH_ARMS.arm.txt'
            if len(f'f, "{buffer_name}"') > CALL_WIDTH:
                arm = 'buffer.rs.BUFFER_DISPLAY_MATCH_ARMS.arm_vertical.txt'
                display_arms.append(templates.render_arm(arm, buffer_name=buffer_name))
            else:
                display_arms.append(templates.render_fitting(arm, buffer_name=buffer_name))

    return variants, display_arms


def generate_buffer_assignment_arms(instructions_info: list[InstructionInfo], templates):
    """Generate buffer assignment arms using templates"""
    buffer_assignment_arms = []

    for info in instructions_info:
        variant_name = info.type_name
        pattern = '(_, _)' if info.has_metadata else '(_)'
        arm = templates.render(
            'buffer.rs.ISA_BUFFER_ASSIGNMENT_MATCH_ARMS.arm.txt',
            variant_name=variant_name,
            pattern=pattern,
            buffer_assignment=info.buffer_assignment,
        )
        if not fits(arm):
            # The assignment in a block, a level deeper
            arm = templates.render(
                'buffer.rs.ISA_BUFFER_ASSIGNMENT_MATCH_ARMS.arm_wrapped.txt',
                variant_name=variant_name,
                pattern=pattern,
                buffer_assignment=info.buffer_assignment.replace('\n', '\n  '),
            )
        buffer_assignment_arms.append(arm)

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
