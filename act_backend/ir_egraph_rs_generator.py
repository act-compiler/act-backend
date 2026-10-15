import os
from typing import List

from .instruction_info import (
    InstructionInfo,
    extract_instruction_info,
    get_kebab_name,
    get_rust_variant_name,
)
from .template_loader import fill_placeholders, get_backend_template_loader


def generate_instruction_variants(instructions_info: List[InstructionInfo], templates):
    """Generate instruction variants using templates"""
    enum_variants = []
    num_children_arms = []
    set_metadata_arms = []
    children_arms = []
    children_mut_arms = []
    from_op_arms = []
    display_arms = []

    for info in instructions_info:
        variant_name = get_rust_variant_name(info.name)
        kebab_name = get_kebab_name(info.name)

        # Enum variant
        if info.has_metadata:
            signature = f'(String, [Id; {info.arity}])'
        else:
            signature = f'([Id; {info.arity}])'
        enum_variants.append(
            templates.render(
                'egraph.rs.ISA_ENUM_VARIANTS.variant.txt',
                variant_name=variant_name,
                signature=signature,
            )
        )

        # Num children arm
        num_children_arms.append(
            templates.render(
                'egraph.rs.ISA_NUM_CHILDREN_MATCH_ARMS.arm.txt',
                variant_name=variant_name,
                arity=str(info.arity),
            )
        )

        # Set metadata arm (only for instructions with metadata)
        if info.has_metadata:
            set_metadata_arms.append(
                templates.render(
                    'egraph.rs.ISA_SET_METADATA_MATCH_ARMS.arm.txt', variant_name=variant_name
                )
            )

        # Children arms
        data_pattern = '(_, ids)' if info.has_metadata else '(ids)'
        children_arms.append(
            templates.render(
                'egraph.rs.ISA_CHILDREN_MATCH_ARMS.arm.txt',
                variant_name=variant_name,
                data_pattern=data_pattern,
            )
        )
        children_mut_arms.append(
            templates.render(
                'egraph.rs.ISA_CHILDREN_MUT_MATCH_ARMS.arm.txt',
                variant_name=variant_name,
                data_pattern=data_pattern,
            )
        )

        # From op arms
        if info.has_metadata:
            from_op_arms.append(
                templates.render(
                    'egraph.rs.ISA_FROM_OP_MATCH_ARMS.arm_with_metadata.txt',
                    kebab_name=kebab_name,
                    arity=str(info.arity),
                    variant_name=variant_name,
                )
            )
        else:
            from_op_arms.append(
                templates.render(
                    'egraph.rs.ISA_FROM_OP_MATCH_ARMS.arm_no_metadata.txt',
                    kebab_name=kebab_name,
                    arity=str(info.arity),
                    variant_name=variant_name,
                )
            )

        # Display arms
        if info.has_metadata:
            display_arms.append(
                templates.render(
                    'egraph.rs.ISA_DISPLAY_MATCH_ARMS.arm_with_metadata.txt',
                    variant_name=variant_name,
                    instruction_name=info.name,
                )
            )
        else:
            display_arms.append(
                templates.render(
                    'egraph.rs.ISA_DISPLAY_MATCH_ARMS.arm_no_metadata.txt',
                    variant_name=variant_name,
                    instruction_name=info.name,
                )
            )

    return {
        'enum_variants': enum_variants,
        'num_children_arms': num_children_arms,
        'set_metadata_arms': set_metadata_arms,
        'children_arms': children_arms,
        'children_mut_arms': children_mut_arms,
        'from_op_arms': from_op_arms,
        'display_arms': display_arms,
    }


def generate_egraph_rs_file(act_dest_dir, instructions):
    """Template egraph.rs from generic file"""
    templates = get_backend_template_loader()
    egraph_file = os.path.join(act_dest_dir, 'src', 'ir', 'egraph.rs')

    with open(egraph_file, 'r') as f:
        content = f.read()

    instructions_info = extract_instruction_info(instructions)
    instruction_parts = generate_instruction_variants(instructions_info, templates)

    replacements = {
        'ISA_ENUM_VARIANTS': '\n'.join(instruction_parts['enum_variants']),
        'ISA_NUM_CHILDREN_MATCH_ARMS': '\n'.join(instruction_parts['num_children_arms']),
        'ISA_SET_METADATA_MATCH_ARMS': '\n'.join(instruction_parts['set_metadata_arms']),
        'ISA_CHILDREN_MATCH_ARMS': '\n'.join(instruction_parts['children_arms']),
        'ISA_CHILDREN_MUT_MATCH_ARMS': '\n'.join(instruction_parts['children_mut_arms']),
        'ISA_FROM_OP_MATCH_ARMS': '\n'.join(instruction_parts['from_op_arms']),
        'ISA_DISPLAY_MATCH_ARMS': '\n'.join(instruction_parts['display_arms']),
    }

    content = fill_placeholders(content, replacements, egraph_file)

    with open(egraph_file, 'w') as f:
        f.write(content)
