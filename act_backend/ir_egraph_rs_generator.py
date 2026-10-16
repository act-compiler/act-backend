import os

from .instruction_info import (
    InstructionInfo,
    extract_instruction_info,
    get_graph_name,
    get_rewrite_word,
)
from .template_loader import (
    CALL_WIDTH,
    LINE_WIDTH,
    fill_placeholders,
    fits,
    get_backend_template_loader,
)


def built_node(variant_name, arguments):
    """The e-node a from_op arm builds, an argument a line where rustfmt would break the call"""
    callee = f'TensorOp::{variant_name}'
    call = f'{callee}({", ".join(arguments)})'
    # A nested call is held to the call width, unless its callee alone takes it
    if len(call) <= CALL_WIDTH or len(callee) >= CALL_WIDTH and fits(f'        Ok({call})'):
        return call
    if len(callee) >= CALL_WIDTH and fits(f'          {call},'):
        # The call on a line of its own
        return f'\n          {call},\n        '
    lines = ''.join(f'          {argument},\n' for argument in arguments)
    if len(callee) >= CALL_WIDTH:
        # On a line of its own, an argument a line
        lines = lines.replace('\n', '\n  ')
        return f'\n          {callee}(\n  {lines}        ),\n        '
    return f'{callee}(\n{lines}        )'


def from_op_arm(rewrite_word):
    """The with-metadata from_op arm, its guard's chain kept clear of ` => {`"""
    chain = f'        .is_some_and(|(word, _)| word == "{rewrite_word}")'
    if len(chain) <= LINE_WIDTH - len(' => {'):
        return 'egraph.rs.ISA_FROM_OP_MATCH_ARMS.arm_with_metadata.txt'
    return 'egraph.rs.ISA_FROM_OP_MATCH_ARMS.arm_with_metadata_wrapped.txt'


def set_metadata_arm(variant_name, templates):
    """The set_metadata arm as rustfmt lays it out, each of its layouts tried until one fits"""
    for arm in ('arm', 'arm_wrapped', 'arm_vertical', 'arm_chained', 'arm_called'):
        text = templates.render(
            f'egraph.rs.ISA_SET_METADATA_MATCH_ARMS.{arm}.txt', variant_name=variant_name
        )
        if fits(text):
            break
    return text


def display_arm(info, templates):
    """The Display arm of an instruction as rustfmt lays it out"""
    graph_name = get_graph_name(info.name)
    keys = {'variant_name': info.type_name, 'graph_name': graph_name}
    arm = 'egraph.rs.ISA_DISPLAY_MATCH_ARMS.arm_no_metadata'
    arguments = f'f, "{graph_name}"'
    if info.has_metadata:
        arm = 'egraph.rs.ISA_DISPLAY_MATCH_ARMS.arm_with_metadata'
        arguments = f'f, "{graph_name}[rows=\'{{}}\']", data'
    if len(arguments) <= CALL_WIDTH:
        return templates.render_fitting(f'{arm}.txt', **keys)
    # write!'s arguments too wide for a line go a line each
    return templates.render_arm(f'{arm}_vertical.txt', **keys)


def generate_instruction_variants(instructions_info: list[InstructionInfo], templates):
    """Generate instruction variants using templates"""
    enum_variants = []
    num_children_arms = []
    set_metadata_arms = []
    children_arms = []
    children_mut_arms = []
    from_op_arms = []
    display_arms = []

    for info in instructions_info:
        variant_name = info.type_name
        rewrite_word = get_rewrite_word(info.name)

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
            set_metadata_arms.append(set_metadata_arm(variant_name, templates))

        # Children arms
        data_pattern = '(_, ids)' if info.has_metadata else '(ids)'
        children_arms.append(
            templates.render_fitting(
                'egraph.rs.ISA_CHILDREN_MATCH_ARMS.arm.txt',
                variant_name=variant_name,
                data_pattern=data_pattern,
            )
        )
        children_mut_arms.append(
            templates.render_fitting(
                'egraph.rs.ISA_CHILDREN_MUT_MATCH_ARMS.arm.txt',
                variant_name=variant_name,
                data_pattern=data_pattern,
            )
        )

        # From op arms
        if info.has_metadata:
            from_op_arms.append(
                templates.render(
                    from_op_arm(rewrite_word),
                    rewrite_word=rewrite_word,
                    arity=str(info.arity),
                    node=built_node(variant_name, ['data.to_string()', 'children']),
                )
            )
        else:
            from_op_arms.append(
                templates.render_fitting(
                    'egraph.rs.ISA_FROM_OP_MATCH_ARMS.arm_no_metadata.txt',
                    rewrite_word=rewrite_word,
                    arity=str(info.arity),
                    node=built_node(variant_name, ['children']),
                )
            )

        # Display arms
        display_arms.append(display_arm(info, templates))

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
