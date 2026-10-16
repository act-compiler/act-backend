import os

from .instruction_info import get_buffer_name
from .template_loader import fill_placeholders, fits, get_backend_template_loader


def generate_buffer_names(data_models, templates):
    """Generate buffer names using templates"""
    buffer_names = []
    for dm in data_models:
        if dm.var_name != 'd0':
            buffer_name = get_buffer_name(dm.var_name)
            buffer_names.append(
                templates.render(
                    'act_malloc.cc.TOPOLOGICAL_ORDER.name.txt', buffer_name=f'"{buffer_name}"'
                )
            )
    return buffer_names


def generate_topological_order(data_models, templates):
    """The scheduler's call, its buffer names listed above it where the call is too wide"""
    buffer_names = generate_buffer_names(data_models, templates)
    call = templates.render(
        'act_malloc.cc.TOPOLOGICAL_ORDER.call.txt', buffer_names=', '.join(buffer_names)
    )
    if fits(call):
        return call
    # A trailing comma keeps each name on a line of its own
    listed = '\n'.join(f'      {buffer_name},' for buffer_name in buffer_names)
    return templates.render('act_malloc.cc.TOPOLOGICAL_ORDER.call_listed.txt', buffer_names=listed)


def generate_act_malloc_file(act_dest_dir, data_models):
    """Template act_malloc.cc from generic file"""
    templates = get_backend_template_loader()
    file_path = os.path.join(act_dest_dir, 'cpp', 'malloc', 'src', 'act_malloc.cc')

    with open(file_path, 'r') as f:
        content = f.read()

    order = generate_topological_order(data_models, templates)
    content = fill_placeholders(content, {'TOPOLOGICAL_ORDER': order}, file_path)

    with open(file_path, 'w') as f:
        f.write(content)
