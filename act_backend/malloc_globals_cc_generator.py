import os

from .instruction_info import get_buffer_name
from .template_loader import fill_placeholders


def generate_globals_models(data_models):
    models = []

    idx = 0
    for data_model in data_models:
        idx += 1
        if data_model.var_name != 'd0':
            buffer_name = get_buffer_name(data_model.var_name)

            access_dims = '{' + ','.join(str(dim) for dim in data_model.access_dim) + '}'
            num_unit = len(data_model.unit_dim)
            out_str = (
                f'    {{"{buffer_name}", new Storage("{buffer_name}", {access_dims}, {num_unit})}}'
            )
            if idx < len(data_models):
                out_str += ','
            models.append(out_str)

    return models


def generate_globals_file(act_dest_dir: str, data_models) -> None:
    globals_file = os.path.join(act_dest_dir, 'cpp', 'malloc', 'src', 'globals.cc')

    with open(globals_file, 'r') as f:
        globals_content = f.read()

    globals_models = generate_globals_models(data_models)

    replacements = {
        'GLOBALS_DATA_MODELS': '\n'.join(globals_models),
    }

    globals_content = fill_placeholders(globals_content, replacements, globals_file)

    with open(globals_file, 'w') as f:
        f.write(globals_content)
