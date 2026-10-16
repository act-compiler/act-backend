import os

from .instruction_info import get_buffer_name
from .template_loader import call_lines, fill_placeholders, fits


def generate_globals_models(data_models):
    models = []

    for data_model in data_models:
        if data_model.var_name != 'd0':
            buffer_name = get_buffer_name(data_model.var_name)

            access_dims = '{' + ', '.join(str(dim) for dim in data_model.access_dim) + '}'
            num_unit = len(data_model.unit_dim)
            # Every entry ends in a comma, the last one too
            entry = (
                f'    {{"{buffer_name}", new Storage("{buffer_name}", {access_dims}, {num_unit})}},'
            )
            if not fits(entry):
                # Broken after the name, the storage's arguments packed
                arguments = [f'"{buffer_name}"', access_dims, str(num_unit)]
                storage = call_lines('     new Storage(', arguments, ')},', 5)
                entry = f'    {{"{buffer_name}",\n{storage}'
            models.append(entry)

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
