import os
import shutil

from .template_loader import fill_placeholders, get_backend_template_loader


def rust_string(text):
    """Escape a text for a Rust string literal"""
    return text.replace('\\', '\\\\').replace('"', '\\"')


def generate_cost_model(cost, templates):
    """Generate the lines that link the cost model: the default, or the library named"""
    if cost is None:
        return templates.render('build.rs.COST_MODEL.default.txt')

    lines = [templates.render('build.rs.COST_MODEL.library.txt', library=rust_string(cost))]
    # A shared library is found again at each run, in its directory
    if cost.endswith('.so'):
        directory = rust_string(os.path.dirname(cost))
        lines.append(templates.render('build.rs.COST_MODEL.run_path.txt', directory=directory))
    return '\n'.join(lines)


def generate_build_rs_file(act_dest_dir, cost):
    """Template build.rs from generic file"""
    templates = get_backend_template_loader()
    file_path = os.path.join(act_dest_dir, 'build.rs')

    with open(file_path, 'r') as f:
        content = f.read()

    cost_model = generate_cost_model(cost, templates)
    content = fill_placeholders(content, {'COST_MODEL': cost_model}, file_path)

    with open(file_path, 'w') as f:
        f.write(content)

    # A backend that names a cost model carries no default
    if cost is not None:
        shutil.rmtree(os.path.join(act_dest_dir, 'cpp', 'cost'))
