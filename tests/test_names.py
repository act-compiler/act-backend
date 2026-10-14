"""Tests of the names a spec gives: whatever the generated sources spell, a backend compiles."""

import os
import re
import subprocess

import pytest
from taidl import Accelerator

import act_backend
from act_backend import generate_backend


def add_load(acc, name, model, attribute, addresses):
    """A load of rows of bf16 into a buffer, its attribute and addresses named by the caller"""
    source, target = addresses
    instr = acc.add_instruction(name, [attribute], [source, target])
    instr.set_inputs([['d0', [f'@a.{source}'], [f'@c.{attribute} * 128']]])
    instr.set_outputs([[model, [f'@a.{target}'], [f'@c.{attribute}']]])
    instr.add_semantics(f"""
ENTRY {{
    %In0 = u8[`@c.{attribute} * 128`] parameter(0);
    %a = u8[`@c.{attribute}`,64,2] reshape(%In0);
    ROOT %Out0 = bf16[`@c.{attribute}`,64] bitcast_convert(%a);
}}
""")


def names():
    """Two loads, a product and a store, each attribute named as the sources' own names are"""
    acc = Accelerator('NAMES')
    acc.add_data_model('d1', [128], [64], 'bf16')
    acc.add_data_model('d2', [128], [64], 'bf16')
    add_load(acc, 'load_a', 'd1', 'type', ('y', 'X'))
    add_load(acc, 'load_b', 'd2', 'lhs_metadata', ('start', 'num'))

    instr = acc.add_instruction('mul', [], ['register', 'solver', 'get_h0'])
    instr.set_inputs([['d1', ['@a.register'], ['64']], ['d2', ['@a.solver'], ['64']]])
    instr.set_outputs([['d1', ['@a.get_h0'], ['64']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %In1 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In0, %In1), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

    instr = acc.add_instruction('store', ['value'], ['str', 'Tensor'])
    instr.set_inputs([['d1', ['@a.str'], ['@c.value']]])
    instr.set_outputs([['d0', ['@a.Tensor'], ['@c.value * 128']]])
    instr.add_semantics("""
ENTRY {
    %In0 = bf16[`@c.value`,64] parameter(0);
    %a = u8[`@c.value`,64,2] bitcast_convert(%In0);
    ROOT %Out0 = u8[`@c.value * 128`] reshape(%a);
}
""")
    return acc


@pytest.fixture(scope='module')
def backend(tmp_path_factory):
    """Generate the spec's backend once and return the binary's path"""
    work_dir = str(tmp_path_factory.mktemp('names'))
    generate_backend(names(), work_dir)
    return os.path.join(work_dir, 'backends', 'NAMES')


# The product of two inputs
PRODUCT = """HloModule product, entry_computation_layout={(bf16[64,64]{1,0}, bf16[64,64]{1,0})->bf16[64,64]{1,0}}

ENTRY main {
  %a = bf16[64,64] parameter(0)
  %b = bf16[64,64] parameter(1)
  ROOT %c = bf16[64,64] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
"""


def test_names_the_generated_sources_spell_compile(backend, tmp_path):
    path = tmp_path / 'kernel.hlo'
    path.write_text(PRODUCT)
    output = tmp_path / 'kernel.py'
    arguments = ['--input', str(path), '--output', str(output), '--log', str(tmp_path / 'log')]
    run = subprocess.run([backend, *arguments], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    # Each call by the spec's names, its attributes by theirs
    calls = [
        (name, re.findall(r'(\w+) = ', arguments))
        for name, arguments in re.findall(r'api\.(\w+)\(([^)]*)\)', output.read_text())
    ]
    assert sorted(calls) == [
        ('load_a', ['type', 'y', 'X']),
        ('load_b', ['lhs_metadata', 'start', 'num']),
        ('mul', ['register', 'solver', 'get_h0']),
        ('store', ['value', 'str', 'Tensor']),
    ]


def test_prefixes_are_the_generated_names_own():
    package = os.path.dirname(act_backend.__file__)
    spelled = set()
    # The skeletons and the fragments filled into them
    for folder in ('generic', 'templates'):
        for root, dirs, files in os.walk(os.path.join(package, folder)):
            dirs[:] = [d for d in dirs if d not in ('target', 'build')]
            for name in files:
                if name.endswith(('.rs', '.cc', '.h', '.txt')):
                    with open(os.path.join(root, name)) as f:
                        # A placeholder is filled, so it spells no name
                        text = re.sub(r'\{\{\w+\}\}', '', f.read())
                    spelled |= set(re.findall(r'\b[A-Za-z_]\w*', text))
    # No skeleton name in an attribute's namespace
    assert [name for name in spelled if name.startswith('attr_')] == []
