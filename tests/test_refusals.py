"""Tests of what generation refuses: a spec no backend can be built for, each by its error."""

import os
import re

import pytest
from taidl import Accelerator

import act_backend
from act_backend import generate_backend, spec_checks


def add_load(acc, instruction='load', attribute='n', address='addr_out', model='d1'):
    """Add an instruction that loads rows into a buffer, each of its names the caller's"""
    instr = acc.add_instruction(instruction, [attribute], ['addr_in', address])
    instr.set_inputs([['d0', ['@a.addr_in'], [f'@c.{attribute} * 128']]])
    instr.set_outputs([[model, [f'@a.{address}'], [f'@c.{attribute}']]])
    instr.add_semantics(f"""
ENTRY load {{
    %In1 = u8[`@c.{attribute} * 128`] parameter(0);
    %a = u8[`@c.{attribute}`,64,2] reshape(%In1);
    ROOT %Out0 = bf16[`@c.{attribute}`,64] bitcast_convert(%a);
}}
""")
    return instr


def loader(name='ONE', model='d1', **names):
    """An accelerator with one data model and one instruction that loads into it"""
    acc = Accelerator(name)
    acc.add_data_model(model, [128], [64], 'bf16')
    add_load(acc, model=model, **names)
    return acc


def refusal(accelerator, tmp_path):
    """Return the error a generation ends on, which has written nothing"""
    with pytest.raises(ValueError) as error:
        generate_backend(accelerator, str(tmp_path / 'base'))
    assert not (tmp_path / 'base').exists()
    return str(error.value)


@pytest.mark.parametrize('name', ['', '.', '..', '../../outside', 'a/b'])
def test_accelerator_name_that_is_no_directory_refused(name, tmp_path):
    # What the name would reach from the base directory
    kept = tmp_path / 'outside' / 'backend' / 'kept'
    kept.parent.mkdir(parents=True)
    kept.touch()
    message = refusal(loader(name=name), tmp_path)
    assert message == f"Accelerator name '{name}' is not the name of one directory"
    assert kept.exists()


@pytest.mark.parametrize(
    'name, reason',
    [
        ('load-2', 'is not an identifier'),
        ('while', 'is a Python keyword, which no assembly can spell'),
        ('dot', "is also the e-graph's operator dot"),
        ('shift_left', "is also the e-graph's operator shift-left"),
        ('Variant', 'starts with Var or DCC, which the allocator reads as a declaration'),
        ('_2', "gives the generated name '2', which is not an identifier"),
        ('var', 'gives the generated name Var, which the backend already holds'),
        ('tensor', 'gives the generated name Tensor, which the backend already holds'),
    ],
)
def test_instruction_name_refused(name, reason, tmp_path):
    message = refusal(loader(instruction=name), tmp_path)
    assert message == f"Instruction name '{name}' {reason}"


def test_instruction_names_that_fold_to_one_refused(tmp_path):
    acc = loader(instruction='load_2')
    add_load(acc, instruction='load2')
    message = refusal(acc, tmp_path)
    assert message == "Instruction names 'load_2' and 'load2' both give the generated name Load2"


@pytest.mark.parametrize(
    'name, reason',
    [
        ('d-1', 'is not an identifier'),
        ('hbm', 'gives the buffer HBM, which the backend already holds'),
        ('any', 'gives the buffer ANY, which the backend already holds'),
    ],
)
def test_data_model_name_refused(name, reason, tmp_path):
    assert refusal(loader(model=name), tmp_path) == f"Data model name '{name}' {reason}"


def test_data_model_names_that_fold_to_one_refused(tmp_path):
    acc = loader(model='d1')
    acc.add_data_model('D1', [64], [64], 'bf16')
    assert refusal(acc, tmp_path) == "Data model names 'd1' and 'D1' both give the buffer D1"


def test_reserved_names_are_the_skeletons_own():
    generic = os.path.join(os.path.dirname(act_backend.__file__), 'generic')

    def read(*path):
        with open(os.path.join(generic, *path)) as f:
            return f.read()

    # Each word the e-graph reads an operator by, and each variant beside the instructions'
    egraph = read('src', 'ir', 'egraph.rs')
    assert set(re.findall(r'== "([a-z-]+)"', egraph)) == spec_checks.EGRAPH_WORDS
    variants = egraph[egraph.index('pub enum TensorOp') : egraph.index('impl TensorOp')]
    variants = set(re.findall(r'^  (\w+)[(,]', variants, re.M))
    assert variants | {'Self'} == spec_checks.EGRAPH_VARIANTS

    # Each buffer beside the data models'
    buffer = read('src', 'ir', 'buffer.rs')
    buffers = buffer[buffer.index('pub enum Buffer') : buffer.index('impl ')]
    assert set(re.findall(r'^  (\w+),', buffers, re.M)) == spec_checks.BUFFERS

    # Each type the allocator declares
    allocator = os.path.join('cpp', 'malloc')
    sources = [
        read(allocator, directory, name)
        for directory in ['include', 'src']
        for name in sorted(os.listdir(os.path.join(generic, allocator, directory)))
    ]
    types = re.findall(r'^(?:class|struct) (\w+)|^using (\w+) =', '\n'.join(sources), re.M)
    assert {a or b for a, b in types} == spec_checks.ALLOCATOR_TYPES
