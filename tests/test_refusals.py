"""Tests of what generation refuses: a spec no backend can be built for, each by its error."""

import pytest
from taidl import Accelerator

from act_backend import generate_backend


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
