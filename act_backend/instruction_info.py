"""What several generators read off an instruction: its record and its names"""

from dataclasses import dataclass


@dataclass
class InstructionInfo:
    name: str
    arity: int
    has_metadata: bool
    buffer_assignment: str


def to_camel_case(snake_str):
    return ''.join(word.capitalize() for word in snake_str.split('_'))


def get_rust_variant_name(instr_name: str) -> str:
    # A hyphen separates words like an underscore
    return to_camel_case(instr_name.replace('-', '_'))


def get_kebab_name(instr_name: str) -> str:
    return instr_name.replace('_', '-')


def get_buffer_name(data_model_name: str) -> str:
    return 'HBM' if data_model_name == 'd0' else data_model_name.upper()


def extract_instruction_info(instructions) -> list[InstructionInfo]:
    info_list = []

    for instruction in instructions:
        name = instruction.instruction
        arity = len(instruction.instr_inputs)
        has_metadata = len(instruction.comp_attr) > 0

        buffers = []
        for output in instruction.instr_outputs:
            buffer_name = get_buffer_name(output[0])
            buffers.append(f'Buffer::{buffer_name}')

        for input_item in instruction.instr_inputs:
            buffer_name = get_buffer_name(input_item[0])
            buffers.append(f'Buffer::{buffer_name}')

        buffer_assignment = f'Some(vec![{", ".join(buffers)}])'

        info_list.append(InstructionInfo(name, arity, has_metadata, buffer_assignment))

    return info_list
