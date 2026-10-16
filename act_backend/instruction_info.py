"""What several generators read off an instruction: its record and its names"""

from collections import Counter
from dataclasses import dataclass


@dataclass
class InstructionInfo:
    name: str
    type_name: str
    arity: int
    attributes: list[str]
    buffer_assignment: str

    @property
    def has_metadata(self):
        # The computational attributes travel as the instruction's metadata
        return bool(self.attributes)


def to_camel_case(snake_str, digit_underscore=False):
    words = snake_str.split('_')
    # If asked, an underscore stays before a word that starts with a digit
    return words[0].capitalize() + ''.join(
        ('_' if digit_underscore and word[:1].isdigit() else '') + word.capitalize()
        for word in words[1:]
    )


def get_type_names(instr_names):
    """Each instruction's generated type: Instr and its camel case

    Two names of one camel case each keep an underscore before a word that starts with a digit;
    the later of two that still collide is numbered.
    """
    folds = Counter(to_camel_case(name) for name in instr_names)
    type_names = {}
    for name in instr_names:
        base = f'Instr{to_camel_case(name, folds[to_camel_case(name)] > 1)}'
        type_name, number = base, 1
        while type_name in type_names.values():
            number += 1
            type_name = f'{base}{number}'
        type_names[name] = type_name
    return type_names


def get_attribute_name(attribute: str) -> str:
    # A prefix no name of the generated sources starts with
    return f'attr_{attribute}'


def get_kebab_name(instr_name: str) -> str:
    return instr_name.replace('_', '-')


def get_rewrite_word(instr_name: str) -> str:
    # The instruction's rewrite word, apart from every operator's
    return f'instr_{get_kebab_name(instr_name)}'


def get_graph_name(instr_name: str) -> str:
    # The instruction as the allocator's selected graph names it
    return f'instr_{instr_name}'


def get_buffer_name(data_model_name: str) -> str:
    # d0 is the backend's HBM, other data models prefixed
    return 'HBM' if data_model_name == 'd0' else f'Model{data_model_name}'


# rustfmt's widest vec! on one line
ARRAY_WIDTH = 60


def extract_instruction_info(instructions) -> list[InstructionInfo]:
    info_list = []
    type_names = get_type_names([instruction.instruction for instruction in instructions])

    for instruction in instructions:
        name = instruction.instruction
        arity = len(instruction.instr_inputs)

        buffers = []
        for output in instruction.instr_outputs:
            buffer_name = get_buffer_name(output[0])
            buffers.append(f'Buffer::{buffer_name}')

        for input_item in instruction.instr_inputs:
            buffer_name = get_buffer_name(input_item[0])
            buffers.append(f'Buffer::{buffer_name}')

        buffer_assignment = f'Some(vec![{", ".join(buffers)}])'

        info_list.append(
            InstructionInfo(
                name, type_names[name], arity, list(instruction.comp_attr), buffer_assignment
            )
        )

    return info_list
