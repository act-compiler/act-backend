"""What a backend cannot be generated for: each refused by name, before anything is written"""

import os
import re
import keyword

from .instruction_info import get_buffer_name, get_kebab_name, to_camel_case

# A name the generated sources can spell as an identifier
IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

# Operators of the e-graph, by the word a rule spells each with
EGRAPH_WORDS = {
    'add',
    'bitcvt',
    'broadcast',
    'concat',
    'constant',
    'convert',
    'copy',
    'divide',
    'dot',
    'exponential',
    'eye',
    'or',
    'reduce',
    'reshape',
    'shift-left',
    'shift-right-logical',
    'slice',
    'xor',
    'transpose',
}

# Variants the e-graph holds beside the instructions', and the name Rust keeps for the type
EGRAPH_VARIANTS = {
    'OpAdd',
    'OpBitcvt',
    'OpBroadcast',
    'OpConcat',
    'OpConstant',
    'OpConvert',
    'OpCopy',
    'OpDivide',
    'OpDot',
    'OpExp',
    'OpEye',
    'OpOr',
    'OpReduceSum',
    'OpReshape',
    'OpShiftLeft',
    'OpShiftRightLogical',
    'OpSlice',
    'OpXor',
    'OpTranspose',
    'DetectedConst',
    'Var',
    'Self',
}

# Types the allocator declares beside the instructions' classes
ALLOCATOR_TYPES = {
    'Instruction',
    'Slice',
    'Concat',
    'Tensor',
    'Storage',
    'MetaData',
    'MetaDataInfo',
    'ParsedLine',
    'IntExpr',
    'IntVar',
    'Constraints',
    'LiveRange',
    'InterferenceGraph',
    'EsunCache',
}

# Words the allocator reads as a declaration where a line's instruction stands
DECLARATION_WORDS = ('Var', 'DCC')

# Buffers the selector holds beside the data models'
BUFFERS = {'HBM', 'ANY'}


def check_accelerator_name(name):
    """Refuse an accelerator name that is no directory's name: the backend is written under it"""
    if name in ('', '.', '..') or os.path.basename(name) != name:
        raise ValueError(f"Accelerator name '{name}' is not the name of one directory")


def check_identifier(what, name):
    """Refuse a name the generated sources cannot spell"""
    if not IDENTIFIER.fullmatch(name):
        raise ValueError(f"{what} '{name}' is not an identifier")


def check_assembly_name(what, name):
    """Refuse a name the assembly, a Python file, cannot spell"""
    check_identifier(what, name)
    if keyword.iskeyword(name):
        raise ValueError(f"{what} '{name}' is a Python keyword, which no assembly can spell")


def check_instruction_names(instructions):
    """Refuse an instruction name the selector or the allocator cannot carry"""
    carried = {}
    for instruction in instructions:
        name = instruction.instruction
        check_assembly_name('Instruction name', name)
        if get_kebab_name(name) in EGRAPH_WORDS:
            raise ValueError(
                f"Instruction name '{name}' is also the e-graph's operator {get_kebab_name(name)}"
            )
        if name.startswith(DECLARATION_WORDS):
            raise ValueError(
                f"Instruction name '{name}' starts with Var or DCC, "
                'which the allocator reads as a declaration'
            )
        # The variant of the e-graph and the class of the allocator
        generated = to_camel_case(name)
        if not generated[:1].isalpha():
            raise ValueError(
                f"Instruction name '{name}' gives the generated name '{generated}', "
                'which is not an identifier'
            )
        if generated in EGRAPH_VARIANTS | ALLOCATOR_TYPES:
            raise ValueError(
                f"Instruction name '{name}' gives the generated name {generated}, "
                'which the backend already holds'
            )
        if generated in carried:
            raise ValueError(
                f"Instruction names '{carried[generated]}' and '{name}' "
                f'both give the generated name {generated}'
            )
        carried[generated] = name


def check_data_model_names(data_models):
    """Refuse a data model name that gives no buffer of its own"""
    carried = {}
    for data_model in data_models:
        name = data_model.var_name
        # d0 is the HBM, which the backend holds itself
        if name == 'd0':
            continue
        check_identifier('Data model name', name)
        buffer = get_buffer_name(name)
        if buffer in BUFFERS:
            raise ValueError(
                f"Data model name '{name}' gives the buffer {buffer}, which the backend already holds"
            )
        if buffer in carried:
            raise ValueError(
                f"Data model names '{carried[buffer]}' and '{name}' both give the buffer {buffer}"
            )
        carried[buffer] = name


def check_names(instructions, data_models):
    """Refuse a spec whose names the generated backend cannot carry"""
    check_instruction_names(instructions)
    check_data_model_names(data_models)
