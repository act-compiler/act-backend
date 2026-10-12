"""What a backend cannot be generated for: each refused by name, before anything is written"""

import os
import re

from .instruction_info import get_buffer_name

# A name the generated sources can spell as an identifier
IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

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
    check_data_model_names(data_models)
