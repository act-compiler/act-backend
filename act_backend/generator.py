"""Backend code generator for ACT compiler

This module provides the main interface for generating backend code.
It coordinates all the individual file generators to produce a complete
ACT backend implementation.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from taidl import Accelerator

from .build_rs_generator import generate_build_rs_file
from .ir_buffer_rs_generator import generate_buffer_rs_file
from .ir_egraph_rs_generator import generate_egraph_rs_file
from .isel_applier_rs_generator import generate_applier_file
from .isel_ir2isa_rewrites_rs_generator import generate_ir2isa_rewrites_rs_file
from .isel_ir2isa_rewrites_txt_generator import generate_ir2isa_rewrites_txt_file
from .malloc_act_malloc_cc_generator import generate_act_malloc_file
from .malloc_globals_cc_generator import generate_globals_file
from .malloc_instructions_h_generator import generate_instructions_file
from .malloc_parser_cc_generator import generate_parser_file
from .spec_checks import (
    check_inplace,
    check_names,
    check_operands,
    check_outputs,
    check_semantics,
)


@dataclass
class InstructionMetadata:
    name: str
    semantics_ast: object
    expressions: dict
    has_comp_attrs: bool
    rhs_size: int


def get_instruction_metadata(instructions) -> list[InstructionMetadata]:
    """Extract instruction metadata needed for rewrite rule generation"""
    metadata_list = []
    for instruction in instructions:
        metadata = InstructionMetadata(
            name=instruction.instruction,
            semantics_ast=instruction.instr_semantics_ast,
            expressions=instruction.instr_expression_asts,
            has_comp_attrs=len(instruction.comp_attr) > 0,
            rhs_size=len(instruction.instr_inputs) + 1,
        )
        metadata_list.append(metadata)
    return metadata_list


def generate_backend(
    accelerator: Accelerator, base_dir: str | None = None, cost: str | None = None
) -> None:
    """
    Generate complete ACT backend code for an accelerator.

    Args:
        accelerator: TAIDL Accelerator object
        base_dir: Base directory of the project. Defaults to accelerator.base_dir.
        cost: Library that defines the cost function act_cost. Defaults to the instruction count.
    """
    if base_dir is None:
        base_dir = accelerator.base_dir

    # An earlier binary and a part of one go first
    final_dest_path = os.path.join(base_dir, 'backends', accelerator.name)
    partial_path = final_dest_path + '.partial'
    for path in (final_dest_path, partial_path):
        if os.path.lexists(path):
            os.remove(path)

    # The cost library is linked from where it lies
    if cost is not None:
        cost = os.path.abspath(cost)
        if not cost.endswith(('.o', '.a', '.so')):
            raise ValueError(
                f'Cost library {cost} is not an object file (.o), a static library (.a) '
                'or a shared library (.so)'
            )
        # The build passes the path on as one line
        if '\n' in cost:
            raise ValueError(f'Cost library {cost!r} has a line break in its path')
        # A run path, which finds a shared library again, is split at colons
        if cost.endswith('.so') and ':' in os.path.dirname(cost):
            raise ValueError(
                f'Cost library {cost} is a shared library under a directory with a colon'
            )
        if not os.path.isfile(cost):
            raise FileNotFoundError(f'Cost library {cost} does not exist')

    # An instruction with no semantics is never selected, so the backend leaves it out
    instructions = [
        instruction for instruction in accelerator.instructions if instruction.instr_semantics_ast
    ]

    # A spec the checks refuse is refused by name
    check_names(instructions, accelerator.data_model)
    # Count the outputs first: each is a semantics line
    check_outputs(instructions)
    check_semantics(instructions)
    check_operands(instructions, accelerator.data_model)
    check_inplace(instructions)

    # Setup directories
    target_dir = os.path.join(base_dir, 'targets', accelerator.name)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    generic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generic')
    backend_gen_dir = os.path.join(target_dir, 'backend')

    # Copy generic backend structure, without a build directory left in it
    if os.path.exists(backend_gen_dir):
        shutil.rmtree(backend_gen_dir)
    shutil.copytree(generic_dir, backend_gen_dir, ignore=shutil.ignore_patterns('target', 'build'))

    print(f'Copied generic backend structure to {backend_gen_dir}')

    instruction_metadata = get_instruction_metadata(instructions)

    # Generate rewrite rules
    generate_ir2isa_rewrites_txt_file(backend_gen_dir, instruction_metadata)
    print('Generated ir2isa_rewrites.txt')

    # Generate Rust files
    generate_ir2isa_rewrites_rs_file(backend_gen_dir, instruction_metadata)
    print('Generated ir2isa_rewrites.rs')

    generate_buffer_rs_file(backend_gen_dir, instructions, accelerator.data_model)
    print('Generated buffer.rs')

    generate_egraph_rs_file(backend_gen_dir, instructions)
    print('Generated egraph.rs')

    generate_applier_file(backend_gen_dir, instructions)
    print('Generated applier.rs')

    # Generate C++ malloc files
    generate_globals_file(backend_gen_dir, accelerator.data_model)
    print('Generated globals.cc')

    generate_instructions_file(backend_gen_dir, instructions)
    print('Generated instructions.h')

    generate_parser_file(backend_gen_dir, instructions)
    print('Generated parser.cc')

    generate_act_malloc_file(backend_gen_dir, accelerator.data_model)
    print('Generated act_malloc.cc')

    # Generate the build script, which links the cost model
    generate_build_rs_file(backend_gen_dir, cost)

    print(f'Backend generation complete for {accelerator.name}')

    # Build the backend
    print(f'Building backend for {accelerator.name}')
    cargo_build_dir = os.path.join(backend_gen_dir, 'target')
    build = subprocess.run(
        ['cargo', 'build', '--release', '--locked'],
        cwd=backend_gen_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    # A failed build is reported with what cargo printed
    if build.returncode != 0:
        raise RuntimeError(f'Backend build failed in {backend_gen_dir}:\n{build.stdout}')
    print(f'Backend build complete for {accelerator.name}')

    # Copy built backend to final destination
    build_backend_path = os.path.join(cargo_build_dir, 'release', 'backend')
    Path(final_dest_path).parent.mkdir(parents=True, exist_ok=True)
    # The binary takes its name once it is whole
    try:
        shutil.copy(build_backend_path, partial_path)
        os.replace(partial_path, final_dest_path)
    finally:
        # A copy that stops leaves no part
        if os.path.lexists(partial_path):
            os.remove(partial_path)
    print(f'Final backend binary located at {final_dest_path}')
