"""Backend code generator for ACT compiler

This module provides the main interface for generating backend code.
It coordinates all the individual file generators to produce a complete
ACT backend implementation.
"""

import os
from pathlib import Path
import shutil
from typing import List
import subprocess

from .ir_buffer_rs_generator import generate_buffer_rs_file
from .ir_egraph_rs_generator import generate_egraph_rs_file
from .malloc_globals_cc_generator import generate_globals_file
from .malloc_instructions_h_generator import generate_instructions_file
from .malloc_parser_cc_generator import generate_parser_file
from .isel_applier_rs_generator import generate_applier_file
from .malloc_act_malloc_cc_generator import generate_act_malloc_file
from .isel_ir2isa_rewrites_txt_generator import generate_ir2isa_rewrites_txt_file
from .isel_ir2isa_rewrites_rs_generator import generate_ir2isa_rewrites_rs_file


def generate_backend(accelerator_name: str, instructions: List, data_models: List,
                     instruction_metadata: List, base_dir: str) -> None:
    """
    Generate complete ACT backend code for an accelerator.

    Args:
        accelerator_name: Name of the accelerator
        instructions: List of Instruction objects
        data_models: List of DataModel objects
        instruction_metadata: List of InstructionMetadata objects
        base_dir: Base directory of the project
    """
    # Setup directories
    target_dir = os.path.join(base_dir, 'targets', accelerator_name)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    generic_dir = os.path.join(base_dir, 'generators', 'backend', 'generic')
    backend_gen_dir = os.path.join(target_dir, 'backend')

    # Copy generic backend structure
    if os.path.exists(backend_gen_dir):
        shutil.rmtree(backend_gen_dir)
    shutil.copytree(generic_dir, backend_gen_dir)

    print(f"Copied generic backend structure to {backend_gen_dir}")

    # Generate rewrite rules
    generate_ir2isa_rewrites_txt_file(backend_gen_dir, instruction_metadata)
    print(f"Generated ir2isa_rewrites.txt")

    # Generate Rust files
    generate_ir2isa_rewrites_rs_file(backend_gen_dir, instruction_metadata)
    print(f"Generated ir2isa_rewrites.rs")

    generate_buffer_rs_file(backend_gen_dir, instructions, data_models)
    print(f"Generated buffer.rs")

    generate_egraph_rs_file(backend_gen_dir, instructions)
    print(f"Generated egraph.rs")

    generate_applier_file(backend_gen_dir, instructions)
    print(f"Generated applier.rs")

    # Generate C++ malloc files
    generate_globals_file(backend_gen_dir, data_models)
    print(f"Generated globals.cc")

    generate_instructions_file(backend_gen_dir, instructions, data_models)
    print(f"Generated instructions.h")

    generate_parser_file(backend_gen_dir, instructions)
    print(f"Generated parser.cc")

    generate_act_malloc_file(backend_gen_dir, data_models)
    print(f"Generated act_malloc.cc")

    print(f"Backend generation complete for {accelerator_name}")

    # Build the backend
    print(f"Building backend for {accelerator_name}")
    cargo_build_dir = os.path.join(backend_gen_dir, 'target')
    if os.path.exists(cargo_build_dir):
        shutil.rmtree(cargo_build_dir)
    subprocess.run(["cargo", "build", "--release"], cwd=backend_gen_dir,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"Backend build complete for {accelerator_name}")

    # Copy built backend to final destination
    build_backend_path = os.path.join(cargo_build_dir, 'release', 'backend')
    final_dest_path = os.path.join(base_dir, 'backends', accelerator_name)
    if os.path.exists(build_backend_path):
        Path(final_dest_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(build_backend_path, final_dest_path)
        print(f"Final backend binary located at {final_dest_path}")
    else:
        raise RuntimeError("Backend build failed. Please check the build logs.")
