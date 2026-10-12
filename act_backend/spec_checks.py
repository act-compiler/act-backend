"""What a backend cannot be generated for: each refused by name, before anything is written"""

import os


def check_accelerator_name(name):
    """Refuse an accelerator name that is no directory's name: the backend is written under it"""
    if name in ('', '.', '..') or os.path.basename(name) != name:
        raise ValueError(f"Accelerator name '{name}' is not the name of one directory")
