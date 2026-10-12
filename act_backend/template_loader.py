"""Template management for backend code generation

Provides a clean interface for loading and using code templates without
mutable global variables.
"""

import os
import re
from typing import Dict


class TemplateLoader:
    """Manages loading and caching of templates for code generation"""

    def __init__(self, template_dir: str):
        """
        Initialize template loader

        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = template_dir
        self._cache: Dict[str, str] = {}

    def load(self, template_file: str) -> str:
        """
        Load a template file

        Args:
            template_file: Name of the template file

        Returns:
            Template content as string
        """
        if template_file not in self._cache:
            file_path = os.path.join(self.template_dir, template_file)
            with open(file_path, 'r') as f:
                self._cache[template_file] = f.read()
        return self._cache[template_file]

    def render(self, template_file: str, **kwargs) -> str:
        """
        Load and render a template with the given variables

        Args:
            template_file: Name of the template file
            **kwargs: Template variables

        Returns:
            Rendered template content
        """
        template = self.load(template_file)
        result = template
        for key, value in kwargs.items():
            placeholder = f'{{{{{key}}}}}'
            result = result.replace(placeholder, str(value))
        return result


# A skeleton's placeholder: a comment line of its own, a block comment inside a line, or bare
PLACEHOLDER = re.compile(
    r'^[ \t]*// \{\{([A-Z0-9_]+)\}\}$|/\* \{\{([A-Z0-9_]+)\}\} \*/|\{\{([A-Z0-9_]+)\}\}',
    re.MULTILINE,
)


def fill_placeholders(content: str, replacements: Dict[str, str], source: str) -> str:
    """
    Replace each placeholder of a skeleton with its generated code

    Raises ValueError naming `source` when a placeholder has no value or a value no placeholder.
    """
    names = {match.group(match.lastindex) for match in PLACEHOLDER.finditer(content)}
    unfilled = sorted(names - replacements.keys())
    if unfilled:
        raise ValueError(f'{source}: no value for the placeholder {", ".join(unfilled)}')
    unknown = sorted(replacements.keys() - names)
    if unknown:
        raise ValueError(f'{source}: no placeholder for the value {", ".join(unknown)}')
    return PLACEHOLDER.sub(lambda match: replacements[match.group(match.lastindex)], content)


def get_backend_template_loader() -> TemplateLoader:
    """Get a template loader for backend templates"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    return TemplateLoader(template_dir)
