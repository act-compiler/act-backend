"""The templates a backend's sources are generated from, loaded and filled"""

import os
import re

# A fragment's key
KEY = re.compile(r'\{\{(\w+)\}\}')
# The formatters' line width, as rustfmt.toml and .clang-format set it
LINE_WIDTH = 100


class TemplateLoader:
    """Manages loading and caching of templates for code generation"""

    def __init__(self, template_dir: str):
        """Read templates from template_dir, caching each file"""
        self.template_dir = template_dir
        self.cache: dict[str, str] = {}

    def load(self, template_file: str) -> str:
        """Return a template file's content, its final newline dropped"""
        if template_file not in self.cache:
            file_path = os.path.join(self.template_dir, template_file)
            with open(file_path, 'r') as f:
                # The final newline ends the file, not the fragment
                self.cache[template_file] = f.read().removesuffix('\n')
        return self.cache[template_file]

    def render(self, template_file: str, **kwargs) -> str:
        """Render a template with the given values of its keys

        Raises ValueError naming the template when a key has no value or a value no key.
        """
        template = self.load(template_file)
        keys = set(KEY.findall(template))
        unfilled = sorted(keys - kwargs.keys())
        if unfilled:
            raise ValueError(f'{template_file}: no value for the key {", ".join(unfilled)}')
        unknown = sorted(kwargs.keys() - keys)
        if unknown:
            raise ValueError(f'{template_file}: no key for the value {", ".join(unknown)}')
        return KEY.sub(lambda match: str(kwargs[match.group(1)]), template)

    def render_lines(self, template_file: str, **kwargs) -> str:
        """Render a template as whole lines, the last one ended with a newline"""
        return self.render(template_file, **kwargs) + '\n'

    def render_fitting(self, template_file: str, **kwargs) -> str:
        """Render a template, or its _wrapped twin where a line would not fit"""
        text = self.render(template_file, **kwargs)
        if all(len(line) <= LINE_WIDTH for line in text.split('\n')):
            return text
        return self.render(template_file.removesuffix('.txt') + '_wrapped.txt', **kwargs)


# A placeholder: a comment line, inline block comment, or bare
PLACEHOLDER = re.compile(
    r'(?P<above>^\n)?^[ \t]*// \{\{(?P<line>[A-Z0-9_]+)\}\}\n'
    r'|/\* \{\{(?P<block>[A-Z0-9_]+)\}\} \*/|\{\{(?P<bare>[A-Z0-9_]+)\}\}',
    re.MULTILINE,
)


def placeholder_name(match):
    """Return the name a placeholder holds, whatever its form"""
    return match.group('line') or match.group('block') or match.group('bare')


def fill_placeholders(content: str, replacements: dict[str, str], source: str) -> str:
    """
    Replace each placeholder of a skeleton with its generated code

    Raises ValueError naming `source` when a placeholder has no value or a value no placeholder.
    """
    names = {placeholder_name(match) for match in PLACEHOLDER.finditer(content)}
    unfilled = sorted(names - replacements.keys())
    if unfilled:
        raise ValueError(f'{source}: no value for the placeholder {", ".join(unfilled)}')
    unknown = sorted(replacements.keys() - names)
    if unknown:
        raise ValueError(f'{source}: no placeholder for the value {", ".join(unknown)}')

    def code(match):
        value = replacements[placeholder_name(match)]
        if not match.group('line'):
            return value
        # With no code, drop the line and a blank above
        if not value:
            return ''
        return (match.group('above') or '') + value + '\n'

    return PLACEHOLDER.sub(code, content)


def get_backend_template_loader() -> TemplateLoader:
    """Get a template loader for backend templates"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    return TemplateLoader(template_dir)
