"""The templates a backend's sources are generated from, loaded and filled"""

import os
import re

# A fragment's key
KEY = re.compile(r'\{\{(\w+)\}\}')
# The formatters' line width, as rustfmt.toml and .clang-format set it
LINE_WIDTH = 100
# rustfmt's fn_call_width: the widest a call's arguments stay on one line
CALL_WIDTH = 60
# A template's twins, in the order the formatters prefer them
TWINS = ('wrapped', 'aligned', 'stacked', 'vertical', 'stacked_vertical')


def fits(text: str) -> bool:
    """Whether every line of text fits the formatters' line width"""
    return all(len(line) <= LINE_WIDTH for line in text.split('\n'))


def packed(lead: str, items: list[str], tail: str, column: int) -> list[str]:
    """Items after lead, comma-separated, as many to a line as fit, each new line at column"""
    lines = [lead]
    for k, item in enumerate(items):
        item += tail if k == len(items) - 1 else ','
        gap = '' if lines[-1].endswith('(') or lines[-1].isspace() else ' '
        if fits(lines[-1] + gap + item):
            lines[-1] += gap + item
        else:
            lines.append(' ' * column + item)
    return lines


def call_lines(lead: str, items: list[str], tail: str, indent: int, declaration=False) -> str:
    """A C++ call as clang-format lays it out: on one line, else its arguments packed

    Aligned after the parenthesis lead ends in, else below it on lines of indent plus four,
    where aligned does not fit or, in a call, takes more lines.
    """
    one = lead + ', '.join(items) + tail
    if fits(one):
        return one
    aligned = packed(lead, items, tail, len(lead))
    below = [lead, *packed(' ' * (indent + 4), items, tail, indent + 4)]
    # A declaration stays aligned wherever it fits
    if fits('\n'.join(aligned)) and (declaration or len(aligned) <= len(below)):
        return '\n'.join(aligned)
    return '\n'.join(below)


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
        """Render a template, else the first of its twins, in the order of TWINS, whose lines fit

        The last twin where none fits.
        """
        stem = template_file.removesuffix('.txt')
        twins = [f'{stem}_{twin}.txt' for twin in TWINS]
        twins = [twin for twin in twins if os.path.isfile(os.path.join(self.template_dir, twin))]
        text = self.render(template_file, **kwargs)
        for twin in twins:
            if fits(text):
                break
            text = self.render(twin, **kwargs)
        return text

    def render_arm(self, template_file: str, **kwargs) -> str:
        """Render a match arm, else its _wrapped twin where its first line leaves no column

        rustfmt keeps that column for the comma ending the arm.
        """
        text = self.render(template_file, **kwargs)
        if fits(text.split('\n')[0] + ','):
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
