"""Unit tests of the generator functions, without a build."""

import pytest

from act_backend.template_loader import fill_placeholders


def test_fill_placeholders():
    source = 'enum E {\n  A,\n  // {{VARIANTS}}\n}\nf(x, {/* {{NAMES}} */});\n{{RULES}}\n'
    values = {'VARIANTS': '  B,\n  C,', 'NAMES': '"B", "C"', 'RULES': 'r: a => b'}
    # A comment line goes whole, its indentation included; a block comment and a bare name in place
    filled = 'enum E {\n  A,\n  B,\n  C,\n}\nf(x, {"B", "C"});\nr: a => b\n'
    assert fill_placeholders(source, values, 'e.rs') == filled


def test_placeholder_without_value_refused():
    source = '// {{VARIANTS}}\n/* {{NAMES}} */\n'
    with pytest.raises(ValueError, match=r'^e\.rs: no value for the placeholder NAMES$'):
        fill_placeholders(source, {'VARIANTS': ''}, 'e.rs')


def test_value_without_placeholder_refused():
    with pytest.raises(ValueError, match=r'^e\.rs: no placeholder for the value NAMES$'):
        fill_placeholders('// {{VARIANTS}}\n', {'VARIANTS': '', 'NAMES': ''}, 'e.rs')
