"""Tests of the installed package: what its wheel holds."""

from importlib.metadata import distribution


def test_wheel_holds_the_package_alone():
    assert distribution('act-backend').read_text('top_level.txt').split() == ['act_backend']
