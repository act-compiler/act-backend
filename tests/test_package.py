"""Tests of the installed package: what its wheel holds and what builds it."""

import os
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import distribution

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_wheel_holds_the_package_alone():
    assert distribution('act-backend').read_text('top_level.txt').split() == ['act_backend']


def test_wheel_builds_with_the_least_setuptools(tmp_path):
    # The floor the build declares, as the one version pip may take
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as f:
        (floor,) = tomllib.load(f)['build-system']['requires']
    constraint = tmp_path / 'build.txt'
    constraint.write_text(floor.replace('>=', '==') + '\n')

    # A copy of what the wheel is built from, so the tree keeps no build files
    source = tmp_path / 'source'
    source.mkdir()
    for name in ['pyproject.toml', 'README.md', 'LICENSE']:
        shutil.copy(os.path.join(ROOT, name), source)
    shutil.copytree(os.path.join(ROOT, 'act_backend'), source / 'act_backend')

    subprocess.run(
        [sys.executable, '-m', 'pip', 'wheel', str(source), '--no-deps', '-q']
        + ['-w', str(tmp_path / 'wheel'), '--build-constraint', str(constraint)],
        check=True,
    )
