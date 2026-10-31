"""Shared fixtures.

These tests make **no API calls**. Everything runs against the committed dataset
and the committed submission, so `pytest` is safe offline and in CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"

# The modules import each other by bare name (`from schemas import Decision`),
# so `code/` goes on the path rather than being treated as a package.
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def dataset():
    from data_io import Dataset

    return Dataset.load(REPO_ROOT / "dataset")
