"""Tests for src/config/settings.py."""
import dataclasses
from pathlib import Path

import pytest

from src.config.settings import CONFIG, ProjectConfig


def test_config_is_frozen() -> None:
    """Config dataclasses should be immutable."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.paths.data_dir = Path("/tmp")


def test_project_root_exists() -> None:
    """PROJECT_ROOT should point to a real directory."""
    assert CONFIG.paths.data_dir.parent.parent.exists()


def test_all_paths_are_pathlib() -> None:
    """All path fields should be pathlib.Path instances."""
    for field_name in vars(CONFIG.paths):
        assert isinstance(getattr(CONFIG.paths, field_name), Path)
