"""
Invariant Tests for Pydantic Settings and Configuration Validation (P3).
Asserts that configuration parameters (database ports, probability ratios,
and detector thresholds) enforce type safety and domain constraints at initialization.
"""

import pytest
from pydantic import ValidationError

from src.config import Settings, settings


def test_default_settings_valid():
    """Asserts that current environment and default settings load without error."""
    assert settings.db_port >= 1024
    assert 0.0 < settings.training_window_ratio < 1.0
    assert 0.0 < settings.adwin_delta < 0.1
    assert settings.ddm_drift_level > settings.ddm_warning_level


def test_invalid_db_port_validation():
    """Asserts that out-of-range port numbers raise ValidationError."""
    # Port too large (> 65535)
    with pytest.raises(ValidationError):
        Settings(db_port=70000)

    # Privileged port (< 1024)
    with pytest.raises(ValidationError):
        Settings(db_port=80)


def test_invalid_training_window_ratio():
    """Asserts that training window ratio outside (0, 1) raises ValidationError."""
    with pytest.raises(ValidationError):
        Settings(training_window_ratio=1.5)

    with pytest.raises(ValidationError):
        Settings(training_window_ratio=-0.1)

    with pytest.raises(ValidationError):
        Settings(training_window_ratio=0.0)


def test_ddm_threshold_hierarchy():
    """
    Asserts that ddm_drift_level must be strictly greater than ddm_warning_level.
    """
    # drift level less than warning level
    with pytest.raises(ValidationError):
        Settings(ddm_warning_level=3.0, ddm_drift_level=2.0)

    # drift level equal to warning level
    with pytest.raises(ValidationError):
        Settings(ddm_warning_level=2.5, ddm_drift_level=2.5)


def test_adwin_delta_bounds():
    """Asserts that adwin_delta must be within valid bounds (0.0, 0.1)."""
    with pytest.raises(ValidationError):
        Settings(adwin_delta=0.5)

    with pytest.raises(ValidationError):
        Settings(adwin_delta=0.0)
