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
    assert len(settings.api_key) > 0


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


def test_drift_monitor_wires_ddm_config():
    """Asserts that DriftMonitor wires DDM hyperparameters into river DDM instance."""
    from src.drift_detector import DriftMonitor
    monitor = DriftMonitor(
        ddm_min_instances=50,
        ddm_warning_level=2.2,
        ddm_drift_level=3.3
    )
    assert monitor.ddm.warm_start == 50
    assert monitor.ddm.warning_threshold == 2.2
    assert monitor.ddm.drift_threshold == 3.3

    monitor.reset_detectors()
    assert monitor.ddm.warm_start == 50
    assert monitor.ddm.warning_threshold == 2.2
    assert monitor.ddm.drift_threshold == 3.3


def test_drift_monitor_reset_behavior():
    """Asserts that reset_detectors() clears accumulated drift statistics after adaptation."""
    from src.drift_detector import DriftMonitor
    monitor = DriftMonitor()

    # Feed errors to accumulate detector window state
    for i in range(100):
        monitor.update(y_true=1, y_pred=0, slice_index=0, step_number=i, trans_id=f"tx_{i}")

    assert monitor.adwin.width > 0

    # Reset detectors after model adaptation
    old_adwin = monitor.adwin
    old_ddm = monitor.ddm
    monitor.reset_detectors()

    assert monitor.adwin is not old_adwin
    assert monitor.ddm is not old_ddm
    assert monitor.adwin.width == 0

