"""
Configuration manager for the Concept-Drift-Aware Fraud Detection Platform.
Uses Pydantic BaseSettings for type-safe environment variable parsing,
validation rules, and deterministic directory structure management.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Validated configuration settings loaded from environment variables and .env file.
    Enforces type constraints, valid port ranges, and probability bounds.
    """
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Database Configuration (MySQL Server 9.7)
    db_host: str = Field(default="127.0.0.1", description="Database host IP or hostname")
    db_port: int = Field(default=3306, ge=1024, le=65535, description="Database port number")
    db_user: str = Field(default="root", description="Database username")
    db_password: str = Field(default="", description="Database password")
    db_name: str = Field(default="fraud_drift_db", description="Target database schema name")

    # Hardware Acceleration Configuration (NVIDIA RTX 5080 eGPU)
    use_gpu: bool = Field(default=True, description="Enable CUDA acceleration for XGBoost")
    gpu_device: str = Field(default="cuda", description="Target GPU device string")

    # Experiment Reproducibility
    random_seed: int = Field(default=42, ge=0, description="Master deterministic random seed")

    # Temporal Slicing & Window Configuration
    default_hours_per_week: int = Field(default=168, gt=0, description="Hours per temporal slice")
    total_simulation_weeks: int = Field(default=24, gt=0, description="Default stream slice count")
    training_window_ratio: float = Field(
        default=0.20, gt=0.0, lt=1.0,
        description="Fraction of initial stream reserved for baseline training (e.g. 0.20 = Weeks 1-5)"
    )

    # Drift Detector Hyperparameters
    adwin_delta: float = Field(default=0.002, gt=0.0, lt=0.1, description="ADWIN confidence delta")
    ddm_min_instances: int = Field(default=30, ge=5, description="DDM minimum sample warmup count")
    ddm_warning_level: float = Field(default=2.0, gt=1.0, description="DDM warning threshold (std)")
    ddm_drift_level: float = Field(default=3.0, gt=2.0, description="DDM drift trigger threshold (std)")

    @field_validator("ddm_drift_level")
    @classmethod
    def validate_ddm_levels(cls, v: float, info) -> float:
        warning_level = info.data.get("ddm_warning_level", 2.0)
        if v <= warning_level:
            raise ValueError(f"ddm_drift_level ({v}) must be strictly greater than ddm_warning_level ({warning_level})")
        return v


# Instantiate validated settings singleton
settings = Settings()

# Directory Paths Management
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Backward-Compatible Module-Level Constants
DB_HOST = settings.db_host
DB_PORT = settings.db_port
DB_USER = settings.db_user
DB_PASSWORD = settings.db_password
DB_NAME = settings.db_name

USE_GPU = settings.use_gpu
GPU_DEVICE = settings.gpu_device
RANDOM_SEED = settings.random_seed

DEFAULT_HOURS_PER_WEEK = settings.default_hours_per_week
TOTAL_SIMULATION_WEEKS = settings.total_simulation_weeks
TRAINING_WINDOW_RATIO = settings.training_window_ratio

ADWIN_DELTA = settings.adwin_delta
DDM_MIN_INSTANCES = settings.ddm_min_instances
DDM_WARNING_LEVEL = settings.ddm_warning_level
DDM_DRIFT_LEVEL = settings.ddm_drift_level
