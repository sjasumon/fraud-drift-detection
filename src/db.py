"""
Database management module for storing transactions metadata, drift events,
and model performance evaluation metrics via MySQL Server 9.7 (schema: fraud_drift_db).
"""

import sys
import logging
from urllib.parse import quote_plus
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, Text, ForeignKey, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.engine import Engine

from src.config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
)

logger = logging.getLogger("fraud_drift.db")
Base = declarative_base()


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    run_id = Column(String(64), primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    dataset_name = Column(String(128))
    total_rows = Column(Integer)
    time_slices_count = Column(Integer)
    gpu_accelerated = Column(String(128))
    static_f1_overall = Column(Float)
    adaptive_f1_overall = Column(Float)
    drift_alerts_count = Column(Integer)
    db_backend = Column(String(64))
    notes = Column(Text, nullable=True)

    slices = relationship("TimeSliceRecord", back_populates="experiment", cascade="all, delete-orphan")
    drift_events = relationship("DriftEventRecord", back_populates="experiment", cascade="all, delete-orphan")
    metrics = relationship("SliceMetricRecord", back_populates="experiment", cascade="all, delete-orphan")
    attributions = relationship("DriftAttributionRecord", back_populates="experiment", cascade="all, delete-orphan")


class TimeSliceRecord(Base):
    __tablename__ = "time_slices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("experiment_runs.run_id"))
    slice_index = Column(Integer, nullable=False)
    slice_name = Column(String(64))
    start_step = Column(Integer)
    end_step = Column(Integer)
    total_transactions = Column(Integer)
    fraud_transactions = Column(Integer)
    fraud_ratio = Column(Float)

    experiment = relationship("ExperimentRun", back_populates="slices")


class DriftEventRecord(Base):
    __tablename__ = "drift_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("experiment_runs.run_id"))
    slice_index = Column(Integer)
    step_number = Column(Integer)
    detector_type = Column(String(64))  # e.g. ADWIN, DDM
    error_rate = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    status = Column(String(32))  # WARNING or DRIFT
    notes = Column(Text, nullable=True)

    experiment = relationship("ExperimentRun", back_populates="drift_events")


class SliceMetricRecord(Base):
    __tablename__ = "slice_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("experiment_runs.run_id"))
    slice_index = Column(Integer)
    slice_name = Column(String(64))
    model_type = Column(String(32))  # 'static' or 'adaptive'
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_pr = Column(Float)

    experiment = relationship("ExperimentRun", back_populates="metrics")


class DriftAttributionRecord(Base):
    __tablename__ = "drift_attributions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("experiment_runs.run_id"))
    slice_index = Column(Integer, nullable=False)
    detector_type = Column(String(32), default="ADWIN")
    feature_name = Column(String(64), nullable=False)
    psi_score = Column(Float, nullable=False)
    shap_importance = Column(Float, nullable=False)
    attribution_score = Column(Float, nullable=False)
    rank_order = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    experiment = relationship("ExperimentRun", back_populates="attributions")


class DatabaseManager:
    """Manages database connectivity and persistence for experimental runs via MySQL Server 9.7."""

    def __init__(self):
        self.engine: Optional[Engine] = None
        self.session_factory = None
        self.backend_name = "MySQL"
        self._setup_connection()

    def _setup_connection(self):
        try:
            encoded_user = quote_plus(DB_USER)
            encoded_pw = quote_plus(DB_PASSWORD)

            # Strictly validate DB_NAME identifier against SQL injection before DDL execution
            import re
            if not re.match(r"^[a-zA-Z0-9_]+$", DB_NAME):
                raise ValueError(f"Insecure or invalid database name identifier: {DB_NAME!r}")

            # Connect to MySQL Server and create database if not exists
            server_url = f"mysql+pymysql://{encoded_user}:{encoded_pw}@{DB_HOST}:{DB_PORT}"
            temp_engine = create_engine(server_url, connect_args={"connect_timeout": 5})
            with temp_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`"))
            temp_engine.dispose()

            # Connect to fraud_drift_db
            db_url = f"mysql+pymysql://{encoded_user}:{encoded_pw}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            self.engine = create_engine(db_url, pool_pre_ping=True)
            Base.metadata.create_all(self.engine)
            self.session_factory = sessionmaker(bind=self.engine)
            self.backend_name = f"MySQL Server 9.7 (`{DB_NAME}` on {DB_HOST}:{DB_PORT})"
            logger.info(f"Connected to MySQL: {self.backend_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL database at {DB_HOST}:{DB_PORT}: {e}")
            raise ConnectionError(
                f"MySQL connection error ({e}). Please ensure MySQL Server 9.7 is running on {DB_HOST}:{DB_PORT}."
            )

    def get_session(self):
        return self.session_factory()

    def ensure_run_exists(self, run_id: str, dataset_name: str = "IEEE-CIS", total_rows: int = 0):
        with self.get_session() as session:
            existing = session.query(ExperimentRun).filter_by(run_id=run_id).first()
            if not existing:
                run = ExperimentRun(
                    run_id=run_id,
                    dataset_name=dataset_name,
                    total_rows=total_rows,
                    db_backend=self.backend_name,
                    notes="Initialized run"
                )
                session.add(run)
                session.commit()

    def log_run(self, run_data: Dict[str, Any]):
        with self.get_session() as session:
            run = ExperimentRun(
                run_id=run_data["run_id"],
                dataset_name=run_data.get("dataset_name", "IEEE-CIS"),
                total_rows=run_data.get("total_rows", 0),
                time_slices_count=run_data.get("time_slices_count", 0),
                gpu_accelerated=run_data.get("gpu_accelerated", "cuda"),
                static_f1_overall=run_data.get("static_f1_overall", 0.0),
                adaptive_f1_overall=run_data.get("adaptive_f1_overall", 0.0),
                drift_alerts_count=run_data.get("drift_alerts_count", 0),
                db_backend=self.backend_name,
                notes=run_data.get("notes", "")
            )
            session.merge(run)
            session.commit()

    def log_slices(self, run_id: str, slices: List[Dict[str, Any]]):
        self.ensure_run_exists(run_id)
        with self.get_session() as session:
            for s in slices:
                rec = TimeSliceRecord(
                    run_id=run_id,
                    slice_index=s["slice_index"],
                    slice_name=s.get("slice_name", f"Week_{s['slice_index']+1}"),
                    start_step=s.get("start_step", 0),
                    end_step=s.get("end_step", 0),
                    total_transactions=s.get("total_transactions", 0),
                    fraud_transactions=s.get("fraud_transactions", 0),
                    fraud_ratio=s.get("fraud_ratio", 0.0),
                )
                session.add(rec)
            session.commit()

    def log_drift_events(self, run_id: str, events: List[Dict[str, Any]]):
        self.ensure_run_exists(run_id)
        with self.get_session() as session:
            for ev in events:
                rec = DriftEventRecord(
                    run_id=run_id,
                    slice_index=ev.get("slice_index"),
                    step_number=ev.get("step_number"),
                    detector_type=ev.get("detector_type"),
                    error_rate=ev.get("error_rate"),
                    confidence_score=ev.get("confidence_score"),
                    status=ev.get("status", "DRIFT"),
                    notes=ev.get("notes", "")
                )
                session.add(rec)
            session.commit()

    def log_metrics(self, run_id: str, metrics: List[Dict[str, Any]]):
        self.ensure_run_exists(run_id)
        with self.get_session() as session:
            for m in metrics:
                rec = SliceMetricRecord(
                    run_id=run_id,
                    slice_index=m["slice_index"],
                    slice_name=m.get("slice_name", f"Slice_{m['slice_index']+1}"),
                    model_type=m["model_type"],
                    precision=m.get("precision", 0.0),
                    recall=m.get("recall", 0.0),
                    f1_score=m.get("f1_score", 0.0),
                    auc_pr=m.get("auc_pr", 0.0)
                )
                session.add(rec)
            session.commit()

    def log_attributions(self, run_id: str, attributions: List[Dict[str, Any]]):
        self.ensure_run_exists(run_id)
        with self.get_session() as session:
            for attr in attributions:
                rec = DriftAttributionRecord(
                    run_id=run_id,
                    slice_index=attr["slice_index"],
                    detector_type=attr.get("detector_type", "ADWIN"),
                    feature_name=attr["feature_name"],
                    psi_score=float(attr.get("psi_score", 0.0)),
                    shap_importance=float(attr.get("shap_importance", 0.0)),
                    attribution_score=float(attr.get("attribution_score", 0.0)),
                    rank_order=int(attr.get("rank_order", 1)),
                )
                session.add(rec)
            session.commit()
