"""
Targeted Acceptance Tests for Remediation Findings R01 & R08.
Verifies:
- Byte-level cryptographic SHA-256 provenance verification (R01)
- Rejection of arbitrary train_transaction.csv, missing files, synthetic fixtures (R01)
- Rejection of contradictory metadata (data_kind=synthetic carrying IEEE metadata) (R01)
- Support for cryptographically traced derived subsets (R01)
- Clean no-model and invalid-path bootstrap without process crash (R08)
- HTTP 503 service unavailable on scoring when unready (R08)
- Production-mode rejection of insecure default credentials (R08)
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.provenance import (
    GENUINE_OBSERVED,
    SIMULATOR_GENERATED,
    SYNTHETIC,
    UNVERIFIED,
    STATUS_VERIFIED,
    STATUS_NON_GENUINE,
    STATUS_HASH_MISMATCH,
    STATUS_IDENTITY_MISMATCH,
    STATUS_IDENTITY_MISSING,
    STATUS_MISSING_FILE,
    DATASET_REGISTRY,
    compute_file_sha256,
    create_derived_subset_manifest,
    create_verified_test_double_provenance,
    get_dataset_provenance,
    verify_bundle_provenance,
)
from src.artifacts import ModelBundle
from src.config import Settings, API_KEY
from src import api


# =========================================================================
# Finding R01 Tests: Provenance & Cryptographic Verification
# =========================================================================

def test_arbitrary_train_transaction_fails_verification(tmp_path):
    """An arbitrary CSV named train_transaction.csv MUST fail genuine verification due to hash mismatch."""
    arbitrary_file = tmp_path / "train_transaction.csv"
    arbitrary_file.write_text("TransactionID,TransactionDT,isFraud\n1001,86400,0\n")

    prov = get_dataset_provenance("IEEE-CIS", file_path=arbitrary_file)
    assert prov["classification"] == UNVERIFIED
    assert prov["acceptable_for_proposal_benchmark"] is False
    assert prov["verification_status"] == STATUS_HASH_MISMATCH
    assert "does not match approved reference digest" in prov["exclusion_reason"]


def test_missing_file_fails_verification(tmp_path):
    """A missing file path MUST fail genuine verification with EXCLUDED_FILE_NOT_FOUND."""
    missing_file = tmp_path / "nonexistent_train_transaction.csv"

    prov = get_dataset_provenance("IEEE-CIS", file_path=missing_file)
    assert prov["classification"] == UNVERIFIED
    assert prov["acceptable_for_proposal_benchmark"] is False
    assert prov["verification_status"] == STATUS_MISSING_FILE
    assert "does not exist" in prov["exclusion_reason"]


def test_ieee_shaped_generated_fixture_fails_verification(tmp_path):
    """A generated IEEE-shaped fixture without official Kaggle bytes MUST fail genuine verification."""
    fixture_file = tmp_path / "ieee_shaped_fixture.csv"
    lines = ["TransactionID,TransactionDT,TransactionAmt,ProductCD,card1,isFraud"]
    for i in range(100):
        lines.append(f"{i},{86400 + i * 60},100.0,W,1001,{1 if i % 10 == 0 else 0}")
    fixture_file.write_text("\n".join(lines))

    prov = get_dataset_provenance("IEEE-CIS", file_path=fixture_file)
    assert prov["acceptable_for_proposal_benchmark"] is False
    assert prov["verification_status"] == STATUS_HASH_MISMATCH


def test_missing_provenance_package_not_ready():
    """A model package missing verified provenance metadata MUST NOT be ready."""
    mock_model = MagicMock()
    mock_model.trained = True
    mock_prep = MagicMock()

    bundle_no_prov = ModelBundle(
        mock_model, mock_prep, {"run_id": "test", "data_kind": "real"}
    )
    assert bundle_no_prov.ready is False

    is_verified, reason = verify_bundle_provenance(bundle_no_prov.metadata)
    assert is_verified is False
    assert "Missing verified provenance metadata" in reason


def test_explicitly_synthetic_package_carrying_ieee_metadata_cannot_be_ready():
    """
    An explicitly synthetic package cannot become READY by carrying IEEE registry metadata.
    Scoring endpoint must return HTTP 503 unavailable.
    """
    mock_model = MagicMock()
    mock_model.trained = True
    mock_prep = MagicMock()
    mock_prep.required_input_columns = ["feat1"]
    mock_prep.transform.return_value = MagicMock()

    # Contradictory bundle: data_kind is synthetic, but provenance claims IEEE-CIS
    contradictory_meta = {
        "run_id": "test_run",
        "data_kind": "synthetic",
        "provenance": create_verified_test_double_provenance("test_run"),
    }
    bundle = ModelBundle(mock_model, mock_prep, contradictory_meta)

    # 1. Bundle.ready must be strictly False
    assert bundle.ready is False

    # 2. verify_bundle_provenance must explicitly reject it
    is_verified, reason = verify_bundle_provenance(bundle.metadata)
    assert is_verified is False
    assert "synthetic" in reason

    # 3. Serving via API must return HTTP 503 Service Unavailable
    buffer = api.ModelDoubleBuffer()
    buffer.atomic_swap(bundle)
    
    # Test client using this buffer
    with tempfile.TemporaryDirectory() as td:
        store = api.PredictionStore(Path(td) / "test.sqlite")
        client = TestClient(api.app, headers={"X-API-Key": API_KEY})
        # Temporarily swap model_buffer
        orig_buf = api.model_buffer
        orig_ledger = api.ledger
        try:
            api.model_buffer = buffer
            api.ledger = store
            resp = client.post("/v1/score", json={"trans_id": "tx1", "features": {"feat1": 1.0}})
            assert resp.status_code == 503
            assert "Service Unavailable" in resp.json()["detail"]
        finally:
            api.model_buffer = orig_buf
            api.ledger = orig_ledger


def test_transaction_identity_hash_mismatch(tmp_path, monkeypatch):
    """If transaction hash matches but identity file hash does not, verification MUST fail."""
    tx_file = tmp_path / "mock_tx.csv"
    id_file = tmp_path / "mock_id.csv"
    tx_file.write_text("tx")
    id_file.write_text("id_wrong")

    # Mock official transaction hash to match tx_file
    tx_hash = compute_file_sha256(tx_file)
    expected_id_hash = "correct_id_hash_12345"
    registry_mock = {
        "IEEE-CIS": {
            "classification": GENUINE_OBSERVED,
            "provider": "Vesta Corporation",
            "source_uri": "https://kaggle.com",
            "description": "Mocked registry",
            "reference_sha256": {
                "train_transaction.csv": tx_hash,
                "train_identity.csv": expected_id_hash,
            },
            "acceptable_for_proposal_benchmark": True,
        }
    }
    monkeypatch.setattr("src.provenance.DATASET_REGISTRY", registry_mock)

    prov = get_dataset_provenance("IEEE-CIS", file_path=tx_file, identity_path=id_file)
    assert prov["acceptable_for_proposal_benchmark"] is False
    assert prov["verification_status"] == STATUS_IDENTITY_MISMATCH
    assert "Identity file SHA-256 mismatch" in prov["exclusion_reason"]


def test_identity_missing_when_specified(tmp_path, monkeypatch):
    """If identity path is specified but does not exist, verification MUST fail."""
    tx_file = tmp_path / "mock_tx.csv"
    tx_file.write_text("tx")
    tx_hash = compute_file_sha256(tx_file)

    registry_mock = {
        "IEEE-CIS": {
            "classification": GENUINE_OBSERVED,
            "provider": "Vesta Corporation",
            "source_uri": "https://kaggle.com",
            "description": "Mocked registry",
            "reference_sha256": {
                "train_transaction.csv": tx_hash,
                "train_identity.csv": "abc",
            },
            "acceptable_for_proposal_benchmark": True,
        }
    }
    monkeypatch.setattr("src.provenance.DATASET_REGISTRY", registry_mock)

    prov = get_dataset_provenance(
        "IEEE-CIS", file_path=tx_file, identity_path=tmp_path / "absent_id.csv"
    )
    assert prov["acceptable_for_proposal_benchmark"] is False
    assert prov["verification_status"] == STATUS_IDENTITY_MISSING


def test_approved_genuine_inputs_succeed(tmp_path, monkeypatch):
    """When both transaction and identity files match approved reference hashes, verification succeeds."""
    tx_file = tmp_path / "train_transaction.csv"
    id_file = tmp_path / "train_identity.csv"
    tx_file.write_text("approved_tx_bytes")
    id_file.write_text("approved_id_bytes")

    tx_hash = compute_file_sha256(tx_file)
    id_hash = compute_file_sha256(id_file)

    registry_mock = {
        "IEEE-CIS": {
            "classification": GENUINE_OBSERVED,
            "provider": "Vesta Corporation / IEEE CIS",
            "source_uri": "https://kaggle.com",
            "description": "Official genuine dataset",
            "reference_sha256": {
                "train_transaction.csv": tx_hash,
                "train_identity.csv": id_hash,
            },
            "acceptable_for_proposal_benchmark": True,
        }
    }
    monkeypatch.setattr("src.provenance.DATASET_REGISTRY", registry_mock)

    prov = get_dataset_provenance("IEEE-CIS", file_path=tx_file, identity_path=id_file)
    assert prov["classification"] == GENUINE_OBSERVED
    assert prov["verification_status"] == STATUS_VERIFIED
    assert prov["acceptable_for_proposal_benchmark"] is True


def test_correctly_traced_derived_subset_succeeds(tmp_path, monkeypatch):
    """A cryptographically traced derived subset with valid parent link succeeds verification."""
    child_file = tmp_path / "derived_subset.csv"
    child_file.write_text("derived_child_records")
    child_hash = compute_file_sha256(child_file)

    official_parent_hash = "3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642"

    manifest = create_derived_subset_manifest(
        parent_sha256=official_parent_hash,
        child_path=child_file,
        transformation_spec={"sample_rate": 0.1, "seed": 42},
        description="10% deterministic sample of IEEE-CIS",
    )
    assert manifest["child_sha256"] == child_hash
    assert manifest["parent_sha256"] == official_parent_hash

    prov = get_dataset_provenance(
        "IEEE-CIS", file_path=child_file, derived_manifest=manifest
    )
    assert prov["classification"] == GENUINE_OBSERVED
    assert prov["verification_status"] == STATUS_VERIFIED
    assert prov["acceptable_for_proposal_benchmark"] is True
    assert "derived_subset" in prov
    assert prov["derived_subset"]["parent_sha256"] == official_parent_hash


def test_paysim_and_feedzai_excluded_from_research_benchmark():
    """PaySim (simulator) and Feedzai BAF (synthetic) must be strictly excluded from research benchmarks."""
    paysim_prov = get_dataset_provenance("PaySim")
    assert paysim_prov["classification"] == SIMULATOR_GENERATED
    assert paysim_prov["acceptable_for_proposal_benchmark"] is False
    assert paysim_prov["verification_status"] == STATUS_NON_GENUINE

    feedzai_prov = get_dataset_provenance("Feedzai-BAF")
    assert feedzai_prov["classification"] == SYNTHETIC
    assert feedzai_prov["acceptable_for_proposal_benchmark"] is False
    assert feedzai_prov["verification_status"] == STATUS_NON_GENUINE


# =========================================================================
# Finding R08 Tests: Clean Bootstrap & Startup Robustness
# =========================================================================

def test_no_model_startup_clean_unready():
    """When no model bundle path is set, API starts cleanly in UNREADY state and returns 503 on scoring."""
    buffer = api.ModelDoubleBuffer()  # Fresh unready buffer
    assert buffer.is_ready is False

    with tempfile.TemporaryDirectory() as td:
        store = api.PredictionStore(Path(td) / "test.sqlite")
        client = TestClient(api.app, headers={"X-API-Key": API_KEY})
        orig_buf = api.model_buffer
        orig_ledger = api.ledger
        try:
            api.model_buffer = buffer
            api.ledger = store

            # /health returns NOT_READY
            health = client.get("/health").json()
            assert health["status"] == "NOT_READY"
            assert health["is_ready"] is False
            assert health["model_loaded"] is False

            # /v1/score returns 503
            resp = client.post("/v1/score", json={"trans_id": "tx1", "features": {"feat1": 1.0}})
            assert resp.status_code == 503
            assert "Service Unavailable" in resp.json()["detail"]

            # /v1/schema returns 503
            schema_resp = client.get("/v1/schema")
            assert schema_resp.status_code == 503
        finally:
            api.model_buffer = orig_buf
            api.ledger = orig_ledger


def test_invalid_model_path_startup_clean(monkeypatch, tmp_path):
    """When MODEL_BUNDLE_PATH points to a non-existent path, startup does not crash with FileNotFoundError."""
    non_existent = str(tmp_path / "non_existent_model_bundle_dir")
    monkeypatch.setattr(api.settings, "model_bundle_path", non_existent)

    # Calling initialize_app_model() must execute cleanly without raising an unhandled exception
    api.initialize_app_model()
    # Model buffer remains unready
    assert api.model_buffer.is_ready is False


def test_production_rejection_of_default_credentials():
    """In production and staging environments, default test API keys and insecure keys MUST be rejected."""
    insecure_keys = ["dev-secret-key", "test-key-123", "admin-key-123", "secret", "short"]

    for key in insecure_keys:
        with pytest.raises(ValueError, match="Insecure API key configured in 'production'"):
            Settings(app_env="production", api_key=key)

        with pytest.raises(ValueError, match="Insecure API key configured in 'staging'"):
            Settings(app_env="staging", api_key=key)


def test_production_rejection_of_empty_db_password_when_mysql_enabled():
    """In production with MySQL enabled, empty db_password MUST be rejected."""
    with pytest.raises(ValueError, match="Empty database password configured"):
        Settings(
            app_env="production",
            api_key="a_very_secure_and_long_production_key_32chars",
            enable_mysql=True,
            db_password="",
        )


def test_production_valid_credentials_accepted():
    """In production with strong 32-character API key, configuration validation succeeds."""
    secure_settings = Settings(
        app_env="production",
        api_key="production_super_secure_key_1234567890_abc",
        enable_mysql=False,
    )
    assert secure_settings.app_env == "production"
    assert len(secure_settings.api_key) >= 16
