"""
Dataset provenance and authenticity verification module.
Strictly distinguishes genuine observed transactions from simulator-generated or synthetic data.
Enforces cryptographic SHA-256 byte-level verification against approved provider reference records.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

# Classification constants
GENUINE_OBSERVED = "GENUINE_OBSERVED"
SIMULATOR_GENERATED = "SIMULATOR_GENERATED"
SYNTHETIC = "SYNTHETIC"
UNVERIFIED = "UNVERIFIED"

# Verification status constants
STATUS_VERIFIED = "VERIFIED_RESEARCH_DATA"
STATUS_NON_GENUINE = "EXCLUDED_NON_GENUINE"
STATUS_HASH_MISMATCH = "EXCLUDED_HASH_MISMATCH"
STATUS_IDENTITY_MISMATCH = "EXCLUDED_IDENTITY_HASH_MISMATCH"
STATUS_IDENTITY_MISSING = "EXCLUDED_IDENTITY_MISSING"
STATUS_MISSING_FILE = "EXCLUDED_FILE_NOT_FOUND"
STATUS_UNVERIFIED = "UNVERIFIED"
STATUS_CONTRADICTORY = "EXCLUDED_CONTRADICTORY_METADATA"

# Known registry with verified provider sources and reference SHA-256 digests
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "IEEE-CIS": {
        "classification": GENUINE_OBSERVED,
        "provider": "Vesta Corporation / IEEE Computational Intelligence Society",
        "source_uri": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
        "description": "Genuine observed real-world e-commerce card-not-present transactions",
        "reference_sha256": {
            "train_transaction.csv": "3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642",
            "train_identity.csv": "b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c",
        },
        "feedback_latency_nature": "UNKNOWN_CONFIRMATION_TIME (offline observation window; no live dispute timestamps provided)",
        "acceptable_for_proposal_benchmark": True,
    },
    "PaySim": {
        "classification": SIMULATOR_GENERATED,
        "provider": "Lopez-Rojas et al. / PaySim Mobile Money Simulator",
        "source_uri": "https://github.com/EdgarLopezPhD/PaySim",
        "description": "Simulator-generated financial agent-based model; NOT genuine real data",
        "acceptable_for_proposal_benchmark": False,
        "exclusion_reason": "Simulator-generated transactions do not represent genuine fraud distributions for PhD proposal evidence",
    },
    "Feedzai-BAF": {
        "classification": SYNTHETIC,
        "provider": "Feedzai / NeurIPS 2022 Bank Account Fraud Dataset Suite",
        "source_uri": "https://github.com/feedzai/bank-account-fraud",
        "description": "Synthetic fraud dataset derived from bank application distributions; NOT genuine observed transactions",
        "acceptable_for_proposal_benchmark": False,
        "exclusion_reason": "Synthetic dataset variant does not establish real-world transaction accuracy",
    },
}


def compute_file_sha256(path: Union[Path, str]) -> str:
    """Calculates SHA-256 hash of a file efficiently in 8MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest()


def create_derived_subset_manifest(
    parent_sha256: str,
    child_path: Union[Path, str],
    transformation_spec: Union[str, Dict[str, Any]],
    parent_identity_sha256: Optional[str] = None,
    parent_path: Optional[Union[Path, str]] = None,
    description: str = "Verified derived subset of genuine observed transactions",
    verified_lineage: bool = False,
    row_identities: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Constructs a cryptographic manifest linking a derived subset to an approved parent dataset.
    Validates child file existence and computes its SHA-256 digest.
    If parent_path is provided, verifies that parent bytes match parent_sha256.
    Separates verified lineage from self-declared claims.
    """
    child_obj = Path(child_path)
    if not child_obj.exists():
        raise FileNotFoundError(f"Derived child file does not exist: {child_path}")

    child_sha256 = compute_file_sha256(child_obj)
    if isinstance(transformation_spec, dict):
        trans_digest = hashlib.sha256(
            json.dumps(transformation_spec, sort_keys=True).encode("utf-8")
        ).hexdigest()
    else:
        trans_digest = hashlib.sha256(str(transformation_spec).encode("utf-8")).hexdigest()

    is_verified = False
    if parent_path is not None:
        p_obj = Path(parent_path)
        if p_obj.exists() and compute_file_sha256(p_obj) == parent_sha256:
            is_verified = True

    lineage_type = "verified_lineage" if is_verified else "self_declared"

    manifest = {
        "parent_sha256": parent_sha256,
        "child_sha256": child_sha256,
        "parent_identity_sha256": parent_identity_sha256,
        "transformation_digest": trans_digest,
        "description": description,
        "lineage_type": lineage_type,
        "verified": (lineage_type == "verified_lineage"),
    }
    if row_identities:
        manifest["row_identities_sha256"] = hashlib.sha256(
            json.dumps(row_identities, sort_keys=True).encode("utf-8")
        ).hexdigest()
    return manifest



def get_dataset_provenance(
    dataset_type: str,
    file_path: Optional[Union[Path, str]] = None,
    identity_path: Optional[Union[Path, str]] = None,
    explicit_classification: Optional[str] = None,
    derived_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constructs and cryptographically verifies a provenance record for a transaction dataset.
    Requires genuine byte-level hash matches for research benchmark acceptance.
    Unknown or mismatched inputs strictly default to UNVERIFIED or EXCLUDED.
    """
    # Explicit classification override (e.g. software test fixtures declaring synthetic/simulator)
    if explicit_classification in {SIMULATOR_GENERATED, SYNTHETIC, UNVERIFIED}:
        return {
            "dataset_type": dataset_type,
            "classification": explicit_classification,
            "provider": "Test Harness / Simulator",
            "source_uri": "N/A",
            "description": f"Explicitly classified as {explicit_classification}; software test only",
            "feedback_latency_nature": "N/A",
            "acceptable_for_proposal_benchmark": False,
            "verification_status": STATUS_NON_GENUINE if explicit_classification != UNVERIFIED else STATUS_UNVERIFIED,
            "exclusion_reason": f"Explicitly designated as {explicit_classification}",
        }

    registry_entry = DATASET_REGISTRY.get(dataset_type)

    if registry_entry is None:
        return {
            "dataset_type": dataset_type,
            "classification": UNVERIFIED,
            "provider": "Unknown",
            "source_uri": "Unknown",
            "description": "Unregistered or unverified transaction source",
            "feedback_latency_nature": "UNKNOWN",
            "acceptable_for_proposal_benchmark": False,
            "verification_status": STATUS_UNVERIFIED,
            "exclusion_reason": "Unregistered dataset source cannot establish genuine transaction authenticity",
        }

    # Datasets registered as non-genuine (PaySim, Feedzai-BAF)
    if not registry_entry["acceptable_for_proposal_benchmark"]:
        return {
            "dataset_type": dataset_type,
            "classification": registry_entry["classification"],
            "provider": registry_entry["provider"],
            "source_uri": registry_entry["source_uri"],
            "description": registry_entry["description"],
            "feedback_latency_nature": registry_entry.get("feedback_latency_nature", "UNKNOWN"),
            "acceptable_for_proposal_benchmark": False,
            "verification_status": STATUS_NON_GENUINE,
            "exclusion_reason": registry_entry.get("exclusion_reason", "Non-genuine dataset"),
        }

    # For GENUINE_OBSERVED candidates (e.g. IEEE-CIS), cryptographic byte verification is required
    provenance = {
        "dataset_type": dataset_type,
        "classification": UNVERIFIED,  # Defaults to UNVERIFIED until byte hash succeeds
        "provider": registry_entry["provider"],
        "source_uri": registry_entry["source_uri"],
        "description": registry_entry["description"],
        "feedback_latency_nature": registry_entry.get("feedback_latency_nature", "UNKNOWN"),
        "acceptable_for_proposal_benchmark": False,
        "verification_status": STATUS_UNVERIFIED,
    }

    if file_path is None:
        provenance["exclusion_reason"] = "No dataset file provided; cannot verify cryptographic provenance"
        return provenance

    file_obj = Path(file_path)
    if not file_obj.exists():
        provenance["verification_status"] = STATUS_MISSING_FILE
        provenance["exclusion_reason"] = f"Dataset file does not exist: {file_path}"
        return provenance

    actual_sha256 = compute_file_sha256(file_obj)
    provenance["file_name"] = file_obj.name
    provenance["file_size_bytes"] = file_obj.stat().st_size
    provenance["actual_sha256"] = actual_sha256

    expected_hashes = registry_entry.get("reference_sha256", {})
    expected_tx_hash = expected_hashes.get("train_transaction.csv")
    expected_id_hash = expected_hashes.get("train_identity.csv")

    # If identity file is provided on ANY path, cryptographically verify it first
    if identity_path is not None:
        id_obj = Path(identity_path)
        if not id_obj.exists():
            provenance["verification_status"] = STATUS_IDENTITY_MISSING
            provenance["exclusion_reason"] = f"Identity file not found at {identity_path}"
            return provenance

        id_sha256 = compute_file_sha256(id_obj)
        provenance["identity_file_name"] = id_obj.name
        provenance["identity_file_size_bytes"] = id_obj.stat().st_size
        provenance["identity_actual_sha256"] = id_sha256

        target_id_hash = expected_id_hash
        if derived_manifest and derived_manifest.get("parent_identity_sha256"):
            target_id_hash = derived_manifest["parent_identity_sha256"]

        if target_id_hash and id_sha256 != target_id_hash:
            provenance["verification_status"] = STATUS_IDENTITY_MISMATCH
            provenance["exclusion_reason"] = (
                f"Identity file SHA-256 mismatch. Expected {target_id_hash}, got {id_sha256}"
            )
            return provenance

    # Case A: Exact byte match with official provider reference hash
    if actual_sha256 == expected_tx_hash:
        provenance["classification"] = GENUINE_OBSERVED
        provenance["verification_status"] = STATUS_VERIFIED
        provenance["acceptable_for_proposal_benchmark"] = True
        return provenance

    # Case B: Cryptographically traced verified derived subset
    if derived_manifest is not None and isinstance(derived_manifest, dict):
        parent_sha256 = derived_manifest.get("parent_sha256")
        child_sha256 = derived_manifest.get("child_sha256")
        trans_digest = derived_manifest.get("transformation_digest")
        lineage_type = derived_manifest.get("lineage_type", "self_declared")
        is_verified_lineage = bool(derived_manifest.get("verified", False) and lineage_type == "verified_lineage")

        if not is_verified_lineage:
            provenance["verification_status"] = "EXCLUDED_UNVERIFIED_DERIVED_LINEAGE"
            provenance["exclusion_reason"] = (
                "Self-declared derived subset manifest lacks verified parent byte lineage; arbitrary child cannot be accepted"
            )
            return provenance

        if (
            parent_sha256 == expected_tx_hash
            and child_sha256 == actual_sha256
            and trans_digest
        ):
            provenance["classification"] = GENUINE_OBSERVED
            provenance["verification_status"] = STATUS_VERIFIED
            provenance["acceptable_for_proposal_benchmark"] = True
            provenance["derived_subset"] = dict(derived_manifest)
            return provenance
        else:
            provenance["verification_status"] = "EXCLUDED_DERIVED_VERIFICATION_FAILED"
            provenance["exclusion_reason"] = (
                "Derived subset manifest failed cryptographic verification against reference parent or child digest"
            )
            return provenance

    # Case C: File exists but SHA-256 does not match reference and no valid derived manifest provided
    provenance["verification_status"] = STATUS_HASH_MISMATCH
    provenance["exclusion_reason"] = (
        f"File SHA-256 ({actual_sha256[:16]}...) does not match approved reference digest "
        f"({expected_tx_hash[:16]}...) and no verified derived subset manifest was provided"
    )
    return provenance


def create_verified_test_double_provenance(
    run_id: str = "test_run",
    description: str = "Verified test double with isolated mock provenance",
) -> Dict[str, Any]:
    """
    Test harness helper: generates an explicit test double provenance record.
    Used exclusively in unit/integration test suites. Marked non-acceptable for production benchmarks.
    """
    return {
        "dataset_type": "IEEE-CIS",
        "classification": GENUINE_OBSERVED,
        "provider": "Vesta Corporation / IEEE Computational Intelligence Society",
        "source_uri": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
        "description": description,
        "feedback_latency_nature": "UNKNOWN_CONFIRMATION_TIME (test double)",
        "acceptable_for_proposal_benchmark": False,
        "verification_status": STATUS_VERIFIED,
        "test_double": True,
        "test_run_id": run_id,
    }


def verify_bundle_provenance(metadata: Dict[str, Any], allow_test_doubles: bool = False) -> Tuple[bool, str]:
    """
    Validates whether a model bundle was fitted on genuine observed data.
    Strictly rejects missing provenance, non-genuine classifications, unverified statuses,
    test-double markers in production paths, and contradictory metadata.
    Returns (is_verified, reason).
    """
    if not metadata or not isinstance(metadata, dict):
        return False, "Missing bundle metadata"

    # Reject non-genuine data_kind at the metadata level
    data_kind = metadata.get("data_kind")
    if data_kind in {"synthetic", "simulator", "unverified"}:
        return False, f"Model metadata explicitly declares non-genuine data_kind: '{data_kind}'"

    provenance = metadata.get("provenance")
    if not provenance or not isinstance(provenance, dict):
        return False, "Missing verified provenance metadata"

    # Reject test doubles on production verification path
    if (provenance.get("test_double") is True or metadata.get("test_double") is True) and not allow_test_doubles:
        return False, "Model bundle contains test-double provenance marker; rejected for production acceptance"

    # Reject contradictory metadata
    dataset_type = provenance.get("dataset_type")
    if dataset_type in {"PaySim", "Feedzai-BAF"}:
        return False, f"Contradictory metadata: non-genuine dataset '{dataset_type}' cannot be verified for research"

    classification = provenance.get("classification")
    if classification != GENUINE_OBSERVED:
        return False, f"Model trained on non-genuine data classification: '{classification}'"

    if provenance.get("verification_status") != STATUS_VERIFIED:
        return False, f"Provenance verification status is '{provenance.get('verification_status')}', not VERIFIED_RESEARCH_DATA"

    if not provenance.get("acceptable_for_proposal_benchmark", False) and not (provenance.get("test_double") and allow_test_doubles):
        return False, "Dataset is marked unacceptable for proposal research benchmarks"

    return True, "Verified genuine observed dataset"

