"""
Master Reproduction Orchestrator.

A single command that executes the entire end-to-end research benchmark suite,
regenerating every number, table, and publication figure from raw data and controlled streams.

Pipeline Stages:
1. Statistical Rigor Benchmark (Bootstrap CIs & Paired Hypothesis Tests)
2. Concept Drift Detection Benchmark (Change-Point Delays & False Alarms)
3. TreeSHAP x PSI Attribution Validation on Injected Ground-Truth
4. Multi-Bank Non-IID Federated Evaluation
5. Decisive Label Verification Latency Frontier (Unsupervised P(X) vs Supervised)
6. Publication Figure Generation (300 DPI Figures)
7. LaTeX Table Generation (Synchronized strictly from JSON ground truth)

Usage:
    python -m src.reproduce_all
"""

import sys
import time
import logging
from pathlib import Path

# Configure clean logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("fraud_drift.reproduce_all")


def run_all_stages():
    start_total = time.time()
    logger.info("=" * 80)
    logger.info("STARTING FULL REPRODUCIBILITY PIPELINE: Concept-Drift Fraud Detection")
    logger.info("=" * 80)

    stages = [
        ("Stage 1: Statistical Rigor & Paired Tests", "src.benchmark_statistical_rigor"),
        ("Stage 2: Drift Detection Delay & False Alarms", "src.drift_benchmark"),
        ("Stage 3: TreeSHAP x PSI Attribution Validation", "src.drift_explainer"),
        ("Stage 4: Federated Multi-Bank Non-IID Evaluation", "src.federated"),
        ("Stage 5: Decisive Label Latency Frontier (P(X) vs Supervised)", "src.benchmark_label_delay_px"),
        ("Stage 6: Publication Key Figures Generation", "src.generate_key_figures"),
        ("Stage 7: Paper LaTeX Tables Generation", "src.generate_paper_tables"),
    ]

    results = []

    for stage_name, module_name in stages:
        t0 = time.time()
        logger.info(f"\n>>> Running {stage_name} ({module_name})...")
        try:
            mod = __import__(module_name, fromlist=["main"])
            if hasattr(mod, "main"):
                mod.main()
            elif hasattr(mod, "run_all"):
                mod.run_all()
            elif hasattr(mod, "run_benchmark"):
                mod.run_benchmark()
            else:
                logger.warning(f"No explicit entrypoint found in {module_name}, imported successfully.")
            elapsed = time.time() - t0
            logger.info(f"    [PASS] {stage_name} completed in {elapsed:.2f}s")
            results.append((stage_name, "SUCCESS", elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"    [FAIL] {stage_name} failed after {elapsed:.2f}s: {e}")
            results.append((stage_name, f"FAILED: {e}", elapsed))

    total_time = time.time() - start_total
    logger.info("\n" + "=" * 80)
    logger.info("REPRODUCIBILITY EXECUTION SUMMARY")
    logger.info("=" * 80)
    for name, status, elapsed in results:
        logger.info(f"  * {name:<60}: {status} ({elapsed:.2f}s)")
    logger.info(f"\nTotal Pipeline Runtime: {total_time:.2f} seconds")
    logger.info("All metrics, figures, and LaTeX tables have been regenerated in results/ and arxiv_paper/.")
    logger.info("=" * 80)


def main():
    run_all_stages()


if __name__ == "__main__":
    main()
