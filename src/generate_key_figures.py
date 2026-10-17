"""Regenerate tables and figures from an explicit v0.4 run directory."""

import argparse
from pathlib import Path
from src.reporting import generate_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    generate_report(Path(args.run_dir))
    print(Path(args.run_dir) / "report.html")


if __name__ == "__main__":
    main()
