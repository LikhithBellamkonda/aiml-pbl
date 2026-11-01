from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.ingestion.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire, standardize, and assess pregnancy-safety datasets."
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Acquire missing pinned source files using their official APIs/releases.",
    )
    parser.add_argument(
        "--max-pubchem-queries",
        type=int,
        default=0,
        help="Maximum new PubChem lookups; defaults to cached/offline resolution.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.max_pubchem_queries < 0:
        parser.error("--max-pubchem-queries must be non-negative.")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    report = run_pipeline(
        root=ROOT,
        download_missing=args.download_missing,
        max_pubchem_queries=args.max_pubchem_queries,
    )
    print(
        f"Dataset ingestion complete: {report['master_record_count']} evidence records; "
        f"{report['unique_mapped_drug_count']} mapped drugs; "
        f"{sum(item['decision'] == 'trainable' for item in report['target_decisions'])} "
        "pregnancy endpoints trainable."
    )
    print(f"Report: {ROOT / 'reports' / 'dataset_integration_feasibility.md'}")


if __name__ == "__main__":
    main()
