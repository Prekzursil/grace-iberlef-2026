#!/usr/bin/env python3
"""User-facing submission CLI for GRACE 2026.

Usage:
    python scripts/submit.py --track {1,2} --prediction <path> --validate-only
    python scripts/submit.py --track {1,2} --prediction <path> --package

NEVER uploads directly to Codabench. Per the explicit_permission rules,
the user performs the actual upload by opening the Codabench page in a
browser. This script only packages the zip and prints the upload URL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grace.submit.package import package_submission
from grace.submit.validator import SubmissionValidationError, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=int, required=True, choices=(1, 2))
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run the official scorer on the prediction file and report pass/fail",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="Package the prediction file into a Codabench-ready zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/submissions"),
        help="Where to write the zip (default: experiments/submissions/)",
    )
    args = parser.parse_args()

    if not args.prediction.exists():
        raise SystemExit(f"prediction file not found: {args.prediction}")

    # Always validate before any packaging / upload
    try:
        scores = validate_submission(args.prediction, track=args.track)
    except SubmissionValidationError as e:
        raise SystemExit(f"VALIDATION FAILED: {e}") from e
    print(f"Validation passed. Self-consistency scores: {scores}")

    if args.validate_only:
        return

    if args.package:
        zip_path = package_submission(args.prediction, args.track, args.output_dir)
        print(f"Packaged: {zip_path}")
        print(
            f"\nTo upload, open: https://www.codabench.org/competitions/13280/"
            f"\nNavigate to the Submit tab and upload {zip_path.name}."
            f"\nClaude does NOT upload automatically — the user does this manually."
        )


if __name__ == "__main__":
    main()
