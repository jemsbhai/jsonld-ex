#!/usr/bin/env python3
"""
Download Synthea FHIR R4 sample data for jsonld-ex experiments.

Downloads the pre-generated 1K patient sample from the Synthea project
and extracts it to ``data/synthea/fhir_r4/`` under the repository root.
The ``data/`` directory is git-ignored — this script is the reproducible
record of how the data was obtained.

Usage:
    python tools/download_synthea.py              # download + extract
    python tools/download_synthea.py --verify     # also print resource stats
    python tools/download_synthea.py --stats-only # stats on existing data

For NeurIPS reproducibility, larger datasets should be generated locally
using Synthea with a pinned seed::

    git clone https://github.com/synthetichealth/synthea.git
    cd synthea
    ./run_synthea -s 12345 -p 1000 Florida

See: https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running

Data source
-----------
- URL: https://synthetichealth.github.io/synthea-sample-data/downloads/
       synthea_sample_data_fhir_r4_sep2019.zip
- License: Apache 2.0 (Synthea), data itself is CC0 / unrestricted
- Citation: Jason Walonoski et al., "Synthea: An Approach, Method, and
  Software Mechanism for Generating Synthetic Patients and the Synthetic
  Electronic Health Care Record", JAMIA 25(3), 2018.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────

SAMPLE_URL = (
    "https://synthetichealth.github.io/synthea-sample-data/downloads/"
    "synthea_sample_data_fhir_r4_sep2019.zip"
)

# SHA-256 of the Sep 2019 zip — set to None to skip verification on
# first run, then fill in the actual hash for reproducibility.
EXPECTED_SHA256: str | None = (
    "a6fc595d9c0f4c646746af42f861b5a12d03c856af158dd837c764dfb81b66f8"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "synthea" / "fhir_r4"
ZIP_PATH = REPO_ROOT / "data" / "synthea" / "synthea_sample_fhir_r4.zip"


# ── Download ──────────────────────────────────────────────────────


def download(url: str, dest: Path, *, expected_sha256: str | None = None) -> None:
    """Download *url* to *dest* with optional SHA-256 verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  zip already exists: {dest}")
        if expected_sha256 and _sha256(dest) != expected_sha256:
            print("  ⚠  SHA-256 mismatch — re-downloading")
            dest.unlink()
        else:
            return

    print(f"  downloading {url} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress

    actual = _sha256(dest)
    print(f"  SHA-256: {actual}")

    if expected_sha256 and actual != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch!\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}"
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1 << 20)
        total_mb = total_size / (1 << 20)
        print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)
    else:
        mb = downloaded / (1 << 20)
        print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)


# ── Extract ───────────────────────────────────────────────────────


def extract(zip_path: Path, dest_dir: Path) -> int:
    """Extract patient bundle JSON files from *zip_path* into *dest_dir*.

    Returns the number of files extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = list(dest_dir.glob("*.json"))
    if existing:
        print(f"  {len(existing)} JSON files already in {dest_dir}")
        return len(existing)

    print(f"  extracting to {dest_dir} ...")
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.filename.endswith(".json") and not info.is_dir():
                # Flatten: strip directory prefixes, keep just the filename
                name = Path(info.filename).name
                target = dest_dir / name
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                count += 1

    print(f"  extracted {count} files")
    return count


# ── Statistics ────────────────────────────────────────────────────

# Resource types that jsonld-ex currently supports
SUPPORTED_TYPES = frozenset({
    "RiskAssessment", "Observation", "DiagnosticReport", "Condition",
    "AllergyIntolerance", "MedicationStatement", "ClinicalImpression",
    "DetectedIssue", "Immunization", "FamilyMemberHistory", "Procedure",
    "Consent", "Provenance",
})


def compute_stats(data_dir: Path) -> dict:
    """Scan all JSON bundles and compute resource type distribution.

    Returns a dict with keys: total_bundles, total_entries,
    resource_type_counts, supported_counts, unsupported_counts,
    coverage_ratio.
    """
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"  no JSON files found in {data_dir}")
        return {}

    total_bundles = 0
    total_entries = 0
    type_counts: Counter[str] = Counter()

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                bundle = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        rt = bundle.get("resourceType")
        if rt != "Bundle":
            continue

        total_bundles += 1
        entries = bundle.get("entry", [])
        total_entries += len(entries)

        for entry in entries:
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType", "<missing>")
            type_counts[rtype] += 1

    supported_counts = {
        k: v for k, v in type_counts.items() if k in SUPPORTED_TYPES
    }
    unsupported_counts = {
        k: v for k, v in type_counts.items() if k not in SUPPORTED_TYPES
    }
    supported_total = sum(supported_counts.values())
    coverage = supported_total / total_entries if total_entries else 0.0

    return {
        "total_bundles": total_bundles,
        "total_entries": total_entries,
        "resource_type_counts": dict(type_counts.most_common()),
        "supported_counts": dict(
            sorted(supported_counts.items(), key=lambda x: -x[1])
        ),
        "unsupported_counts": dict(
            sorted(unsupported_counts.items(), key=lambda x: -x[1])
        ),
        "supported_entry_total": supported_total,
        "coverage_ratio": coverage,
    }


def print_stats(stats: dict) -> None:
    """Pretty-print resource statistics."""
    if not stats:
        return

    print(f"\n{'═' * 60}")
    print(f"  Synthea FHIR R4 — Resource Distribution")
    print(f"{'═' * 60}")
    print(f"  Bundles (patients):  {stats['total_bundles']:,}")
    print(f"  Total entries:       {stats['total_entries']:,}")
    print(f"  Supported entries:   {stats['supported_entry_total']:,}"
          f"  ({stats['coverage_ratio']:.1%})")

    print(f"\n  ── jsonld-ex supported ({len(stats['supported_counts'])}"
          f" types) ──")
    for rtype, count in stats["supported_counts"].items():
        print(f"    {rtype:<30s} {count:>6,}")

    print(f"\n  ── Not yet supported ({len(stats['unsupported_counts'])}"
          f" types) ──")
    for rtype, count in stats["unsupported_counts"].items():
        print(f"    {rtype:<30s} {count:>6,}")

    print(f"{'═' * 60}\n")


# ── CLI ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Synthea FHIR R4 sample data for jsonld-ex."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Print resource type statistics after download.",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Skip download; print stats on existing data.",
    )
    args = parser.parse_args()

    if args.stats_only:
        if not DATA_DIR.exists():
            print(f"  data directory not found: {DATA_DIR}")
            print("  run without --stats-only first to download.")
            sys.exit(1)
        stats = compute_stats(DATA_DIR)
        print_stats(stats)
        return

    print("Step 1/2: Download")
    download(SAMPLE_URL, ZIP_PATH, expected_sha256=EXPECTED_SHA256)

    print("\nStep 2/2: Extract")
    extract(ZIP_PATH, DATA_DIR)

    if args.verify:
        print("\nComputing statistics ...")
        stats = compute_stats(DATA_DIR)
        print_stats(stats)

    print("Done. Data is in:", DATA_DIR)
    print("This directory is git-ignored (data/ in .gitignore).")


if __name__ == "__main__":
    main()
