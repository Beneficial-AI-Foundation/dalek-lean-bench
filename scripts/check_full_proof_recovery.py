#!/usr/bin/env python3
"""
Check how many sorries a coding agent proved in full_proof_recovery_benchmark/.

The benchmark is a copy of the project where every theorem/lemma proof body has
been replaced with `sorry`.  A coding agent is expected to fill in proofs without:
  - introducing new sorry-using declarations, and
  - creating new .lean files.

`lake build` is used as the authoritative source for the remaining-sorry count
(it reports one warning per *declaration* that uses sorry, not per raw occurrence).

The baseline sorry count is derived from the backup directory
`full_proof_recovery_benchmark/full_proof_recovery_benchmark_back/`, which is
created by `make_full_proof_recovery_benchmark.py` alongside the benchmark.

Usage:
    # Basic report (exits 0 if no violations, 1 if agent broke the rules)
    python scripts/check_full_proof_recovery.py

    # Show full lake build output
    python scripts/check_full_proof_recovery.py --verbose
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "full_proof_recovery_benchmark"
BACKUP_DIR_NAME = "full_proof_recovery_benchmark_back"

# File-path prefixes (as reported by lake build) that belong to project source.
# Dependencies (Aeneas, Mathlib, PrimeCert, …) use other prefixes and are excluded.
_PROJECT_PREFIXES = ("Curve25519Dalek/", "Utils/", "Curve25519Dalek.lean", "Utils.lean")

_SORRY_WARNING_RE = re.compile(r"^warning: (.+?):\d+:\d+: declaration uses `sorry`")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_project_sorry_warnings(build_output: str) -> int:
    """Count sorry-warning lines that belong to project source files."""
    count = 0
    for line in build_output.splitlines():
        m = _SORRY_WARNING_RE.match(line)
        if m and m.group(1).startswith(_PROJECT_PREFIXES):
            count += 1
    return count


def _baseline_sorry_count(backup_dir: Path) -> int:
    """Count injected sorrys in the backup (original benchmark state).

    `make_full_proof_recovery_benchmark.py` injects exactly one sorry per
    theorem/lemma proof body, in one of two forms:
      - tactic mode: a line containing only ``sorry`` with leading whitespace
      - term  mode: a line ending with ``:= sorry``

    Counting these patterns gives the same number as ``lake build`` warning
    lines would for the baseline state, because each injected sorry corresponds
    to exactly one Lean declaration.
    """
    if not backup_dir.exists():
        return -1  # backup unavailable; caller must handle

    tactic_re = re.compile(r"^\s+sorry\s*$")
    term_re = re.compile(r":=\s*sorry\s*$")

    count = 0
    for lean_file in sorted(backup_dir.rglob("*.lean")):
        if ".lake" in lean_file.parts:
            continue
        for line in lean_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if tactic_re.match(line) or term_re.search(line):
                count += 1
    return count


def _detect_new_lean_files(benchmark_dir: Path, backup_dir: Path) -> list[Path]:
    """Return relative paths of .lean files present in benchmark but not in backup."""
    if not backup_dir.exists():
        return []

    backup_rel: set[Path] = {
        f.relative_to(backup_dir)
        for f in backup_dir.rglob("*.lean")
        if ".lake" not in f.parts
    }

    new_files: list[Path] = []
    backup_dir_name = backup_dir.name
    for f in sorted(benchmark_dir.rglob("*.lean")):
        if ".lake" in f.parts:
            continue
        rel = f.relative_to(benchmark_dir)
        # Ignore the backup sub-directory itself.
        if rel.parts and rel.parts[0] == backup_dir_name:
            continue
        if rel not in backup_rel:
            new_files.append(rel)
    return new_files


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def count_sorries_proven(
    benchmark_dir: Path = BENCHMARK_DIR,
) -> dict:
    """Run ``lake build`` in *benchmark_dir* and compute agent progress.

    Returns a dict with the following keys:

    ``baseline_sorries``
        Total sorry-using declarations in the original benchmark (from backup).
        -1 if the backup directory is missing.
    ``remaining_sorries``
        Sorry-using declarations left in project source files after the agent ran,
        as reported by ``lake build``.
    ``proven_sorries``
        ``baseline_sorries - remaining_sorries``.  None if baseline unavailable.
    ``new_lean_files``
        List of .lean file paths (str) the agent created that were not in the backup.
        Should be empty — agents must not add new files.
    ``new_sorries_added``
        True when ``remaining_sorries > baseline_sorries``, meaning the agent
        introduced at least one new sorry-using declaration.  None if baseline
        is unavailable.
    ``build_exit_code``
        Exit code of ``lake build`` (0 on success).
    ``build_time_s``
        Wall-clock seconds for the build.
    ``build_output``
        Combined stdout + stderr from ``lake build``.
    """
    backup_dir = benchmark_dir / BACKUP_DIR_NAME

    baseline = _baseline_sorry_count(backup_dir)
    new_files = _detect_new_lean_files(benchmark_dir, backup_dir)

    cmd = ["nice", "-n", "19", "lake", "build"]

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=benchmark_dir,
        capture_output=True,
        text=True,
    )
    elapsed = round(time.time() - t0, 2)

    build_output = proc.stdout + proc.stderr
    remaining = _count_project_sorry_warnings(build_output)
    proven = (baseline - remaining) if baseline >= 0 else None
    new_sorries = (remaining > baseline) if baseline >= 0 else None

    return {
        "baseline_sorries": baseline,
        "remaining_sorries": remaining,
        "proven_sorries": proven,
        "new_lean_files": [str(f) for f in new_files],
        "new_sorries_added": new_sorries,
        "build_exit_code": proc.returncode,
        "build_time_s": elapsed,
        "build_output": build_output,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmark-dir",
        default=str(BENCHMARK_DIR),
        help=f"Path to benchmark directory (default: {BENCHMARK_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full lake build output",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.exists():
        print(f"error: benchmark directory not found: {benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmark : {benchmark_dir}")
    print(f"Running   : nice -n 19 lake build")
    print()

    result = count_sorries_proven(benchmark_dir=benchmark_dir)

    if args.verbose:
        print("=== lake build output ===")
        print(result["build_output"])
        print("=========================")
        print()

    baseline = result["baseline_sorries"]
    remaining = result["remaining_sorries"]
    proven = result["proven_sorries"]
    new_files = result["new_lean_files"]
    new_sorries = result["new_sorries_added"]

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"Build exit code   : {result['build_exit_code']}  ({result['build_time_s']:.1f}s)")
    if baseline < 0:
        print(f"Baseline          : N/A (backup dir not found: {benchmark_dir / BACKUP_DIR_NAME})")
    else:
        print(f"Baseline sorries  : {baseline}")
    print(f"Remaining sorries : {remaining}")
    if proven is not None:
        print(f"Proven sorries    : {proven}")
    print()

    violations: list[str] = []

    if new_sorries:
        violations.append(
            f"VIOLATION: agent introduced new sorries "
            f"(remaining {remaining} > baseline {baseline})"
        )

    if new_files:
        violations.append(
            f"VIOLATION: agent created {len(new_files)} new .lean file(s):\n"
            + "\n".join(f"  {f}" for f in new_files)
        )

    if violations:
        for v in violations:
            print(v, file=sys.stderr)
        sys.exit(1)
    else:
        if proven is not None:
            pct = f" ({100 * proven / baseline:.1f}%)" if baseline > 0 else ""
            print(f"OK — {proven}/{baseline} sorries proven{pct}")
        else:
            print(f"OK — {remaining} sorries remaining (baseline unavailable)")
        sys.exit(0)


if __name__ == "__main__":
    main()
