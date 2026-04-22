#!/usr/bin/env python3
"""
Count sorry-bearing theorem/lemma proofs in a Lean project directory.

Usage:
    python scripts/count_sorries.py [PROJECT_DIR]

PROJECT_DIR defaults to full_proof_recovery_benchmark/ next to this script.
Prints a per-file breakdown and a total.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "full_proof_recovery_benchmark"

_TACTIC_RE = re.compile(r"^\s+sorry\s*$")
_TERM_RE   = re.compile(r":=\s*sorry\s*$")


def count_sorries(project_dir: Path) -> dict[Path, int]:
    results: dict[Path, int] = {}
    for lean_file in sorted(project_dir.rglob("*.lean")):
        if ".lake" in lean_file.parts or "full_proof_recovery_benchmark_back" in lean_file.parts:
            continue
        n = sum(
            1 for line in lean_file.read_text(encoding="utf-8", errors="replace").splitlines()
            if _TACTIC_RE.match(line) or _TERM_RE.search(line)
        )
        if n:
            results[lean_file.relative_to(project_dir)] = n
    return results


def main() -> None:
    project_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    if not project_dir.exists():
        print(f"Directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    results = count_sorries(project_dir)
    if not results:
        print("No sorries found.")
        return

    width = max(len(str(p)) for p in results)
    for path, n in results.items():
        print(f"{str(path):<{width}}  {n:>4}")
    print("-" * (width + 6))
    print(f"{'Total':<{width}}  {sum(results.values()):>4}")


if __name__ == "__main__":
    main()
