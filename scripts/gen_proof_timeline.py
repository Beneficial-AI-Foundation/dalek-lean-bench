#!/usr/bin/env python3
"""
Generate proof_timeline.csv by walking git history of status.csv
and detecting when each theorem's `verified` column first transitions
to "verified" or "externally verified".

Schema history:
  - Before commit 951ab02 (2025-12-14): no `lean_name` column; key = `function`
  - From commit 951ab02 onward: has `lean_name` column; key = `lean_name`

We track by `function` throughout, then join with the current status.csv
to fill in `lean_name` for old-schema entries.
"""

import csv
import io
import re
import subprocess
import sys
from pathlib import Path


def _spec_to_id(idx: int, spec: str) -> str:
    """Build a stable eval ID from a row's sorted index and spec_theorem path."""
    stem = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2",
                  re.sub(r"([a-z\d])([A-Z])", r"\1_\2",
                         Path(spec).stem)).lower()
    return f"tl_{idx:04d}_{stem}"


VERIFIED_STATUSES = {"verified", "externally verified"}

# Commit that introduced lean_name; from here on the new schema is used
LEAN_NAME_COMMIT = "951ab02b4ddf3bbfeabbae57e25b1bd55f3d1283"


def git_show_csv(ref, path):
    """Return (has_lean_name, rows_by_function) for status.csv at a given git ref."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return False, {}
    reader = csv.DictReader(io.StringIO(result.stdout))
    has_lean_name = "lean_name" in (reader.fieldnames or [])
    rows = {}
    for row in reader:
        fn = row.get("function", "").strip()
        if fn:
            rows[fn] = row
    return has_lean_name, rows


def git_log_commits(path):
    """Return list of (hash, date, subject) for all commits touching path, oldest first."""
    result = subprocess.run(
        ["git", "log", "--reverse", "--pretty=format:%H\t%ad\t%s", "--date=short", "--", path],
        capture_output=True, text=True
    )
    commits = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            commits.append(tuple(parts))
    return commits


def load_current_function_to_lean_name(path):
    """Build function -> lean_name map from the current status.csv."""
    mapping = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fn = row.get("function", "").strip()
            ln = row.get("lean_name", "").strip()
            if fn and ln:
                mapping[fn] = ln
    return mapping


def main():
    status_path = "status.csv"
    output_path = "proof_timeline.csv"

    commits = git_log_commits(status_path)
    print(f"Found {len(commits)} commits touching {status_path}", file=sys.stderr)

    fn_to_lean_name = load_current_function_to_lean_name(status_path)

    # function -> timeline entry
    timeline = {}

    prev_rows = {}
    for commit_hash, date, subject in commits:
        _, curr_rows = git_show_csv(commit_hash, status_path)

        for fn, curr in curr_rows.items():
            curr_status = curr.get("verified", "").strip()
            if curr_status not in VERIFIED_STATUSES:
                continue

            prev = prev_rows.get(fn)
            prev_status = prev.get("verified", "").strip() if prev else ""

            if prev_status == curr_status:
                continue  # no change

            lean_name = curr.get("lean_name", "").strip() or fn_to_lean_name.get(fn, "")

            if fn not in timeline:
                timeline[fn] = {
                    "lean_name": lean_name,
                    "spec_theorem": curr.get("spec_theorem", "").strip(),
                    "function": fn,
                    "verified": curr_status,
                    "date_proven": date,
                    "commit_hash": commit_hash,
                    "commit_message": subject,
                }
            else:
                # Re-proof: update only if status upgraded (externally verified → verified)
                existing = timeline[fn]
                if existing["verified"] != curr_status:
                    existing["verified"] = curr_status
                    existing["date_proven"] = date
                    existing["commit_hash"] = commit_hash
                    existing["commit_message"] = subject
                    if lean_name:
                        existing["lean_name"] = lean_name

        prev_rows = curr_rows

    # Sort by date_proven, then function
    rows = sorted(timeline.values(), key=lambda r: (r["date_proven"], r["function"]))

    # Assign stable eval IDs using each row's position in the full sorted list.
    # Rows without a spec_theorem get an empty id (they are skipped during eval).
    for idx, row in enumerate(rows):
        spec = row.get("spec_theorem", "").strip()
        row["id"] = _spec_to_id(idx, spec) if spec else ""

    fieldnames = ["id", "lean_name", "spec_theorem", "function", "verified", "date_proven", "commit_hash", "commit_message"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} entries to {output_path}", file=sys.stderr)

    by_status = {}
    for r in rows:
        by_status.setdefault(r["verified"], []).append(r)
    for status, items in sorted(by_status.items()):
        print(f"  {status}: {len(items)}", file=sys.stderr)

    # Show date distribution
    print("\nDate distribution:", file=sys.stderr)
    from collections import Counter
    counts = Counter(r["date_proven"] for r in rows)
    for d, c in sorted(counts.items()):
        print(f"  {d}: {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
