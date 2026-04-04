#!/usr/bin/env python3
"""
Extract Lean theorem-proving dataset from git history.

For each commit where a sorry was replaced by a real proof, emit one JSONL entry
containing the theorem statement, the proof, and the full file context.

Usage:
    python scripts/extract_dataset.py [--output dataset.jsonl] [--branch-only]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level Lean declarations that can terminate a theorem block
TOP_LEVEL_RE = re.compile(
    r"^(theorem|lemma|def |abbrev |instance |class |structure |private |protected |"
    r"@\[|end |namespace |section |#check|#eval|#print|variable |open |set_option |"
    r"noncomputable )"
)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args: str) -> str:
    r = subprocess.run(
        ["git"] + list(args),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return r.stdout


def get_file_at(commit: str, filepath: str) -> str | None:
    r = subprocess.run(
        ["git", "show", f"{commit}:{filepath}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return r.stdout if r.returncode == 0 else None


def get_non_merge_commits(branch_only: bool = False) -> list[tuple[str, str]]:
    """Return list of (commit_hash, parent_hash) for non-merge commits."""
    ref = "HEAD" if branch_only else "--all"
    out = git("log", ref, "--no-merges", "--pretty=format:%H %P", "--", "*.lean")
    result = []
    for line in out.strip().splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip():
            result.append((parts[0], parts[1].strip()))
    return result


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------

def files_with_net_sorry_removal(parent: str, child: str) -> dict[str, int]:
    """
    Return {filepath: net_removed_count} for files where active sorry lines
    were net-removed (i.e. proof was added, not just sorry moved around).
    """
    diff = git("diff", parent, child, "--", "*.lean")
    file_counts: dict[str, list[int]] = {}  # {file: [removed, added]}
    current_file = None

    for line in diff.splitlines():
        if line.startswith("diff --git"):
            m = re.search(r"b/(.+)$", line)
            current_file = m.group(1) if m else None
            if current_file:
                file_counts[current_file] = [0, 0]
        elif current_file:
            # Only count lines where sorry is the whole tactic (not in comments)
            is_sorry_tactic = bool(re.search(r"^\s*(?:·\s*)?sorry\s*$", line[1:]))
            if line.startswith("-") and not line.startswith("---") and is_sorry_tactic:
                file_counts[current_file][0] += 1
            elif line.startswith("+") and not line.startswith("+++") and is_sorry_tactic:
                file_counts[current_file][1] += 1

    return {
        f: removed - added
        for f, (removed, added) in file_counts.items()
        if removed > added
    }


# ---------------------------------------------------------------------------
# Lean file parser
# ---------------------------------------------------------------------------

def strip_lean_comments(text: str) -> str:
    """Remove block comments /- ... -/ and line comments -- ... from text."""
    # Block comments (non-nested for simplicity)
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    # Line comments
    text = re.sub(r"--[^\n]*", "", text)
    return text


def has_active_sorry(proof_body: str) -> bool:
    """Return True if proof_body contains a sorry outside of comments."""
    return bool(re.search(r"\bsorry\b", strip_lean_comments(proof_body)))


def find_theorem_block(lines: list[str], start: int) -> tuple[int, int]:
    """
    Given that lines[start] contains a theorem/lemma declaration,
    return (attr_start, end) where:
      attr_start: index of first attribute/doc-comment line before the theorem
      end:        index of first line AFTER the theorem block
    """
    # Walk backwards to include @[...] and /-- ... -/ doc comment lines
    attr_start = start
    while attr_start > 0:
        prev = lines[attr_start - 1].strip()
        if prev.startswith("@[") or prev.startswith("/--") or prev.startswith("--"):
            attr_start -= 1
        else:
            break

    # Determine base indentation from the theorem line
    thm_line = lines[start]
    base_indent = len(thm_line) - len(thm_line.lstrip())

    # Walk forwards until we hit a top-level declaration at same/lower indent
    end = start + 1
    while end < len(lines):
        curr = lines[end]
        stripped = curr.strip()
        if not stripped:
            end += 1
            continue
        curr_indent = len(curr) - len(curr.lstrip())
        if curr_indent <= base_indent and TOP_LEVEL_RE.match(stripped):
            break
        end += 1

    return attr_start, end


def extract_theorems(content: str) -> list[dict]:
    """
    Extract all theorem/lemma blocks from a Lean file.
    Returns list of dicts with keys:
      name, statement, proof_body, full_block, line_start (1-indexed)
    """
    lines = content.splitlines()
    results = []
    visited: set[int] = set()  # avoid re-processing lines

    thm_re = re.compile(
        r"^(\s*)(?:private\s+|protected\s+)?(?:theorem|lemma)\s+(\w+)"
    )

    for i, line in enumerate(lines):
        if i in visited:
            continue
        m = thm_re.match(line)
        if not m:
            continue

        thm_name = m.group(2)
        attr_start, end = find_theorem_block(lines, i)

        # Mark all lines in this block as visited
        for k in range(attr_start, end):
            visited.add(k)

        block = "\n".join(lines[attr_start:end])

        # Find := by (proof body starts after it)
        by_match = re.search(r":=\s*by\b", block)
        if by_match:
            statement = block[: by_match.end()].rstrip()
            proof_body = block[by_match.end() :]
        else:
            # := without by (term-mode proof or axiom)
            statement = block
            proof_body = ""

        results.append(
            {
                "name": thm_name,
                "statement": statement,
                "proof_body": proof_body,
                "full_block": block,
                "line_start": attr_start + 1,
            }
        )

    return results


def find_theorem_in_file(content: str, thm_name: str) -> dict | None:
    """Find a specific theorem by name in a file."""
    for thm in extract_theorems(content):
        if thm["name"] == thm_name:
            return thm
    return None


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_dataset(branch_only: bool = False) -> list[dict]:
    print("Loading commits...", flush=True)
    commits = get_non_merge_commits(branch_only)
    print(f"  {len(commits)} non-merge commits touching .lean files", flush=True)

    dataset: list[dict] = []
    n_commits_processed = 0
    seen_ids: set[str] = set()  # deduplicate by (theorem_name, file_path)

    for idx, (commit_hash, parent_hash) in enumerate(commits):
        file_map = files_with_net_sorry_removal(parent_hash, commit_hash)
        if not file_map:
            continue

        n_commits_processed += 1

        for filepath, n_sorry_removed in file_map.items():
            content_before = get_file_at(parent_hash, filepath)
            content_after = get_file_at(commit_hash, filepath)
            if content_before is None:
                continue

            theorems_before = extract_theorems(content_before)

            for thm in theorems_before:
                if not has_active_sorry(thm["proof_body"]):
                    continue

                # Check if sorry was eliminated in child
                if content_after is None:
                    continue
                thm_after = find_theorem_in_file(content_after, thm["name"])
                if thm_after is None:
                    continue
                if has_active_sorry(thm_after["proof_body"]):
                    continue  # sorry still present in child

                # Deduplicate: keep the LATEST commit that proved this theorem
                dedup_key = f"{thm['name']}::{filepath}"
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

                proof = thm_after["proof_body"].strip()

                entry = {
                    "id": f"{thm['name']}_{commit_hash[:7]}",
                    "theorem_name": thm["name"],
                    "file_path": filepath,
                    "commit_before": parent_hash,
                    # Core dataset fields
                    "formal_statement": thm["statement"],
                    "proof": proof,
                    "full_theorem_with_sorry": thm["full_block"],
                    # Metadata
                    "n_sorry_removed_in_commit": n_sorry_removed,
                    "proof_lines": len(proof.splitlines()),
                    "statement_lines": len(thm["statement"].splitlines()),
                }
                dataset.append(entry)

        if n_commits_processed % 20 == 0:
            print(
                f"  [{idx+1}/{len(commits)}] processed {n_commits_processed} "
                f"sorry-removal commits, {len(dataset)} entries so far...",
                flush=True,
            )

    return dataset


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="dataset.jsonl",
        help="Output JSONL file path (default: dataset.jsonl)",
    )
    parser.add_argument(
        "--branch-only",
        action="store_true",
        help="Only consider commits reachable from HEAD (default: all branches)",
    )
    args = parser.parse_args()

    dataset = extract_dataset(branch_only=args.branch_only)

    output_path = REPO_ROOT / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    print(f"\nWrote {len(dataset)} entries to {output_path}")
    if dataset:
        unique_theorems = len({e["theorem_name"] for e in dataset})
        unique_files = len({e["file_path"] for e in dataset})
        avg_proof = sum(e["proof_lines"] for e in dataset) / len(dataset)
        avg_stmt = sum(e["statement_lines"] for e in dataset) / len(dataset)
        proof_dist = {}
        for e in dataset:
            bucket = (e["proof_lines"] // 10) * 10
            proof_dist[bucket] = proof_dist.get(bucket, 0) + 1

        print(f"\nSummary:")
        print(f"  Unique theorems:      {unique_theorems}")
        print(f"  Unique files:         {unique_files}")
        print(f"  Avg proof length:     {avg_proof:.1f} lines")
        print(f"  Avg statement length: {avg_stmt:.1f} lines")
        print(f"\n  Proof length distribution (lines):")
        for bucket in sorted(proof_dist):
            bar = "█" * proof_dist[bucket]
            print(f"    {bucket:>4}-{bucket+9:<4}: {proof_dist[bucket]:>4}  {bar}")


if __name__ == "__main__":
    main()
