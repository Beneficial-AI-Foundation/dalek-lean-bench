#!/usr/bin/env python3
"""
Create full_proof_recovery_benchmark/ by copying all Lean files from the
project and replacing every theorem/lemma proof body with `sorry`.

Usage:
    python scripts/make_full_proof_recovery_benchmark.py

The original files are never modified.  The output directory mirrors the
Lean source structure and includes the project config files so it is
buildable as a standalone Lake project.

full_proof_recovery_benchmark/ is listed in .gitignore.
"""

import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "full_proof_recovery_benchmark"

# Regex to detect the start of a top-level Lean declaration (used to find
# where one theorem/lemma block ends and the next begins).
_TOP_LEVEL_RE = re.compile(
    r"^(theorem|lemma|def\b|abbrev |instance |class |structure |private |protected |"
    r"axiom |opaque |@\[|end |namespace |section |#check|#eval|#print|variable |open |"
    r"set_option |noncomputable |/-)"
)


# ---------------------------------------------------------------------------
# Sorry injection
# ---------------------------------------------------------------------------

def _block_comment_lines(lines: list[str]) -> set[int]:
    """Return the set of 0-based line indices that lie inside /- … -/ block comments.

    Handles nested block comments and respects -- line comments.
    Lines that are inside commented-out code regions should not be processed
    for sorry injection.
    """
    inside: set[int] = set()
    depth = 0
    for idx, line in enumerate(lines):
        if depth > 0:
            inside.add(idx)
        i = 0
        while i < len(line):
            if line[i:i+2] == "/-":
                depth += 1
                i += 2
            elif line[i:i+2] == "-/":
                if depth > 0:
                    depth -= 1
                i += 2
            elif line[i:i+2] == "--" and depth == 0:
                break  # rest of line is a line comment
            else:
                i += 1
    return inside


def _inject_sorry(text: str) -> str:
    """Return `text` with every theorem/lemma proof body replaced by `sorry`.

    Handles both tactic-mode proofs (`:= by …`) and term-mode proofs
    (`:= <expr>`).  Processes theorems back-to-front so character offsets
    remain valid throughout.
    """
    lines = text.splitlines(keepends=True)

    # Pre-compute which lines are inside /- … -/ block comments so we skip
    # theorem/lemma declarations that are inside commented-out code regions.
    block_commented = _block_comment_lines(lines)

    # Line indices where a theorem or lemma declaration begins.
    thm_line_re = re.compile(r"^\s*(?:theorem|lemma)\s+\w+\b")
    thm_indices = [
        i for i, ln in enumerate(lines)
        if thm_line_re.match(ln) and i not in block_commented
    ]

    replacements: list[tuple[int, int, str]] = []

    for i in thm_indices:
        # Walk back to include leading @[…] attribute / doc-comment lines so
        # they are preserved in the replacement block.
        attr_start = i
        while attr_start > 0:
            prev = lines[attr_start - 1].strip()
            if prev.startswith("@[") or prev.startswith("/--") or prev.startswith("--"):
                attr_start -= 1
            else:
                break

        base_indent = len(lines[i]) - len(lines[i].lstrip())

        # Find where this block ends: the first non-blank line at the same
        # indentation level that starts a new top-level declaration.
        end = i + 1
        while end < len(lines):
            curr = lines[end]
            stripped = curr.strip()
            if stripped:
                curr_indent = len(curr) - len(curr.lstrip())
                if curr_indent <= base_indent and _TOP_LEVEL_RE.match(stripped):
                    break
            end += 1

        start_char = sum(len(l) for l in lines[:attr_start])
        end_char   = sum(len(l) for l in lines[:end])
        block = text[start_char:end_char]

        # ── Tactic-mode proof: := by … ────────────────────────────────────
        by_match = re.search(r":=\s*by\b", block)
        if by_match is not None:
            proof_body = block[by_match.end():].strip()
            if proof_body == "sorry":
                continue
            new_block = block[: by_match.end()] + "\n  sorry\n"
            replacements.append((start_char, end_char, new_block))
            continue

        # ── Term-mode proof: := <expr> ────────────────────────────────────
        # Find the := that introduces the proof body.  We skip any := that
        # appears on a line whose content (before the :=) starts with `let`
        # or `have` — those are local bindings inside the *type*, not the
        # proof-introducing :=.
        term_match = None
        for m in re.finditer(r":=(?!\s*by\b)", block):
            line_start = block.rfind("\n", 0, m.start()) + 1
            before = block[line_start : m.start()]
            if re.match(r"\s*(let|have)\s", before):
                continue  # binding inside the type signature — skip
            term_match = m
            break
        if term_match is not None:
            proof_body = block[term_match.end():].strip()
            if proof_body == "sorry":
                continue
            # Replace everything after := with a single `sorry`.
            new_block = block[: term_match.end()] + " sorry\n"
            replacements.append((start_char, end_char, new_block))

    if not replacements:
        return text

    # Apply back-to-front so earlier offsets stay valid.
    for start, end, new_block in sorted(
        replacements, key=lambda t: t[0], reverse=True
    ):
        text = text[:start] + new_block + text[end:]

    return text


# ---------------------------------------------------------------------------
# Directory construction
# ---------------------------------------------------------------------------

def _build_benchmark(src: Path, dst: Path) -> int:
    """Inject sorrys into all .lean files from *src* and write them to *dst*.

    Only .lean files are written; config files and .lake are never touched.
    Returns the number of .lean files written.
    """
    dst.mkdir(parents=True, exist_ok=True)

    # Discover and process all .lean files outside .lake/.
    count = 0
    for lean_file in sorted(src.rglob("*.lean")):
        parts = lean_file.parts
        # Skip build artefacts and the output directory itself.
        if ".lake" in parts:
            continue
        try:
            lean_file.relative_to(dst)
            continue  # already inside the output dir
        except ValueError:
            pass

        rel = lean_file.relative_to(src)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        original = lean_file.read_text(encoding="utf-8")
        sorried   = _inject_sorry(original)
        dst_file.write_text(sorried, encoding="utf-8")
        count += 1

    return count


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)


def _commit_config_files(repo: Path, files: list[str]) -> None:
    """Stage *files* that exist in *repo* and commit them if anything changed."""
    to_add = [f for f in files if (repo / f).exists()]
    if not to_add:
        return
    _git(["add"] + to_add, cwd=repo)
    result = _git(["diff", "--cached", "--quiet"], cwd=repo)
    if result.returncode == 0:
        print("Config files already committed, nothing to commit.")
        return
    _git(["commit", "-m", "chore: add project config files"], cwd=repo)
    print(f"Committed config files: {', '.join(to_add)}")


# ---------------------------------------------------------------------------
# .gitignore update
# ---------------------------------------------------------------------------

def _ensure_gitignore(repo_root: Path, entry: str) -> None:
    gitignore = repo_root / ".gitignore"
    content = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if entry in content.splitlines():
        print(f".gitignore already contains '{entry}'")
        return
    sep = "" if content.endswith("\n") else "\n"
    gitignore.write_text(content + sep + entry + "\n", encoding="utf-8")
    print(f"Added '{entry}' to .gitignore")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Output directory : {OUTPUT_DIR}")

    first_run = not OUTPUT_DIR.exists()
    if first_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("First run — copying project config files …")
        for name in ["lakefile.toml", "lean-toolchain", "lake-manifest.json"]:
            src_file = REPO_ROOT / name
            if src_file.exists():
                shutil.copy2(src_file, OUTPUT_DIR / name)
                print(f"  config  {name}")
    else:
        print("Output directory exists — skipping config files, updating .lean files only …")

    # Always copy .gitignore so .lake build artefacts are ignored inside the
    # output directory regardless of whether this is a first or subsequent run.
    gitignore_src = REPO_ROOT / ".gitignore"
    if gitignore_src.exists():
        shutil.copy2(gitignore_src, OUTPUT_DIR / ".gitignore")
        print("  config  .gitignore")

    _commit_config_files(
        OUTPUT_DIR,
        [".gitignore", "lakefile.toml", "lean-toolchain", "lake-manifest.json"],
    )

    print("Injecting sorrys into .lean files …")
    n = _build_benchmark(REPO_ROOT, OUTPUT_DIR)
    print(f"Wrote {n} Lean files with proofs replaced by sorry")

    # Create a backup of the sorry-injected files so that
    # check_full_proof_recovery.py can use it as a baseline.
    backup_dir = OUTPUT_DIR / "full_proof_recovery_benchmark_back"
    print(f"Creating baseline backup in {backup_dir} …")
    _build_benchmark(REPO_ROOT, backup_dir)
    print("Baseline backup written.")

    _ensure_gitignore(REPO_ROOT, "full_proof_recovery_benchmark")
    print("Done.")


if __name__ == "__main__":
    main()
