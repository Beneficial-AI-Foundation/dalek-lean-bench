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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "full_proof_recovery_benchmark"

# Regex to detect the start of a top-level Lean declaration (used to find
# where one theorem/lemma block ends and the next begins).
_TOP_LEVEL_RE = re.compile(
    r"^(theorem|lemma|def |abbrev |instance |class |structure |private |protected |"
    r"axiom |opaque |@\[|end |namespace |section |#check|#eval|#print|variable |open |"
    r"set_option |noncomputable )"
)


# ---------------------------------------------------------------------------
# Sorry injection
# ---------------------------------------------------------------------------

def _inject_sorry(text: str) -> str:
    """Return `text` with every theorem/lemma proof body replaced by `sorry`.

    Handles both tactic-mode proofs (`:= by …`) and term-mode proofs
    (`:= <expr>`).  Processes theorems back-to-front so character offsets
    remain valid throughout.
    """
    lines = text.splitlines(keepends=True)

    # Line indices where a theorem or lemma declaration begins.
    thm_line_re = re.compile(r"^\s*(?:theorem|lemma)\s+\w+\b")
    thm_indices = [i for i, ln in enumerate(lines) if thm_line_re.match(ln)]

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
        # Find := that is NOT immediately followed by 'by'.
        # In theorem/lemma signatures the only := is the one introducing the
        # proof body, so the first match is the right one.
        term_match = re.search(r":=(?!\s*by\b)", block)
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

# Project configuration files that must be present for `lake build` to work.
_CONFIG_FILES = [
    "lakefile.toml",
    "lean-toolchain",
    "lake-manifest.json",
]

def _build_benchmark(src: Path, dst: Path) -> int:
    """Copy the Lean project from *src* to *dst* with sorrys injected.

    Returns the number of .lean files written.
    """
    dst.mkdir(parents=True, exist_ok=True)

    # Copy project config verbatim.
    for name in _CONFIG_FILES:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)
            print(f"  config  {name}")

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

    if OUTPUT_DIR.exists():
        print("Removing existing output directory …")
        shutil.rmtree(OUTPUT_DIR)

    print("Copying and injecting sorrys …")
    n = _build_benchmark(REPO_ROOT, OUTPUT_DIR)
    print(f"Wrote {n} Lean files with proofs replaced by sorry")

    _ensure_gitignore(REPO_ROOT, "full_proof_recovery_benchmark")
    print("Done.")


if __name__ == "__main__":
    main()
