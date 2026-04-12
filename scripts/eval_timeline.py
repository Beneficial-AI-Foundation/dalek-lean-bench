#!/usr/bin/env python3
"""
Timeline-ordered theorem proving evaluation for dalek-lean-bench.

Evaluates LLM theorem proving ability in the same order theorems were proven
historically, as recorded in proof_timeline.csv.

When evaluating theorem B at timeline position i:

  - Theorems at positions 0 .. i-1 (proven BEFORE B) keep their full proofs —
    the LLM can reference them as already-established lemmas.
  - Theorem B itself has `sorry` injected — this is what the LLM must prove.
  - Theorems at positions i+1 .. N (proven AFTER B) also have `sorry` injected,
    simulating that they have not yet been written.

This mirrors human development order: the LLM has exactly the same proof
context that was available when B was originally being worked on.

All evaluation is run in isolated git worktrees at HEAD so the main working
directory is never modified.

Prerequisites:
    lake build                  # build all dependencies once at HEAD
    claude --version            # Claude Code CLI must be on PATH

Usage:
    # Evaluate all timeline entries
    python scripts/eval_timeline.py

    # Quick smoke-test on the first 5 entries
    python scripts/eval_timeline.py --limit 5

    # Specific entries by ID  (use --list to see all IDs)
    python scripts/eval_timeline.py --ids tl_0244_mul_assign tl_0006_to_bytes

    # Resume a previous run
    python scripts/eval_timeline.py --output results/timeline-run.jsonl --resume

    # Dry-run (no claude calls, no lake build)
    python scripts/eval_timeline.py --dry-run --limit 5

    # Keep worktree after evaluation (prints path; useful for debugging)
    python scripts/eval_timeline.py --keep-worktree --ids tl_0244_mul_assign

    # Setup only: create worktree and inject sorries, skip agent and build
    python scripts/eval_timeline.py --setup-only --ids tl_0244_mul_assign

    # Stream agent output to terminal in real-time
    python scripts/eval_timeline.py --live --limit 2

Environment variables:
    ANTHROPIC_API_KEY   required (Claude Code reads this automatically)
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Content-addressed per-package cache shared across all worktrees.
PACKAGE_CACHE_DIR = Path.home() / ".cache" / "dalek-lake-packages"

DEFAULT_BUDGET_USD    = 5.00
DEFAULT_TIMEOUT       = 500
DEFAULT_BUILD_TIMEOUT = None

# Agent prompt — mentions that earlier proofs are already present.
AGENT_PROMPT_TIMELINE = """\
You are inside a Lean 4 project (dalek-lean-bench, a formal verification of \
the curve25519-dalek Rust library).

Your task: replace the `sorry` in `{file_path}` with a correct proof.
{theorem_hint}\
All theorems proven earlier in the project's history are already present and \
can be used as lemmas. Theorems proven later in history have `sorry` — do not \
rely on them.

=== Workflow ===
1. Edit the file: replace `sorry` with your proof attempt.
2. Run `nice -n 19 lake build {module}` and read the compiler output.
3. If there are errors, fix them and repeat from step 1.
4. Stop when all three conditions are satisfied:
   (1) `nice -n 19 lake build {module}` exits with code 0 (no errors),
   (2) the `sorry` has been replaced with a proof,
   (3) no new `sorry` has been introduced.

Only edit `{file_path}`. Do NOT modify any other file.
"""


# ---------------------------------------------------------------------------
# Timeline loading
# ---------------------------------------------------------------------------

def load_timeline(csv_path: Path, dataset_path: Path) -> list[dict]:
    """Load proof_timeline.csv and return entries sorted by date_proven.

    Each entry is a dict with keys from the CSV plus:
      id            – stable evaluation ID (e.g. 'tl_0006_clamp_integer')
      timeline_idx  – 0-based position in the sorted timeline
      theorem_name  – Lean theorem name to prove (from dataset or file parse)

    Only entries whose spec_theorem file exists on disk are included.
    Entries without a spec_theorem field are skipped.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Sort by date_proven, then function name for stability within the same date
    rows.sort(key=lambda r: (r["date_proven"], r["function"]))

    # Build dataset index: file_path → list of dataset entries (for theorem names)
    ds_by_path: dict[str, list[dict]] = {}
    if dataset_path.exists():
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                ds_by_path.setdefault(e["file_path"], []).append(e)

    entries = []
    for idx, row in enumerate(rows):
        spec = row.get("spec_theorem", "").strip()
        if not spec:
            continue
        full_path = REPO_ROOT / spec
        if not full_path.exists():
            continue

        # Use pre-computed ID from CSV if present, otherwise derive it on the fly
        # for backwards compatibility with old CSVs that lack the id column.
        entry_id = row.get("id", "").strip()
        if not entry_id:
            stem = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2",
                          re.sub(r"([a-z\d])([A-Z])", r"\1_\2",
                                 Path(spec).stem)).lower()
            entry_id = f"tl_{idx:04d}_{stem}"

        # Resolve theorem name: dataset first, then file parsing
        theorem_names = _resolve_theorem_names(spec, ds_by_path, full_path)

        entries.append({
            # CSV fields
            "lean_name":      row.get("lean_name", "").strip(),
            "spec_theorem":   spec,
            "function":       row.get("function", "").strip(),
            "verified":       row.get("verified", "").strip(),
            "date_proven":    row["date_proven"],
            "commit_hash":    row.get("commit_hash", "").strip(),
            "commit_message": row.get("commit_message", "").strip(),
            # Derived fields
            "id":             entry_id,
            "timeline_idx":   idx,
            "file_path":      spec,
            "theorem_names":  theorem_names,  # list; may be empty
        })

    return entries


def _resolve_theorem_names(
    spec_theorem: str,
    ds_by_path: dict[str, list[dict]],
    full_path: Path,
) -> list[str]:
    """Return a list of theorem/lemma names to prove in this file.

    Priority:
    1. Names from dataset.jsonl (reliable, manually curated)
    2. Names found by parsing the Lean file
    """
    # 1. Dataset lookup
    if spec_theorem in ds_by_path:
        return [e["theorem_name"] for e in ds_by_path[spec_theorem]]

    # 2. Parse file
    try:
        text = full_path.read_text(encoding="utf-8")
        thm_re = re.compile(r"^\s*(?:theorem|lemma)\s+(\w+)\b", re.MULTILINE)
        return thm_re.findall(text)
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Lean theorem injection helpers (adapted from eval_claude_code.py)
# ---------------------------------------------------------------------------

_TOP_LEVEL_RE = re.compile(
    r"^(theorem|lemma|def |abbrev |instance |class |structure |private |protected |"
    r"@\[|end |namespace |section |#check|#eval|#print|variable |open |set_option |"
    r"noncomputable )"
)


def inject_sorry_all_theorems(worktree: Path, file_path: str) -> int:
    """Replace ALL theorem/lemma proof bodies in `file_path` with `sorry`.

    Processes theorems back-to-front so character offsets remain valid.
    Returns the number of theorems modified.
    """
    target = worktree / file_path
    if not target.exists():
        return 0

    text = target.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Find all theorem/lemma start line indices (handles duplicate names in
    # different namespaces, which name-based dedup would silently miss).
    thm_line_re = re.compile(r"^\s*(?:theorem|lemma)\s+\w+\b")
    thm_line_indices = [i for i, ln in enumerate(lines) if thm_line_re.match(ln)]

    # Build replacement spans directly from line positions
    replacements: list[tuple[int, int, str]] = []
    for i in thm_line_indices:
        # Walk back to include leading @[...] / doc-comment lines
        attr_start = i
        while attr_start > 0:
            prev = lines[attr_start - 1].strip()
            if prev.startswith("@[") or prev.startswith("/--") or prev.startswith("--"):
                attr_start -= 1
            else:
                break

        base_indent = len(lines[i]) - len(lines[i].lstrip())

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

        by_match = re.search(r":=\s*by\b", block)
        if by_match is None:
            continue
        # Skip if the proof body is already just sorry
        proof_body = block[by_match.end():].strip()
        if proof_body == "sorry":
            continue
        new_block = block[: by_match.end()] + "\n  sorry\n"
        replacements.append((start_char, end_char, new_block))

    if not replacements:
        return 0

    # Apply in reverse order (back-to-front) to preserve offsets
    for start, end, new_block in sorted(replacements, key=lambda t: t[0], reverse=True):
        text = text[:start] + new_block + text[end:]

    target.write_text(text, encoding="utf-8")
    return len(replacements)


def inject_sorry_target(worktree: Path, entry: dict) -> bool:
    """Inject sorry into all theorems of the target entry's spec file."""
    n = inject_sorry_all_theorems(worktree, entry["file_path"])
    return n > 0


# ---------------------------------------------------------------------------
# Worktree setup with timeline state
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# git worktree helpers (same as eval_claude_code.py)
# ---------------------------------------------------------------------------

def create_worktree(commit: str, entry_id: str) -> Path:
    worktrees_dir = REPO_ROOT.parent / "dalek-worktrees"
    worktrees_dir.mkdir(exist_ok=True)
    worktree = worktrees_dir / f"dalek-timeline-eval-{entry_id}"
    if worktree.exists():
        import shutil
        shutil.rmtree(worktree)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=REPO_ROOT, capture_output=True,
        )
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), commit],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return worktree


def remove_worktree(worktree: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=REPO_ROOT,
        capture_output=True,
    )


def _read_manifest(commit: str) -> dict | None:
    out = subprocess.run(
        ["git", "show", f"{commit}:lake-manifest.json"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def _setup_worktree_packages(worktree: Path, commit: str) -> None:
    manifest = _read_manifest(commit)
    if manifest is None:
        return
    packages_dir = manifest.get("packagesDir", ".lake/packages")
    wt_packages = worktree / packages_dir
    wt_packages.mkdir(parents=True, exist_ok=True)
    for pkg in manifest.get("packages", []):
        name = pkg["name"]
        rev  = pkg["rev"]
        cache_path = PACKAGE_CACHE_DIR / name / rev
        pkg_path   = wt_packages / name
        if cache_path.exists() and not pkg_path.exists():
            pkg_path.symlink_to(cache_path)


def _hardlink_head_build(worktree: Path) -> None:
    src = REPO_ROOT / ".lake" / "build"
    dst = worktree / ".lake" / "build"
    if not src.exists() or dst.exists():
        return
    (worktree / ".lake").mkdir(exist_ok=True)
    try:
        shutil.copytree(src, dst, copy_function=os.link, symlinks=True)
    except OSError:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True)


def _seed_cache_from_head() -> None:
    head_packages = REPO_ROOT / ".lake" / "packages"
    if not head_packages.exists():
        return
    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    manifest = _read_manifest(head_commit)
    if manifest is None:
        return
    packages_dir = manifest.get("packagesDir", ".lake/packages")
    src_packages = REPO_ROOT / packages_dir
    seeded = 0
    for pkg in manifest.get("packages", []):
        name = pkg["name"]
        rev  = pkg["rev"]
        src  = src_packages / name
        cache_path = PACKAGE_CACHE_DIR / name / rev
        if cache_path.exists() or not src.exists() or src.is_symlink():
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.parent / f"{rev}.tmp.seed"
        try:
            shutil.copytree(src, tmp, symlinks=True)
            tmp.rename(cache_path)
            seeded += 1
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    if seeded:
        print(f"Cache: seeded {seeded} packages from HEAD into {PACKAGE_CACHE_DIR}")


# ---------------------------------------------------------------------------
# Final verification build
# ---------------------------------------------------------------------------

def run_lake_build(worktree: Path, file_path: str, timeout: int | None = None) -> dict:
    module = file_path.replace("/", ".").removesuffix(".lean")
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["lake", "build", module],
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        build_ok = proc.returncode == 0
        target = worktree / file_path
        has_sorry = (
            target.exists()
            and "sorry" in target.read_text(encoding="utf-8", errors="replace")
        )
        return {
            "success": build_ok and not has_sorry,
            "time_s":  round(time.time() - t0, 2),
            "stdout":  "\n".join(proc.stdout.splitlines()[:300]),
            "stderr":  "\n".join(proc.stderr.splitlines()[:300]),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "time_s":  round(time.time() - t0, 2),
            "stdout":  "",
            "stderr":  f"build TIMEOUT after {timeout}s",
        }


# ---------------------------------------------------------------------------
# Claude Code subprocess
# ---------------------------------------------------------------------------

def _format_stream_json_event(line: str, pending_tools: dict[str, str]) -> str | None:
    line = line.strip()
    if not line:
        return None
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return None

    t   = ev.get("type")
    sub = ev.get("subtype", "")

    if t == "assistant":
        parts = []
        for block in ev.get("message", {}).get("content", []):
            btype = block.get("type")
            if btype == "text":
                text = block["text"].strip()
                if text:
                    parts.append(text)
            elif btype == "tool_use":
                tool_id   = block.get("id", "")
                tool_name = block.get("name", "?")
                inp       = block.get("input", {})
                pending_tools[tool_id] = tool_name
                if tool_name == "Bash":
                    desc = inp.get("description") or inp.get("command", "")[:120]
                    parts.append(f"[Bash] {desc}")
                elif tool_name in ("Read", "Write", "Edit", "Glob", "Grep"):
                    key = next(iter(inp), None)
                    val = str(inp.get(key, ""))[:120] if key else ""
                    parts.append(f"[{tool_name}] {val}")
                else:
                    inp_s = ", ".join(
                        f"{k}={str(v)[:40]}" for k, v in list(inp.items())[:3]
                    )
                    parts.append(f"[{tool_name}] {inp_s}")
        return "\n".join(parts) if parts else None

    elif t == "user":
        parts = []
        content = ev.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    tool_id   = block.get("tool_use_id", "")
                    tool_name = pending_tools.pop(tool_id, "?")
                    rc        = block.get("content", "")
                    if isinstance(rc, list):
                        rc = " ".join(
                            c.get("text", "") for c in rc if c.get("type") == "text"
                        )
                    rc = str(rc).strip()
                    if len(rc) > 400:
                        rc = rc[:400] + " ..."
                    parts.append(f"  -> [{tool_name} result] {rc}")
        return "\n".join(parts) if parts else None

    elif t == "result":
        turns    = ev.get("num_turns", "?")
        cost     = ev.get("total_cost_usd")
        cost_str = f"  cost=${cost:.4f}" if cost else ""
        result_text = ev.get("result", "").strip()
        summary = f"[done/{sub}] turns={turns}{cost_str}"
        if result_text:
            summary += f"\n  {result_text}"
        return summary

    elif t == "system" and sub == "api_retry":
        return "[api_retry]"

    return None


def run_claude_code_agent(
    entry: dict,
    worktree: Path,
    budget_usd: float,
    timeout: int,
    model: str | None,
    live: bool = False,
) -> dict:
    """Spawn `claude --print` in the worktree. Returns exit_code, stdout, stderr, time_s."""
    module = entry["file_path"].replace("/", ".").removesuffix(".lean")

    # Build the theorem hint line for the prompt
    names = entry.get("theorem_names", [])
    if names:
        if len(names) == 1:
            theorem_hint = (
                f"The theorem to prove is `{names[0]}`.\n"
            )
        else:
            theorem_hint = (
                f"The theorems to prove are: {', '.join(f'`{n}`' for n in names)}.\n"
            )
    else:
        theorem_hint = ""

    prompt = AGENT_PROMPT_TIMELINE.format(
        file_path=entry["file_path"],
        module=module,
        theorem_hint=theorem_hint,
    )

    cmd = [
        "claude", "--print",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep,Skill,Agent,WebFetch",
        "--max-budget-usd", str(budget_usd),
        "--output-format", "stream-json",
        "--verbose",
        prompt,
    ]
    if model:
        cmd += ["--model", model]

    t0 = time.time()
    timed_out = False
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    prefix = f"[{entry['id']}] " if live else ""
    pending_tools: dict[str, str] = {}
    live_lock = threading.Lock()

    def _stdout_reader(stream) -> None:
        for line in stream:
            stdout_chunks.append(line)
            if live:
                with live_lock:
                    formatted = _format_stream_json_event(line, pending_tools)
                if formatted:
                    for fline in formatted.splitlines():
                        print(f"{prefix}{fline}", flush=True)

    def _stderr_reader(stream) -> None:
        for line in stream:
            stderr_chunks.append(line)
            if live:
                print(f"{prefix}[stderr] {line}", end="", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(worktree),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    t_out = threading.Thread(target=_stdout_reader, args=(proc.stdout,), daemon=True)
    t_err = threading.Thread(target=_stderr_reader, args=(proc.stderr,), daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        timed_out = True

    t_out.join()
    t_err.join()

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    if timed_out:
        msg = f"subprocess TIMEOUT after {timeout}s"
        stderr += f"\n{msg}"
        if live:
            print(f"{prefix}[stderr] {msg}", flush=True)

    return {
        "exit_code": proc.returncode if not timed_out else -1,
        "stdout":    stdout,
        "stderr":    stderr,
        "time_s":    round(time.time() - t0, 2),
        "timed_out": timed_out,
    }


# ---------------------------------------------------------------------------
# Single-entry evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    entry: dict,
    timeline: list[dict],
    budget_usd: float,
    timeout: int,
    build_timeout: int | None,
    model: str | None,
    dry_run: bool,
    live: bool = False,
    keep_worktree: bool = False,
    setup_only: bool = False,
) -> dict:
    """Evaluate one timeline entry with the correct historical proof context."""
    result: dict = {
        "id":             entry["id"],
        "timeline_idx":   entry["timeline_idx"],
        "file_path":      entry["file_path"],
        "function":       entry["function"],
        "date_proven":    entry["date_proven"],
        "commit_hash":    entry["commit_hash"],
        "theorem_names":  entry["theorem_names"],
        "model":          model or "default",
        "timestamp":      datetime.now(tz=timezone.utc).isoformat(),
        "success":        False,
        "agent_time_s":   None,
        "agent_exit_code": None,
        "agent_timed_out": False,
        "agent_stdout":   None,
        "agent_stderr":   None,
        "build_time_s":   None,
        "build_stdout":   "",
        "build_stderr":   "",
        "n_later_files_sorried": 0,
        "error":          None,
    }

    if dry_run:
        result["error"] = "dry-run: skipped"
        return result

    # Resolve HEAD commit for worktree
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.strip()

    try:
        worktree = create_worktree(head, entry["id"])
    except subprocess.CalledProcessError as e:
        result["error"] = f"worktree error: {e.stderr.decode()[:200]}"
        return result

    try:
        _setup_worktree_packages(worktree, head)
        _hardlink_head_build(worktree)

        # ── Inject sorry for all entries proven AFTER the target ──────────────
        # A file should be sorry-injected only if NONE of its occurrences in
        # the timeline are at or before the target's position.  If a file was
        # proven at position 5 *and* re-proven at position 100, evaluating
        # position 10 should leave that file intact (the proof existed at 5).
        target_idx = entry["timeline_idx"]
        proven_by_target: set[str] = {
            e["file_path"]
            for e in timeline
            if e["timeline_idx"] <= target_idx
        }
        # Collect unique files that are purely "later" (not yet proven at target)
        later_files_to_sorry: set[str] = {
            e["file_path"]
            for e in timeline
            if e["timeline_idx"] > target_idx
            and e["file_path"] not in proven_by_target
        }
        n_sorried = sum(
            1 for fp in later_files_to_sorry
            if inject_sorry_all_theorems(worktree, fp) > 0
        )
        result["n_later_files_sorried"] = n_sorried

        if live:
            print(
                f"[{entry['id']}] sorry-injected {n_sorried} later files  "
                f"(timeline pos {entry['timeline_idx']}/{len(timeline)-1})",
                flush=True,
            )

        # ── Inject sorry into the target file itself ──────────────────────────
        n_target = inject_sorry_all_theorems(worktree, entry["file_path"])
        if n_target == 0:
            result["error"] = (
                "inject_sorry failed: no theorem proof found in target file"
            )
            return result

        if setup_only:
            print(f"[setup-only] {entry['id']}: {worktree}")
            result["error"] = "setup-only: skipped agent and build"
            return result

        # ── Run agent ─────────────────────────────────────────────────────────
        agent_res = run_claude_code_agent(
            entry, worktree, budget_usd, timeout, model, live=live
        )
        result["agent_time_s"]    = agent_res["time_s"]
        result["agent_exit_code"] = agent_res["exit_code"]
        result["agent_timed_out"] = agent_res["timed_out"]
        result["agent_stdout"]    = agent_res["stdout"]
        result["agent_stderr"]    = agent_res["stderr"]

        if agent_res["timed_out"]:
            result["error"] = "agent timed out"
        elif agent_res["exit_code"] != 0:
            result["error"] = (
                f"claude exited {agent_res['exit_code']}: "
                f"{agent_res['stderr'][:200]}"
            )

        # ── Final verification build ──────────────────────────────────────────
        build_res = run_lake_build(worktree, entry["file_path"], timeout=build_timeout)
        result["success"]      = build_res["success"]
        result["build_time_s"] = build_res["time_s"]
        result["build_stdout"] = build_res["stdout"]
        result["build_stderr"] = build_res["stderr"]

    finally:
        if keep_worktree or setup_only:
            print(f"[keep-worktree] {entry['id']}: {worktree}")
        else:
            remove_worktree(worktree)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--timeline", default="proof_timeline.csv",
        help="Path to proof_timeline.csv (default: proof_timeline.csv)",
    )
    parser.add_argument(
        "--dataset", default="dataset.jsonl",
        help="Path to dataset.jsonl for theorem name lookup (default: dataset.jsonl)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Claude model alias, e.g. opus or claude-opus-4-6 "
             "(default: Claude Code's default)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL (default: results/timeline-<ts>.jsonl)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N entries",
    )
    parser.add_argument(
        "--ids", nargs="+", metavar="ID",
        help="Evaluate only these entry IDs (e.g. tl_0006_clamp_integer)",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Wall-clock seconds for agent subprocess (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--build-timeout", type=int, default=DEFAULT_BUILD_TIMEOUT,
        help="Wall-clock seconds for final verification build (default: no timeout)",
    )
    parser.add_argument(
        "--budget-usd", type=float, default=DEFAULT_BUDGET_USD,
        help=f"Max API spend per entry in USD (default: {DEFAULT_BUDGET_USD})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append to --output and skip already-evaluated IDs",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Number of entries to evaluate concurrently (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip claude calls and lake build",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Stream agent stdout/stderr to terminal in real-time",
    )
    parser.add_argument(
        "--keep-worktree", action="store_true",
        help="Do not delete the worktree after evaluation (prints path)",
    )
    parser.add_argument(
        "--setup-only", action="store_true",
        help="Create worktree and inject sorries, then stop — implies --keep-worktree",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print all timeline entry IDs and exit (no evaluation)",
    )
    args = parser.parse_args()

    # Load timeline
    timeline_path = REPO_ROOT / args.timeline
    dataset_path  = REPO_ROOT / args.dataset
    timeline = load_timeline(timeline_path, dataset_path)

    if args.list:
        print(f"{'ID':<40}  {'DATE':10}  {'FILE'}")
        print("-" * 100)
        for e in timeline:
            print(f"{e['id']:<40}  {e['date_proven']:10}  {e['file_path']}")
        print(f"\nTotal: {len(timeline)} entries")
        return

    # Resolve output path
    if args.output is None:
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.output = str(results_dir / f"timeline-{ts}.jsonl")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter by --ids
    to_eval = timeline
    if args.ids:
        id_set = set(args.ids)
        to_eval = [e for e in timeline if e["id"] in id_set]
        missing = id_set - {e["id"] for e in to_eval}
        if missing:
            print(f"Warning: IDs not found in timeline: {missing}")

    # Resume: skip already-done IDs
    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        to_eval = [e for e in to_eval if e["id"] not in done_ids]
        print(f"Resuming: {len(done_ids)} done, {len(to_eval)} remaining")

    if args.limit:
        to_eval = to_eval[: args.limit]

    # Seed package cache from HEAD
    _seed_cache_from_head()

    print(f"Timeline:   {timeline_path}  ({len(timeline)} total entries)")
    print(f"Evaluating: {len(to_eval)} entries{' (DRY RUN)' if args.dry_run else ''}")
    print(f"Model:      {args.model or '(claude default)'}")
    print(f"Output:     {output_path}")
    build_timeout_str = f"{args.build_timeout}s" if args.build_timeout else "none"
    print(f"Timeout:    {args.timeout}s (agent)  /  {build_timeout_str} (build)")
    print(f"Budget:     ${args.budget_usd:.2f} per entry")
    print(f"Parallel:   {args.parallel}")
    if args.live:
        print("Live:       enabled")
    print()

    n_pass = n_fail = 0
    total = len(to_eval)
    width = len(str(total))

    def _run(entry: dict) -> dict:
        return evaluate_one(
            entry,
            timeline=timeline,
            budget_usd=args.budget_usd,
            timeout=args.timeout,
            build_timeout=args.build_timeout,
            model=args.model,
            dry_run=args.dry_run,
            live=args.live,
            keep_worktree=args.keep_worktree,
            setup_only=args.setup_only,
        )

    with open(output_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_run, e): e for e in to_eval}
            completed = 0
            for fut in as_completed(futures):
                completed += 1
                res = fut.result()

                if res["success"]:
                    n_pass += 1
                    status = "PASS"
                else:
                    n_fail += 1
                    err_hint = res.get("error") or ""
                    status = f"FAIL  {err_hint[:60]}"

                names_str = ", ".join(res.get("theorem_names") or []) or "(unknown)"
                print(
                    f"[{completed:>{width}}/{total}] "
                    f"{res['id']}  ({names_str})  "
                    f"{status}  "
                    f"agent={res['agent_time_s']}s  "
                    f"build={res['build_time_s']}s  "
                    f"later_sorried={res['n_later_files_sorried']}"
                )

                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()

    pct = f"{100 * n_pass / total:.1f}%" if total else "n/a"
    print()
    print("=== Summary ===")
    print(f"  pass@1:  {n_pass}/{total}  ({pct})")
    print(f"  fail:    {n_fail}/{total}")
    print(f"  output:  {output_path}")


if __name__ == "__main__":
    main()
