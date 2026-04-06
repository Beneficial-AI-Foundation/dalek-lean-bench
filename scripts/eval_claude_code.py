#!/usr/bin/env python3
"""
Claude Code agentic evaluation harness for dalek-lean-bench.

Each dataset entry is evaluated inside a git worktree checked out at
`commit_before`.  A `claude --print --bare` subprocess is spawned per entry;
the agent autonomously reads surrounding Lean files, edits the sorry, runs
`lake build` to read compiler errors, and iterates until it succeeds or hits a
budget/time limit.

Prerequisites:
    lake build                  # build all dependencies once at HEAD
    claude --version            # Claude Code CLI must be on PATH

Usage:
    # Evaluate all entries
    python scripts/eval_claude_code.py

    # Quick smoke-test on 3 entries
    python scripts/eval_claude_code.py --limit 3

    # Specific entries
    python scripts/eval_claude_code.py --ids to_bytes_spec_074536e

    # Resume a previous run
    python scripts/eval_claude_code.py --output results/run1.jsonl --resume

    # Dry-run (no claude calls, no lake build)
    python scripts/eval_claude_code.py --dry-run --limit 3

Environment variables:
    ANTHROPIC_API_KEY   required (Claude Code reads this automatically)
"""

import argparse
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
# Layout: PACKAGE_CACHE_DIR/<pkg-name>/<rev>/
PACKAGE_CACHE_DIR = Path.home() / ".cache" / "dalek-lake-packages"

DEFAULT_BUDGET_USD    = 5.00   # max spend per entry
DEFAULT_TIMEOUT       = 500    # hard wall-clock limit for agent subprocess (seconds)
DEFAULT_BUILD_TIMEOUT = None   # no timeout for final verification build

# AGENT_PROMPT_TEMPLATE = """\
# You are inside a Lean 4 project (dalek-lean-bench, a formal verification of \
# the curve25519-dalek Rust library).

# Your task: replace the `sorry` in `{file_path}` for theorem `{theorem_name}` \
# with a correct proof. Even if the theorem is annotated with externally verified \
# in Verus, you should prove it in Lean.

# Run /init to initialze the structure the structure of the library.

# === Workflow ===
# 1. Edit the file: replace `sorry` with your proof attempt. Do not waste too much time in searching
#    proof patterns, which may cause you do no action before the time exceeds.
# 2. Run `nice -n 19 lake build {module}` and read the compiler output.
# 3. If there are errors, fix them and repeat from step 1.
# 4. Stop when the three conditions are satisfied
#    (1) `nice -n 19 lake build {module}` exits with code 0 (no errors), 
#    (2) the `sorry` has been replaced with a proof, 
#    (3) no new `sorry` has been introduced.

# Only edit `{file_path}`. \
# Do NOT modify any other file.
# """



DEBUG_PROMPT_TEMPLATE = """\
You are inside a Lean 4 project (dalek-lean-bench, a formal verification of \
the curve25519-dalek Rust library).

Your task: replace the `sorry` in `Curve25519Dalek/Specs/Edwards/EdwardsPoint/Add.lean` for theorem `add_spec` \
with a correct proof. Even if the theorem is annotated with externally verified \
in Verus, you should prove it in Lean.

Run /init to initialze the structure the structure of the library.

=== Workflow ===
1. Edit the file: replace `sorry` with your proof attempt.
2. Run `nice -n 19 lake build Curve25519Dalek.Specs.Edwards.EdwardsPoint.Add` and read the compiler output.
3. If there are errors, fix them and repeat from step 1.
4. Stop when the three conditions are satisfied
  (1) `nice -n 19 lake build Curve25519Dalek.Specs.Edwards.EdwardsPoint.Add` exits with code 0 (no errors), 
  (2) the `sorry` has been replaced with a proof, 
  (3) no new `sorry` has been introduced.

Only edit `Curve25519Dalek/Specs/Edwards/EdwardsPoint/Add.lean`. \
Do NOT modify any other file.
"""


# ---------------------------------------------------------------------------
# git worktree helpers
# ---------------------------------------------------------------------------

def create_worktree(commit: str) -> Path:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="dalek-cc-eval-"))
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(tmp), commit],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return tmp


def remove_worktree(worktree: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=REPO_ROOT,
        capture_output=True,
    )


def _read_manifest(commit: str) -> dict | None:
    """Read lake-manifest.json from a git commit. Returns None on failure."""
    out = subprocess.run(
        ["git", "show", f"{commit}:lake-manifest.json"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def _setup_worktree_packages(worktree: Path, commit: str) -> None:
    """Set up .lake/packages in the worktree using the content-addressed cache.

    For each package listed in the commit's lake-manifest.json:
    - Cache hit  → symlink PACKAGE_CACHE_DIR/<name>/<rev>/ into the worktree
    - Cache miss → leave the slot empty; Lake will download and compile it

    We do NOT share .lake/build to avoid false-positive cache hits from
    .olean files compiled at HEAD being used for an older commit_before.
    """
    manifest = _read_manifest(commit)
    if manifest is None:
        return

    # packagesDir is relative to the repo root (e.g. ".lake/packages")
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


def _populate_package_cache(worktree: Path, commit: str) -> None:
    """After Lake has run, copy newly downloaded packages into the cache.

    Only real directories (not symlinks) are candidates — symlinked ones are
    already cached. Uses a temp-then-rename strategy so concurrent workers
    writing the same package don't corrupt each other.
    """
    manifest = _read_manifest(commit)
    if manifest is None:
        return

    packages_dir = manifest.get("packagesDir", ".lake/packages")
    wt_packages = worktree / packages_dir

    for pkg in manifest.get("packages", []):
        name = pkg["name"]
        rev  = pkg["rev"]
        pkg_path   = wt_packages / name
        cache_path = PACKAGE_CACHE_DIR / name / rev

        # Skip if already cached or not yet downloaded by Lake
        if cache_path.exists() or not pkg_path.exists() or pkg_path.is_symlink():
            continue

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.parent / f"{rev}.tmp.{os.getpid()}"
        try:
            shutil.copytree(pkg_path, tmp, symlinks=True)
            tmp.rename(cache_path)
        except OSError:
            # Another parallel worker already populated the cache — that's fine
            shutil.rmtree(tmp, ignore_errors=True)


def _seed_cache_from_head() -> None:
    """Populate the package cache from HEAD's already-compiled .lake/packages.

    This is a one-time bootstrap so the first batch of worktrees that share
    HEAD's manifest version get instant cache hits without recompiling.
    """
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
# Final verification build (authoritative pass/fail)
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
        # Lake exits 0 even when `sorry` is present (it's a warning, not an error).
        # A true success requires: exit code 0 AND no sorry in the target file.
        build_ok = proc.returncode == 0
        target = worktree / file_path
        has_sorry = target.exists() and "sorry" in target.read_text(encoding="utf-8", errors="replace")
        return {
            "success": build_ok and not has_sorry,
            "time_s": round(time.time() - t0, 2),
            "stdout": "\n".join(proc.stdout.splitlines()[:300]),
            "stderr": "\n".join(proc.stderr.splitlines()[:300]),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "time_s": round(time.time() - t0, 2),
            "stdout": "",
            "stderr": f"build TIMEOUT after {timeout}s",
        }


# ---------------------------------------------------------------------------
# Claude Code subprocess
# ---------------------------------------------------------------------------

def _format_stream_json_event(line: str, pending_tools: dict[str, str]) -> str | None:
    """Parse one stream-json line and return a human-readable string, or None to skip."""
    line = line.strip()
    if not line:
        return None
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return None  # partial/non-JSON line

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
                    inp_s = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(inp.items())[:3])
                    parts.append(f"[{tool_name}] {inp_s}")
            # skip thinking blocks
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
                        rc = " ".join(c.get("text", "") for c in rc if c.get("type") == "text")
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

    return None  # skip init/debug/other events


def run_claude_code_agent(
    entry: dict,
    worktree: Path,
    budget_usd: float,
    timeout: int,
    model: str | None,
    live: bool = False,
) -> dict:
    """
    Spawn `claude --print --bare` in the worktree.

    Returns a dict with keys: exit_code, stdout, stderr, time_s.
    If `live` is True, stream parsed agent events to the terminal in real-time.
    """
    module = entry["file_path"].replace("/", ".").removesuffix(".lean")
    prompt = AGENT_PROMPT_TEMPLATE.format(
        file_path=entry["file_path"],
        theorem_name=entry["theorem_name"],
        formal_statement=entry["formal_statement"],
        module=module,
    )

    cmd = [
        "claude", "--print",
        "--no-session-persistence",
        "--dangerously-skip-permissions",  # worktree is isolated; no prompt dialogs
        "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep,Skill,Agent,WebFetch",
        "--max-budget-usd", str(budget_usd),
        "--output-format", "stream-json",
        "--verbose",
        prompt,                            # positional argument, not --prompt
    ]
    if model:
        cmd += ["--model", model]

    t0 = time.time()
    timed_out = False

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    prefix = f"[{entry['id']}] " if live else ""

    # shared state for tool_use_id -> tool name matching across lines
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
        "stdout": stdout,
        "stderr": stderr,
        "time_s": round(time.time() - t0, 2),
        "timed_out": timed_out,
    }


# ---------------------------------------------------------------------------
# Extract proof from modified file
# ---------------------------------------------------------------------------

def extract_proof_from_worktree(
    worktree: Path,
    file_path: str,
    formal_statement: str,
) -> str | None:
    """
    Read the (possibly modified) file and extract what replaced `sorry`.
    Returns the text after `:= by` up to the next theorem/def, or None.
    """
    target = worktree / file_path
    if not target.exists():
        return None
    text = target.read_text(encoding="utf-8", errors="replace")

    # Look for the theorem signature without sorry
    if "sorry" in text and formal_statement.split("\n")[0] in text:
        return None  # sorry still present → agent didn't finish

    # Try to find what's after the formal_statement
    sig_first_line = formal_statement.split("\n")[0].strip()
    idx = text.find(sig_first_line)
    if idx == -1:
        return None
    block = text[idx:]
    # Grab everything after `:= by` until next top-level declaration
    m = re.search(r":=\s*by\s*\n(.*?)(?=\n(?:theorem|lemma|def|abbrev|@\[|#|end\b)|\Z)",
                  block, re.DOTALL)
    if m:
        return m.group(1).rstrip()
    return None


# ---------------------------------------------------------------------------
# Single-entry evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    entry: dict,
    budget_usd: float,
    timeout: int,
    build_timeout: int,
    model: str | None,
    dry_run: bool,
    live: bool = False,
) -> dict:
    result: dict = {
        "id": entry["id"],
        "theorem_name": entry["theorem_name"],
        "file_path": entry["file_path"],
        "commit_before": entry["commit_before"],
        "model": model or "default",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "success": False,
        "extracted_proof": None,
        "agent_time_s": None,
        "agent_exit_code": None,
        "agent_timed_out": False,
        "agent_stdout": None,
        "agent_stderr": None,
        "build_time_s": None,
        "build_stdout": "",
        "build_stderr": "",
        "error": None,
    }

    if dry_run:
        result["error"] = "dry-run: skipped"
        return result

    # ── Step 1: worktree at commit_before ────────────────────────────────────
    try:
        worktree = create_worktree(entry["commit_before"])
    except subprocess.CalledProcessError as e:
        result["error"] = f"worktree error: {e.stderr.decode()[:200]}"
        return result

    try:
        # ── Step 1b: symlink cached packages ─────────────────────────────────
        _setup_worktree_packages(worktree, entry["commit_before"])

        # ── Step 2: run Claude Code agent ────────────────────────────────────
        agent_res = run_claude_code_agent(entry, worktree, budget_usd, timeout, model, live=live)
        result["agent_time_s"]    = agent_res["time_s"]
        result["agent_exit_code"] = agent_res["exit_code"]
        result["agent_timed_out"] = agent_res["timed_out"]
        result["agent_stdout"]    = agent_res["stdout"]
        result["agent_stderr"]    = agent_res["stderr"]

        if agent_res["timed_out"]:
            result["error"] = "agent timed out"
            # still attempt verification — agent may have partially succeeded
        elif agent_res["exit_code"] != 0:
            result["error"] = f"claude exited {agent_res['exit_code']}: {agent_res['stderr'][:200]}"

        # ── Step 3: extract what the agent wrote ─────────────────────────────
        result["extracted_proof"] = extract_proof_from_worktree(
            worktree, entry["file_path"], entry["formal_statement"]
        )

        # ── Step 4: authoritative lake build ─────────────────────────────────
        build_res = run_lake_build(worktree, entry["file_path"], timeout=build_timeout)
        result["success"]       = build_res["success"]
        result["build_time_s"]  = build_res["time_s"]
        result["build_stdout"]  = build_res["stdout"]
        result["build_stderr"]  = build_res["stderr"]

        # ── Step 5: populate cache with newly compiled packages ───────────────
        _populate_package_cache(worktree, entry["commit_before"])

    finally:
        remove_worktree(worktree)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=None,
                        help="Claude model alias, e.g. opus or claude-opus-4-6 "
                             "(default: Claude Code's default)")
    parser.add_argument("--dataset", default="dataset.jsonl",
                        help="Input dataset JSONL (default: dataset.jsonl)")
    parser.add_argument("--output", default=None,
                        help="Output JSONL (default: results/claude-code-<ts>.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N entries")
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="Evaluate only these entry IDs")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Wall-clock seconds for agent subprocess (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--build-timeout", type=int, default=DEFAULT_BUILD_TIMEOUT,
                        help="Wall-clock seconds for final verification build (default: no timeout)")
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD,
                        help=f"Max API spend per entry in USD (default: {DEFAULT_BUDGET_USD})")
    parser.add_argument("--resume", action="store_true",
                        help="Append to --output and skip already-evaluated IDs")
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="Number of entries to evaluate concurrently (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip claude calls and lake build")
    parser.add_argument("--live", action="store_true",
                        help="Stream agent stdout/stderr to terminal in real-time (useful for debugging)")
    args = parser.parse_args()

    # Resolve output path
    if args.output is None:
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.output = str(results_dir / f"claude-code-{ts}.jsonl")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset_path = REPO_ROOT / args.dataset
    with open(dataset_path, encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if args.ids:
        id_set = set(args.ids)
        dataset = [e for e in dataset if e["id"] in id_set]

    # Resume
    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        dataset = [e for e in dataset if e["id"] not in done_ids]
        print(f"Resuming: {len(done_ids)} done, {len(dataset)} remaining")

    if args.limit:
        dataset = dataset[: args.limit]

    # Seed the cache from the current HEAD's already-compiled packages so that
    # worktrees sharing the same manifest version get instant cache hits.
    _seed_cache_from_head()

    print(f"Model:      {args.model or '(claude default)'}")
    print(f"Dataset:    {dataset_path}  ({len(dataset)} entries)")
    print(f"Output:     {output_path}")
    build_timeout_str = f"{args.build_timeout}s" if args.build_timeout else "none"
    print(f"Timeout:    {args.timeout}s (agent)  /  {build_timeout_str} (build)")
    print(f"Budget:     ${args.budget_usd:.2f} per entry")
    print(f"Parallel:   {args.parallel}")
    if args.dry_run:
        print("Mode:       DRY RUN")
    if args.live:
        print("Live:       enabled (streaming agent output)")
    print()

    n_pass = n_fail = 0
    total = len(dataset)
    width = len(str(total))

    def _run(entry: dict) -> dict:
        return evaluate_one(
            entry,
            budget_usd=args.budget_usd,
            timeout=args.timeout,
            build_timeout=args.build_timeout,
            model=args.model,
            dry_run=args.dry_run,
            live=args.live,
        )

    with open(output_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_run, entry): entry for entry in dataset}
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

                print(
                    f"[{completed:>{width}}/{total}] "
                    f"{res['id']} ({res['theorem_name']})  "
                    f"{status}  "
                    f"agent={res['agent_time_s']}s  "
                    f"build={res['build_time_s']}s"
                )

                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()

    total = n_pass + n_fail
    pct = f"{100 * n_pass / total:.1f}%" if total else "n/a"
    print()
    print("=== Summary ===")
    print(f"  pass@1:  {n_pass}/{total}  ({pct})")
    print(f"  fail:    {n_fail}/{total}")
    print(f"  output:  {output_path}")


if __name__ == "__main__":
    main()
