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
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BUDGET_USD = 0.50   # max spend per entry
DEFAULT_TIMEOUT    = 600    # hard wall-clock limit per entry (seconds)

AGENT_PROMPT_TEMPLATE = """\
You are inside a Lean 4 project (dalek-lean-bench, a formal verification of \
the curve25519-dalek Rust library).

Your task: replace the `sorry` in `{file_path}` for theorem `{theorem_name}` \
with a correct proof.

=== Theorem ===
{formal_statement}
  sorry

=== Workflow ===
1. Read `{file_path}` to understand the context.
2. Search for similar proved theorems nearby to find tactic patterns \
   (use Grep or Bash with rg).
3. Edit the file: replace `sorry` with your proof attempt.
4. Run `lake build {module}` and read the compiler output.
5. If there are errors, fix them and repeat from step 4.
6. Stop when `lake build {module}` exits with code 0 (no errors).

Do NOT explain your reasoning. Only edit `{file_path}`. \
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


def _ensure_lake_cache_symlink(worktree: Path) -> None:
    """Symlink only .lake/packages (external deps) into the worktree.

    We intentionally do NOT share .lake/build: those .olean files were compiled
    from the current HEAD and would cause false-positive cache hits when the
    worktree is checked out at an older commit_before.  Each worktree builds
    its own project .oleans from scratch; parallel runs compensate for the cost.
    """
    repo_lake = REPO_ROOT / ".lake"
    if not repo_lake.exists():
        return

    wt_lake = worktree / ".lake"
    wt_lake.mkdir(exist_ok=True)

    repo_packages = repo_lake / "packages"
    if repo_packages.exists():
        wt_packages = wt_lake / "packages"
        if not wt_packages.exists():
            wt_packages.symlink_to(repo_packages)


# ---------------------------------------------------------------------------
# Final verification build (authoritative pass/fail)
# ---------------------------------------------------------------------------

def run_lake_build(worktree: Path, file_path: str, timeout: int = 300) -> dict:
    _ensure_lake_cache_symlink(worktree)
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
        return {
            "success": proc.returncode == 0,
            "time_s": round(time.time() - t0, 2),
            "stdout": proc.stdout[-3000:],
            "stderr": proc.stderr[-3000:],
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "time_s": round(time.time() - t0, 2),
            "stdout": "",
            "stderr": f"TIMEOUT after {timeout}s",
        }


# ---------------------------------------------------------------------------
# Claude Code subprocess
# ---------------------------------------------------------------------------

def run_claude_code_agent(
    entry: dict,
    worktree: Path,
    budget_usd: float,
    timeout: int,
    model: str | None,
) -> dict:
    """
    Spawn `claude --print --bare` in the worktree.

    Returns a dict with keys: exit_code, stdout, stderr, time_s.
    """
    _ensure_lake_cache_symlink(worktree)

    module = entry["file_path"].replace("/", ".").removesuffix(".lean")
    prompt = AGENT_PROMPT_TEMPLATE.format(
        file_path=entry["file_path"],
        theorem_name=entry["theorem_name"],
        formal_statement=entry["formal_statement"],
        module=module,
    )

    cmd = [
        "claude", "--print",
        "--bare",                          # skip hooks/memory/plugins for reproducibility
        "--no-session-persistence",
        "--dangerously-skip-permissions",  # worktree is isolated; no prompt dialogs
        "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep",
        "--max-budget-usd", str(budget_usd),
        "--output-format", "json",
        "--prompt", prompt,
    ]
    if model:
        cmd += ["--model", model]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(worktree),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-5000:],
            "stderr": proc.stderr[-2000:],
            "time_s": round(time.time() - t0, 2),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"subprocess TIMEOUT after {timeout}s",
            "time_s": round(time.time() - t0, 2),
            "timed_out": True,
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
    model: str | None,
    dry_run: bool,
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
        # ── Step 2: run Claude Code agent ────────────────────────────────────
        agent_res = run_claude_code_agent(entry, worktree, budget_usd, timeout, model)
        result["agent_time_s"]    = agent_res["time_s"]
        result["agent_exit_code"] = agent_res["exit_code"]
        result["agent_timed_out"] = agent_res["timed_out"]

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
        build_res = run_lake_build(worktree, entry["file_path"], timeout=120)
        result["success"]       = build_res["success"]
        result["build_time_s"]  = build_res["time_s"]
        result["build_stdout"]  = build_res["stdout"]
        result["build_stderr"]  = build_res["stderr"]

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
                        help=f"Wall-clock seconds per entry (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD,
                        help=f"Max API spend per entry in USD (default: {DEFAULT_BUDGET_USD})")
    parser.add_argument("--resume", action="store_true",
                        help="Append to --output and skip already-evaluated IDs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip claude calls and lake build")
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

    print(f"Model:      {args.model or '(claude default)'}")
    print(f"Dataset:    {dataset_path}  ({len(dataset)} entries)")
    print(f"Output:     {output_path}")
    print(f"Timeout:    {args.timeout}s per entry")
    print(f"Budget:     ${args.budget_usd:.2f} per entry")
    if args.dry_run:
        print("Mode:       DRY RUN")
    print()

    n_pass = n_fail = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, entry in enumerate(dataset):
            print(
                f"[{i+1:>{len(str(len(dataset)))}}/{len(dataset)}] "
                f"{entry['id']} ({entry['theorem_name']})...",
                end=" ", flush=True,
            )
            res = evaluate_one(
                entry,
                budget_usd=args.budget_usd,
                timeout=args.timeout,
                model=args.model,
                dry_run=args.dry_run,
            )

            if res["success"]:
                n_pass += 1
                status = "PASS"
            else:
                n_fail += 1
                err_hint = res.get("error") or ""
                status = f"FAIL  {err_hint[:60]}"

            print(
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
