#!/usr/bin/env python3
"""
Agentic LLM evaluation harness for dalek-lean-bench.

Each dataset entry is evaluated inside a git worktree checked out at
`commit_before`, so the LLM sees exactly the project state when the sorry
existed.  The LLM is given file-reading tools (read_file, list_files,
search_files) and runs a tool-use loop until it either produces a proof or
exhausts its turn budget.

Prerequisites:
    lake build          # build all dependencies once at HEAD

Usage:
    # Evaluate all entries with Claude
    python scripts/eval_agentic.py --model claude-opus-4-6

    # Quick smoke-test on 3 entries
    python scripts/eval_agentic.py --model claude-opus-4-6 --limit 3

    # Specific entries
    python scripts/eval_agentic.py --model claude-opus-4-6 --ids to_bytes_spec_074536e

    # Resume a previous run
    python scripts/eval_agentic.py --model claude-opus-4-6 --output results/run1.jsonl --resume

    # Dry-run (no LLM calls, no lake build)
    python scripts/eval_agentic.py --model claude-opus-4-6 --dry-run --limit 3

Environment variables:
    ANTHROPIC_API_KEY   required for claude-* models
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MAX_TURNS = 20          # max tool-call rounds per entry
MAX_FILE_BYTES = 80_000  # truncate large files sent to LLM

SYSTEM_PROMPT = """\
You are an expert Lean 4 theorem prover specializing in formal verification \
of cryptographic algorithms.

You have access to tools that let you explore the project at the exact git \
state when the sorry was introduced:
  - list_files: browse the directory tree
  - read_file:  read any project file (Lean sources, lakefile, …)
  - search_files: grep for a pattern across the project

Use these tools to:
  1. Understand the function being proved (find its definition in Funs.lean)
  2. Find similar proved theorems for tactic patterns
  3. Locate relevant lemmas in Math/ files

When you have enough context, call the `submit_proof` tool with the tactic \
block that replaces `sorry`.  Output ONLY valid Lean 4 tactics — no markdown, \
no explanation.
"""

USER_TEMPLATE = """\
Complete the Lean 4 proof below.  The theorem body currently contains `sorry`.
Use the provided tools to explore the project, then call `submit_proof` with \
the replacement tactics (what goes after `:= by`).

=== Theorem to prove ===
{formal_statement}
  sorry

=== File it lives in: {file_path} ===
{file_content_before}
"""

# ---------------------------------------------------------------------------
# Tool definitions (sent to the API)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "list_files",
        "description": (
            "List files in a directory of the project. "
            "Returns relative paths from the project root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Relative path from project root, e.g. 'Curve25519Dalek/Specs'",
                },
            },
            "required": ["directory"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a project file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from project root, e.g. 'Curve25519Dalek/Funs.lean'",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to return (1-indexed, optional).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to return (inclusive, optional).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for a regex pattern across project files. "
            "Returns matching lines with file and line number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for.",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter, e.g. '*.lean' (default: '*.lean')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return (default 50).",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "submit_proof",
        "description": (
            "Submit the completed proof tactics. "
            "Call this once you are confident in the proof."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tactics": {
                    "type": "string",
                    "description": "The tactic block that replaces `sorry` (after `:= by`).",
                },
            },
            "required": ["tactics"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations (executed against the worktree)
# ---------------------------------------------------------------------------

def tool_list_files(worktree: Path, directory: str) -> str:
    target = worktree / directory
    if not target.exists():
        return f"ERROR: directory not found: {directory}"
    files = sorted(
        str(p.relative_to(worktree))
        for p in target.rglob("*")
        if p.is_file()
    )
    return "\n".join(files) if files else "(empty)"


def tool_read_file(
    worktree: Path,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    target = worktree / path
    if not target.exists():
        return f"ERROR: file not found: {path}"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"ERROR reading file: {e}"

    lines = text.splitlines()
    total = len(lines)

    if start_line is not None or end_line is not None:
        s = (start_line or 1) - 1
        e = end_line or total
        lines = lines[s:e]
        header = f"[{path}  lines {s+1}-{min(e, total)} of {total}]\n"
    else:
        header = f"[{path}  {total} lines]\n"

    content = "\n".join(lines)
    # Truncate if huge
    if len(content) > MAX_FILE_BYTES:
        content = content[:MAX_FILE_BYTES] + f"\n... (truncated at {MAX_FILE_BYTES} bytes)"
    return header + content


def tool_search_files(
    worktree: Path,
    pattern: str,
    glob: str = "*.lean",
    max_results: int = 50,
) -> str:
    try:
        proc = subprocess.run(
            ["rg", "--glob", glob, "-n", "--no-heading", pattern],
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = proc.stdout.splitlines()
    except FileNotFoundError:
        # Fall back to grep if ripgrep not available
        try:
            proc = subprocess.run(
                ["grep", "-r", "--include=" + glob, "-n", pattern, "."],
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=15,
            )
            lines = proc.stdout.splitlines()
        except Exception as e:
            return f"ERROR: search failed: {e}"
    except subprocess.TimeoutExpired:
        return "ERROR: search timed out"

    if not lines:
        return "(no matches)"
    if len(lines) > max_results:
        lines = lines[:max_results]
        lines.append(f"... (truncated to {max_results} results)")
    return "\n".join(lines)


def dispatch_tool(worktree: Path, name: str, inputs: dict) -> str:
    if name == "list_files":
        return tool_list_files(worktree, inputs["directory"])
    if name == "read_file":
        return tool_read_file(
            worktree,
            inputs["path"],
            inputs.get("start_line"),
            inputs.get("end_line"),
        )
    if name == "search_files":
        return tool_search_files(
            worktree,
            inputs["pattern"],
            inputs.get("glob", "*.lean"),
            inputs.get("max_results", 50),
        )
    return f"ERROR: unknown tool {name}"


# ---------------------------------------------------------------------------
# git worktree helpers
# ---------------------------------------------------------------------------

def create_worktree(commit: str) -> Path:
    """
    Create a detached worktree at `commit` in a temp directory.
    Returns the worktree path.
    """
    tmp = Path(tempfile.mkdtemp(prefix="dalek-eval-"))
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


# ---------------------------------------------------------------------------
# lake build (runs inside worktree, but shares .lake cache via symlink)
# ---------------------------------------------------------------------------

def _ensure_lake_cache_symlink(worktree: Path) -> None:
    """
    Point worktree/.lake → REPO_ROOT/.lake so compiled dependencies are reused.
    Only the spec files change between commits, so this is safe.
    """
    wt_lake = worktree / ".lake"
    repo_lake = REPO_ROOT / ".lake"
    if not wt_lake.exists() and repo_lake.exists():
        wt_lake.symlink_to(repo_lake)


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
# Proof injection (same logic as eval.py)
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:lean4?|lean)?\n?(.*?)```\s*$", text, re.DOTALL)
    return m.group(1).rstrip() if m else text


def inject_proof(
    file_content: str,
    full_theorem_with_sorry: str,
    formal_statement: str,
    proof: str,
) -> str:
    proof = strip_markdown_fences(proof)
    new_block = formal_statement + "\n" + proof
    if full_theorem_with_sorry in file_content:
        return file_content.replace(full_theorem_with_sorry, new_block, 1)
    raise ValueError(
        "Could not locate full_theorem_with_sorry block inside file_content."
    )


# ---------------------------------------------------------------------------
# Agentic loop (Claude tool-use)
# ---------------------------------------------------------------------------

def run_agentic_loop(
    entry: dict,
    worktree: Path,
    model: str,
    max_turns: int = MAX_TURNS,
) -> tuple[str | None, list, int]:
    """
    Run the tool-use agent loop.

    Returns:
        (proof_tactics | None, messages, turns_used)
    """
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic package not found. Run: pip install anthropic")

    client = anthropic.Anthropic()

    messages = [
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                formal_statement=entry["formal_statement"],
                file_path=entry["file_path"],
                file_content_before=entry["file_content_before"],
            ),
        }
    ]

    proof_tactics: str | None = None
    turns = 0

    while turns < max_turns:
        turns += 1
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Agent finished without calling submit_proof — unusual
            break

        if response.stop_reason != "tool_use":
            break

        # Process tool calls
        tool_results = []
        submitted = False

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "submit_proof":
                proof_tactics = str(block.input.get("tactics", ""))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Proof submitted. Verification will now run.",
                })
                submitted = True
            else:
                result_text = dispatch_tool(worktree, block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        messages.append({"role": "user", "content": tool_results})

        if submitted:
            break

    return proof_tactics, messages, turns


# ---------------------------------------------------------------------------
# Single-entry evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    entry: dict,
    model: str,
    timeout: int,
    dry_run: bool = False,
) -> dict:
    result: dict = {
        "id": entry["id"],
        "theorem_name": entry["theorem_name"],
        "file_path": entry["file_path"],
        "commit_before": entry["commit_before"],
        "model": model,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "success": False,
        "llm_proof": None,
        "turns_used": 0,
        "llm_time_s": None,
        "build_time_s": None,
        "build_stdout": "",
        "build_stderr": "",
        "error": None,
    }

    if dry_run:
        result["error"] = "dry-run: skipped"
        return result

    # ── Step 1: create worktree at commit_before ─────────────────────────────
    try:
        worktree = create_worktree(entry["commit_before"])
    except subprocess.CalledProcessError as e:
        result["error"] = f"worktree error: {e.stderr.decode()[:200]}"
        return result

    try:
        # ── Step 2: agentic loop ──────────────────────────────────────────────
        t0 = time.time()
        try:
            proof_tactics, _messages, turns = run_agentic_loop(
                entry, worktree, model
            )
        except Exception as exc:
            result["error"] = f"LLM error: {exc}"
            return result

        result["llm_time_s"] = round(time.time() - t0, 2)
        result["turns_used"] = turns
        result["llm_proof"] = proof_tactics

        if proof_tactics is None:
            result["error"] = "Agent did not call submit_proof"
            return result

        # ── Step 3: inject proof into worktree file ───────────────────────────
        wt_file = worktree / entry["file_path"]
        try:
            original = wt_file.read_text(encoding="utf-8")
            new_content = inject_proof(
                original,
                entry["full_theorem_with_sorry"],
                entry["formal_statement"],
                proof_tactics,
            )
            wt_file.write_text(new_content, encoding="utf-8")
        except Exception as exc:
            result["error"] = f"Injection error: {exc}"
            return result

        # ── Step 4: lake build inside worktree ────────────────────────────────
        build_res = run_lake_build(worktree, entry["file_path"], timeout=timeout)
        result["success"] = build_res["success"]
        result["build_time_s"] = build_res["time_s"]
        result["build_stdout"] = build_res["stdout"]
        result["build_stderr"] = build_res["stderr"]

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
    parser.add_argument("--model", required=True,
                        help="Claude model, e.g. claude-opus-4-6")
    parser.add_argument("--dataset", default="dataset.jsonl",
                        help="Input dataset JSONL (default: dataset.jsonl)")
    parser.add_argument("--output", default=None,
                        help="Output JSONL (default: results/<model>-agentic-<ts>.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N entries")
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="Evaluate only these entry IDs")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Seconds before lake build is killed (default: 300)")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS,
                        help=f"Max tool-call rounds per entry (default: {MAX_TURNS})")
    parser.add_argument("--resume", action="store_true",
                        help="Append to --output and skip already-evaluated IDs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls and lake build")
    args = parser.parse_args()

    # Resolve output path
    if args.output is None:
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_model = re.sub(r"[^a-zA-Z0-9._-]", "-", args.model)
        args.output = str(results_dir / f"{safe_model}-agentic-{ts}.jsonl")

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

    print(f"Model:     {args.model}")
    print(f"Dataset:   {dataset_path}  ({len(dataset)} entries)")
    print(f"Output:    {output_path}")
    print(f"Timeout:   {args.timeout}s / lake build")
    print(f"Max turns: {args.max_turns}")
    if args.dry_run:
        print("Mode:      DRY RUN")
    print()

    n_pass = n_fail = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, entry in enumerate(dataset):
            print(
                f"[{i+1:>{len(str(len(dataset)))}}/{len(dataset)}] "
                f"{entry['id']} ({entry['theorem_name']})...",
                end=" ",
                flush=True,
            )
            res = evaluate_one(entry, args.model, args.timeout, args.dry_run)

            if res["success"]:
                n_pass += 1
                status = "PASS"
            else:
                n_fail += 1
                err_hint = res.get("error") or ""
                status = f"FAIL  {err_hint[:60]}"

            print(
                f"{status}  "
                f"turns={res['turns_used']}  "
                f"llm={res['llm_time_s']}s  "
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
