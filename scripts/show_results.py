#!/usr/bin/env python3
"""
Human-readable viewer for eval result JSONL files.

Usage:
    # Summary table for a run
    python scripts/show_results.py results/claude-code-<ts>.jsonl

    # Full detail for one entry
    python scripts/show_results.py results/claude-code-<ts>.jsonl --id to_bytes_spec_074536e

    # Show all entries (including agent conversation)
    python scripts/show_results.py results/claude-code-<ts>.jsonl --all
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Stream-json parsing
# ---------------------------------------------------------------------------

def extract_agent_events(stdout: str) -> list[dict]:
    """Parse stream-json lines from agent_stdout. Skips partial leading line."""
    events = []
    lines = stdout.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # first line is often truncated mid-stream
    return events


def summarize_agent_stdout(stdout: str) -> str:
    """Extract readable summary from agent stream-json output."""
    if not stdout:
        return "(no output)"

    events = extract_agent_events(stdout)
    if not events:
        return "(could not parse stream-json)"

    parts = []
    retries = 0
    # Map tool_use_id -> tool name for matching results
    pending_tools: dict[str, str] = {}

    for ev in events:
        t = ev.get("type")
        sub = ev.get("subtype", "")

        if t == "system" and sub == "api_retry":
            retries += 1

        elif t == "assistant":
            msg = ev.get("message", {})
            for block in msg.get("content", []):
                btype = block.get("type")
                if btype == "text":
                    text = block["text"].strip()
                    if text:
                        parts.append(f"[assistant] {text}")
                elif btype == "tool_use":
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "?")
                    inp = block.get("input", {})
                    pending_tools[tool_id] = tool_name
                    # Show a concise summary of the call
                    if tool_name == "Bash":
                        desc = inp.get("description") or inp.get("command", "")[:80]
                        parts.append(f"[tool_use] Bash: {desc}")
                    elif tool_name in ("Read", "Write", "Edit", "Glob", "Grep"):
                        key = next(iter(inp), None)
                        val = inp.get(key, "") if key else ""
                        parts.append(f"[tool_use] {tool_name}: {val}")
                    else:
                        inp_summary = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(inp.items())[:3])
                        parts.append(f"[tool_use] {tool_name}({inp_summary})")
                # skip thinking blocks

        elif t == "user":
            msg = ev.get("message", {})
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        tool_id = block.get("tool_use_id", "")
                        tool_name = pending_tools.pop(tool_id, "?")
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = " ".join(
                                c.get("text", "") for c in result_content if c.get("type") == "text"
                            )
                        result_str = str(result_content).strip()
                        # Truncate long results
                        if len(result_str) > 300:
                            result_str = result_str[:300] + " ..."
                        parts.append(f"[tool_result/{tool_name}] {result_str}")

        elif t == "result":
            result_text = ev.get("result", "").strip()
            turns = ev.get("num_turns", "?")
            cost = ev.get("total_cost_usd")
            cost_str = f"  cost=${cost:.4f}" if cost else ""
            parts.append(f"[result/{sub}] turns={turns}{cost_str}\n  {result_text}")

    if retries:
        parts.insert(0, f"[retries] api_retry x{retries}")

    return "\n\n".join(parts) if parts else "(no parseable content in stream-json)"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def status(r: dict) -> str:
    return PASS if r["success"] else FAIL


def fmt_time(s) -> str:
    if s is None:
        return "  -  "
    return f"{s:6.1f}s"


def print_summary(results: list[dict]) -> None:
    n_pass = sum(1 for r in results if r["success"])
    n_total = len(results)
    pct = f"{100 * n_pass / n_total:.1f}%" if n_total else "n/a"

    print(f"{'ID':<40} {'STATUS':<6} {'AGENT':>8} {'BUILD':>7}  ERROR")
    print("-" * 90)
    for r in results:
        err = (r.get("error") or "")[:50]
        print(
            f"{r['id']:<40} {status(r):<15} "
            f"{fmt_time(r.get('agent_time_s')):>8} "
            f"{fmt_time(r.get('build_time_s')):>7}  "
            f"{err}"
        )
    print("-" * 90)
    print(f"pass@1: {n_pass}/{n_total} ({pct})")


def print_detail(r: dict, show_agent: bool = True) -> None:
    sep = "=" * 70
    print(sep)
    print(f"ID:        {r['id']}")
    print(f"Theorem:   {r['theorem_name']}")
    print(f"File:      {r['file_path']}")
    print(f"Commit:    {r['commit_before']}")
    print(f"Model:     {r['model']}")
    print(f"Timestamp: {r['timestamp']}")
    print(f"Status:    {status(r)}")
    print(f"Agent:     {fmt_time(r.get('agent_time_s'))}  exit={r.get('agent_exit_code')}  timed_out={r.get('agent_timed_out')}")
    print(f"Build:     {fmt_time(r.get('build_time_s'))}")

    if r.get("error"):
        print(f"Error:     {r['error']}")

    if r.get("extracted_proof"):
        print()
        print("--- Extracted proof ---")
        print(r["extracted_proof"])

    if r.get("build_stdout") or r.get("build_stderr"):
        print()
        print("--- Build output (last 3000 chars) ---")
        out = (r.get("build_stdout") or "") + (r.get("build_stderr") or "")
        print(out[-3000:].strip())

    if show_agent and r.get("agent_stdout"):
        print()
        print("--- Agent conversation (from truncated stream-json) ---")
        print(summarize_agent_stdout(r["agent_stdout"]))

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", help="Path to result JSONL file")
    parser.add_argument("--id", dest="entry_id", help="Show full detail for this entry ID")
    parser.add_argument("--all", action="store_true", help="Show full detail for every entry")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    if not results:
        print("No results found.")
        return

    if args.entry_id:
        matches = [r for r in results if r["id"] == args.entry_id]
        if not matches:
            print(f"ID not found: {args.entry_id}", file=sys.stderr)
            sys.exit(1)
        for r in matches:
            print_detail(r, show_agent=True)
    elif args.all:
        for r in results:
            print_detail(r, show_agent=True)
    else:
        print_summary(results)
        print()
        print("Use --id <ID> or --all to see full detail including agent conversation.")


if __name__ == "__main__":
    main()
