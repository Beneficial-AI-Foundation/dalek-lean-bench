#!/usr/bin/env python3
"""
LLM evaluation harness for dalek-lean-bench.

For each dataset entry, prompts an LLM to fill the `sorry`, injects the
generated proof into the file, runs `lake build` to verify, and records
pass/fail.

Prerequisites:
    lake build          # build all dependencies first (once)

Usage:
    # Evaluate all entries with Claude Opus
    python scripts/eval.py --model claude-opus-4-6

    # Quick smoke-test on 5 entries
    python scripts/eval.py --model gpt-4o --limit 5

    # Specific entries
    python scripts/eval.py --model claude-opus-4-6 --ids to_bytes_spec_074536e

    # Resume a previous run
    python scripts/eval.py --model claude-opus-4-6 --output results/run1.jsonl --resume

    # Dry-run (test pipeline without API calls)
    python scripts/eval.py --model claude-opus-4-6 --dry-run --limit 3

Environment variables:
    ANTHROPIC_API_KEY   for Claude models (claude-*)
    OPENAI_API_KEY      for OpenAI models (gpt-*, o1-*, o3-*)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover specializing in formal verification "
    "of cryptographic algorithms. You will be given a Lean 4 source file that "
    "contains a theorem with a `sorry` placeholder. Your task is to replace the "
    "`sorry` with a correct, complete proof.\n\n"
    "Output ONLY the proof tactics — the content that replaces `sorry` — with no "
    "explanation, no markdown fences, no extra text. The output must be a valid "
    "Lean 4 tactic block that compiles without errors."
)

USER_TEMPLATE = """\
Complete the Lean 4 proof below. The theorem body currently contains `sorry`.
Provide ONLY the replacement tactics (what goes after `:= by`). No markdown, no explanation.

=== Full file (for context) ===
{file_content_before}

=== Theorem to prove ===
{formal_statement}
  sorry

Proof tactics:\
"""


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


def call_anthropic(model: str, prompt: str, max_tokens: int = 4096) -> str:
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic package not found. Run: pip install anthropic")

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if hasattr(block, "text"):
            return block.text  # type: ignore[union-attr]
    raise ValueError("No text block in response")


def call_openai(model: str, prompt: str, max_tokens: int = 4096) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not found. Run: pip install openai")

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    if content is None:
        raise ValueError("Empty response from OpenAI")
    return content


def call_llm(model: str, prompt: str) -> str:
    if model.startswith("claude"):
        return call_anthropic(model, prompt)
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return call_openai(model, prompt)
    raise ValueError(
        f"Unrecognized model '{model}'. Use a claude-* or gpt-*/o1-*/o3-* model."
    )


# ---------------------------------------------------------------------------
# Proof injection
# ---------------------------------------------------------------------------


def strip_markdown_fences(text: str) -> str:
    """Remove ```lean ... ``` or ``` ... ``` wrappers if the LLM adds them."""
    text = text.strip()
    m = re.match(r"^```(?:lean4?|lean)?\n?(.*?)```\s*$", text, re.DOTALL)
    if m:
        return m.group(1).rstrip()
    return text


def inject_proof(
    file_content: str,
    full_theorem_with_sorry: str,
    formal_statement: str,
    proof: str,
) -> str:
    """
    Replace the sorry-containing theorem block in *file_content* with the
    proved version.

    Raises ValueError if the sorry block cannot be found.
    """
    proof = strip_markdown_fences(proof)
    new_block = formal_statement + "\n" + proof
    if full_theorem_with_sorry in file_content:
        return file_content.replace(full_theorem_with_sorry, new_block, 1)
    raise ValueError(
        "Could not locate `full_theorem_with_sorry` block inside `file_content_before`. "
        "The dataset entry may be malformed."
    )


# ---------------------------------------------------------------------------
# lake build
# ---------------------------------------------------------------------------


def file_path_to_module(file_path: str) -> str:
    """
    Convert a Lean file path to a lake module name.

    Example:
        Curve25519Dalek/Specs/Backend/Serial/U64/Scalar/Scalar52/ToBytes.lean
        → Curve25519Dalek.Specs.Backend.Serial.U64.Scalar.Scalar52.ToBytes
    """
    return file_path.replace("/", ".").removesuffix(".lean")


def run_lake_build(file_path: str, timeout: int = 300) -> dict:
    """
    Run `lake build <module>` and return a result dict:
        success (bool), time_s (float), stdout (str), stderr (str)
    """
    module = file_path_to_module(file_path)
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["lake", "build", module],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": proc.returncode == 0,
            "time_s": round(time.time() - t0, 2),
            # Trim to avoid huge JSONL lines; full output rarely needed
            "stdout": proc.stdout[-3000:] if proc.stdout else "",
            "stderr": proc.stderr[-3000:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "time_s": round(time.time() - t0, 2),
            "stdout": "",
            "stderr": f"TIMEOUT after {timeout}s",
        }


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
        "model": model,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "success": False,
        "llm_proof": None,
        "llm_time_s": None,
        "build_time_s": None,
        "build_stdout": "",
        "build_stderr": "",
        "error": None,
    }

    # ── Step 1: call LLM ────────────────────────────────────────────────────
    prompt = USER_TEMPLATE.format(
        file_content_before=entry["file_content_before"],
        formal_statement=entry["formal_statement"],
    )
    t0 = time.time()
    if dry_run:
        llm_proof = "sorry  -- dry-run placeholder"
    else:
        try:
            llm_proof = call_llm(model, prompt)
        except Exception as exc:
            result["error"] = f"LLM error: {exc}"
            return result

    result["llm_time_s"] = round(time.time() - t0, 2)
    result["llm_proof"] = llm_proof

    # ── Step 2: inject proof ─────────────────────────────────────────────────
    try:
        new_content = inject_proof(
            entry["file_content_before"],
            entry["full_theorem_with_sorry"],
            entry["formal_statement"],
            llm_proof,
        )
    except Exception as exc:
        result["error"] = f"Injection error: {exc}"
        return result

    # ── Step 3: write file → lake build → restore file ──────────────────────
    file_path = REPO_ROOT / entry["file_path"]
    original_content: str | None = None
    if file_path.exists():
        original_content = file_path.read_text(encoding="utf-8")

    try:
        file_path.write_text(new_content, encoding="utf-8")

        if dry_run:
            build_res = {
                "success": False,
                "time_s": 0.0,
                "stdout": "",
                "stderr": "dry-run: lake build skipped",
            }
        else:
            build_res = run_lake_build(entry["file_path"], timeout=timeout)
    finally:
        # Always restore regardless of exceptions
        if original_content is not None:
            file_path.write_text(original_content, encoding="utf-8")

    result["success"] = build_res["success"]
    result["build_time_s"] = build_res["time_s"]
    result["build_stdout"] = build_res["stdout"]
    result["build_stderr"] = build_res["stderr"]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True,
        help="LLM model name, e.g. claude-opus-4-6, gpt-4o",
    )
    parser.add_argument(
        "--dataset", default="dataset.jsonl",
        help="Input dataset JSONL (default: dataset.jsonl)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL path (default: results/<model>-<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after evaluating this many entries",
    )
    parser.add_argument(
        "--ids", nargs="+", metavar="ID",
        help="Evaluate only these entry IDs",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Seconds before lake build is killed (default: 300)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append to --output and skip already-evaluated IDs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip LLM calls and lake build (end-to-end pipeline test)",
    )
    args = parser.parse_args()

    # ── Resolve output path ──────────────────────────────────────────────────
    if args.output is None:
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_model = re.sub(r"[^a-zA-Z0-9._-]", "-", args.model)
        args.output = str(results_dir / f"{safe_model}-{ts}.jsonl")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset_path = REPO_ROOT / args.dataset
    with open(dataset_path, encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if args.ids:
        id_set = set(args.ids)
        dataset = [e for e in dataset if e["id"] in id_set]

    # ── Resume: skip already-done IDs ────────────────────────────────────────
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

    # ── Print header ─────────────────────────────────────────────────────────
    print(f"Model:    {args.model}")
    print(f"Dataset:  {dataset_path}  ({len(dataset)} entries)")
    print(f"Output:   {output_path}")
    print(f"Timeout:  {args.timeout}s / lake build")
    if args.dry_run:
        print("Mode:     DRY RUN (no LLM calls, no lake build)")
    print()

    # ── Evaluation loop ───────────────────────────────────────────────────────
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
                f"llm={res['llm_time_s']}s  "
                f"build={res['build_time_s']}s"
            )

            out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
            out_f.flush()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = n_pass + n_fail
    pct = f"{100 * n_pass / total:.1f}%" if total else "n/a"
    print()
    print("=== Summary ===")
    print(f"  pass@1:  {n_pass}/{total}  ({pct})")
    print(f"  fail:    {n_fail}/{total}")
    print(f"  output:  {output_path}")


if __name__ == "__main__":
    main()
