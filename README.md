<p align="center">
<img
 alt="dalek-cryptography logo"
 width="160px"
 src="https://cdn.jsdelivr.net/gh/dalek-cryptography/curve25519-dalek/docs/assets/dalek-logo-clear.png"/>
</p>

# dalek-lean-bench

A benchmark that simulates the **real-world software verification workflow** by replaying the historical order in which theorems were proven in the [curve25519-dalek Lean verification project](https://github.com/beneficial-ai-foundation/curve25519-dalek-lean-verify).

---

## Motivation

Most theorem-proving benchmarks evaluate models on isolated problems in a vacuum. Real verification work is different: a verifier always has a growing body of already-proven lemmas to build on, and the difficulty of each new theorem depends on what has already been established.

This benchmark captures that structure. It uses the **commit-by-commit history** of a real, ongoing Lean 4 verification project — formally verifying [curve25519-dalek](https://github.com/dalek-cryptography/curve25519-dalek), a widely-used Rust implementation of elliptic curve cryptography. Each entry in the benchmark corresponds to a theorem that was proven at a specific point in the project's development. When evaluating a model on theorem `B` at timeline position `i`:

- Theorems `0 … i-1` (proven **before** `B`) retain their full proofs — available as established lemmas.
- Theorem `B` itself has `sorry` injected — this is the target the model must prove.
- Theorems `i+1 … N` (proven **after** `B`) also have `sorry` injected — simulating that they do not yet exist.

This mirrors the proof context that was actually available when `B` was originally written, making it the most realistic evaluation setting for agentic theorem proving.

---

## Dataset at a Glance

| Artifact | Description |
|---|---|
| `proof_timeline.csv` | 249 theorems in the order they were historically proven |
| `dataset.jsonl` | 146 theorem–proof pairs extracted from git history |
| `Curve25519Dalek/Specs/` | ~150 hand-written Lean 4 spec files (formal statements) |
| `Curve25519Dalek/Funs.lean` | Auto-generated Lean translation of Rust via Aeneas |
| `full_proof_recovery_benchmark/` | Generated snapshot: all proofs replaced by `sorry` (gitignored) |

The underlying Lean project covers the full `curve25519-dalek` library: field arithmetic over GF(2²⁵⁵−19), scalar arithmetic, Edwards/Montgomery/Ristretto group operations, and scalar multiplication.

---

## How It Works

```
curve25519-dalek (Rust)
        │
        │  Aeneas (Rust → Lean extraction)
        ▼
Curve25519Dalek/Funs.lean          ← auto-generated, never edited
        │
        │  Human/LLM writes formal specs + proofs
        ▼
Curve25519Dalek/Specs/**/*.lean    ← one file per Rust function
        │
        │  git history records when each proof was completed
        ▼
proof_timeline.csv                 ← 249-entry ordered timeline
        │
        │  eval_timeline.py: replay history, inject sorry, run agent
        ▼
results/*.jsonl                    ← pass/fail per theorem, per model
```

### Evaluation loop (per theorem)

1. Create an isolated **git worktree** at HEAD.
2. **Inject `sorry`** into the target theorem and all theorems proven after it.
3. Run the **LLM agent** (e.g., Claude Code CLI) with a budget cap; the agent edits the spec file and runs `lake build` iteratively.
4. Record **pass/fail**, agent wall time, token cost, and build output.
5. Clean up the worktree.

---

## Full Proof Recovery Benchmark

In addition to the timeline evaluation, the repository ships a script that generates a simpler **full proof recovery** benchmark: a standalone, buildable copy of the entire Lean project with every theorem and lemma proof body replaced by `sorry`.

```bash
python scripts/make_full_proof_recovery_benchmark.py
```

This writes to `full_proof_recovery_benchmark/` (gitignored). The directory is a complete Lake project — it copies `lakefile.toml`, `lean-toolchain`, and `lake-manifest.json` on first run, then updates only the `.lean` files on subsequent runs.

**Use case**: evaluating a model's ability to fill in any single proof given the full library context, without the temporal ordering constraint of the timeline benchmark.

---

## Sorry Extraction via the Lean REPL

`scripts/extract_sorries_repl.py` uses the [Lean REPL](https://github.com/leanprover-community/repl) to elaborate every `sorry` in a project and capture its **proof state** — the full elaborated goal string as Lean sees it, not just the raw source text.

Each sorry is assigned a **stable content-based id**: the first 16 hex characters of `sha256(goal_string)`. Because the id depends only on the elaborated goal and not on file path or line number, the same sorry retains the same id across refactors and renames.

```bash
# Scan full_proof_recovery_benchmark/ (default)
python scripts/extract_sorries_repl.py

# Scan a different project, write output elsewhere
python scripts/extract_sorries_repl.py path/to/project output/

# Include non-Prop sorries (term-mode holes in def bodies)
python scripts/extract_sorries_repl.py --all

# Skip lake build when oleans are already up to date
python scripts/extract_sorries_repl.py --no-build
```

On first run the script clones and builds the REPL at the version matching `lean-toolchain`; subsequent runs reuse the cached binary from `lean_data/`.

**Outputs** (written into the project directory by default):

| File | Contents |
|---|---|
| `sorries.jsonl` | One JSON record per sorry (see schema below) |
| `sorries_summary.json` | Total count, unique goal hashes, and per-file breakdown |

**Record schema** (`sorries.jsonl`):

```jsonc
{
  "id":           "a3f1b2c4d5e6f789",   // first 16 hex chars of sha256(goal)
  "file":         "Curve25519Dalek/Funs.lean",
  "lean_version": "v4.28.0-rc1",
  "location": {
    "start_line": 42, "start_column": 4,
    "end_line":   42, "end_column":   9
  },
  "goal": "⊢ a + b = b + a"
}
```

By default only `Prop`-typed goals are included (theorem / lemma bodies). Pass `--all` to also capture term-mode holes.

---

## Quick Start

### Prerequisites

```bash
# Lean 4 / Lake (see lean-toolchain for exact version)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Build all Lean dependencies once (takes ~10–30 min on first run)
lake build

# Claude Code CLI (required for the default agent)
npm install -g @anthropic-ai/claude-code
export ANTHROPIC_API_KEY=sk-...
```

### Run the timeline evaluation

```bash
# Smoke-test: first 5 timeline entries
python scripts/eval_timeline.py --limit 5

# Single theorem by ID
python scripts/eval_timeline.py --ids tl_0007_add

# Full run (parallel, resumable)
python scripts/eval_timeline.py --output results/run-$(date +%Y%m%d).jsonl

# Resume an interrupted run
python scripts/eval_timeline.py --output results/run-20260101.jsonl --resume

# Dry-run (no API calls, no lake build — just inspect worktree setup)
python scripts/eval_timeline.py --dry-run --setup-only --limit 3 --keep-worktree

# Stream agent output live
python scripts/eval_timeline.py --live --limit 2
```

### Explore the timeline

```bash
# List all 249 evaluation IDs with theorem names
python scripts/eval_timeline.py --list
```

### Inspect results

```bash
# Summary table for a completed run
python scripts/show_results.py results/run-20260415.jsonl

# Full detail for one entry (includes agent conversation)
python scripts/show_results.py results/run-20260415.jsonl --id tl_0007_add
```

---

## Repository Structure

```
.
├── Curve25519Dalek/
│   ├── Funs.lean              # Auto-generated Lean translation of Rust (Aeneas output)
│   ├── Types.lean             # Auto-generated type definitions
│   ├── Specs/                 # Formal specifications (one file per Rust function)
│   ├── Math/                  # Supporting math: Edwards, Montgomery, Ristretto
│   ├── Aux.lean               # Shared auxiliary lemmas
│   └── Tactics.lean           # Custom Lean tactics
├── curve25519-dalek/          # Rust source (git submodule, minimal modifications)
├── scripts/
│   ├── eval_timeline.py                    # Main benchmark harness (timeline-ordered evaluation)
│   ├── eval.py                             # Basic single-shot LLM evaluation
│   ├── eval_claude_code.py                 # Agentic evaluation (head / commit-before modes)
│   ├── gen_proof_timeline.py               # Regenerate proof_timeline.csv from git history
│   ├── extract_dataset.py                  # Extract dataset.jsonl from git history
│   ├── make_full_proof_recovery_benchmark.py  # Generate full_proof_recovery_benchmark/
│   ├── extract_sorries_repl.py             # Extract & hash every sorry via the Lean REPL
│   └── show_results.py                     # Human-readable viewer for result JSONL files
├── proof_timeline.csv         # 249-entry ordered proof history
├── dataset.jsonl              # 146 theorem–proof pairs for training/eval
├── status.csv                 # Current verification status of all 197 functions
└── extraction_notes.md        # Notes on Aeneas extraction limitations
```

---

## Evaluation Scripts

| Script | Purpose |
|---|---|
| `eval_timeline.py` | **Primary benchmark**: timeline-ordered agentic evaluation with sorry injection |
| `eval_claude_code.py` | Agentic evaluation in "head" or "commit-before" mode |
| `eval.py` | Single-shot LLM evaluation on `dataset.jsonl` entries |
| `gen_proof_timeline.py` | Rebuild `proof_timeline.csv` by walking git history |
| `extract_dataset.py` | Build `dataset.jsonl` by scanning git history for sorry→proof transitions |
| `make_full_proof_recovery_benchmark.py` | Generate `full_proof_recovery_benchmark/` — a standalone buildable project with all proofs replaced by `sorry` |
| `extract_sorries_repl.py` | Extract every `sorry` via the Lean REPL; hash each proof state with SHA-256 for a stable content-based id |
| `show_results.py` | Pretty-print a `results/*.jsonl` file as a summary table or per-entry detail |

---

## Output Format

Each `results/*.jsonl` file has one JSON object per evaluated theorem:

```jsonc
{
  "id": "tl_0007_add",
  "lean_name": "curve25519_dalek.backend.serial.u64.field.FieldElement51.add",
  "spec_theorem": "Curve25519Dalek/Specs/Backend/.../Add.lean",
  "success": true,
  "agent_time_s": 47.2,
  "build_stdout": "...",
  "build_stderr": "",
  "timestamp": "2026-04-13T10:00:00Z"
}
```

---

## Underlying Verification Project

The benchmark is built on top of the [curve25519-dalek Lean verification project](https://github.com/beneficial-ai-foundation/curve25519-dalek-lean-verify). That project:

- Uses [Aeneas](https://github.com/AeneasVerif/aeneas) to automatically translate Rust to Lean 4
- Follows the Aeneas WP (weakest precondition) style for specs
- Has verified **197 Rust functions** from curve25519-dalek as of the initial benchmark snapshot
- Is maintained by Oliver Butterley and [The Beneficial AI Foundation](https://www.beneficialaifoundation.org/)

---

## Dependencies

| Dependency | Role |
|---|---|
| Lean 4 (`lean-toolchain`) | Proof checker and build system |
| [Aeneas](https://github.com/AeneasVerif/aeneas) | Rust → Lean extraction (Lean package) |
| [Mathlib](https://github.com/leanprover-community/mathlib4) | Mathematical library |
| [PrimeCert](https://github.com/BoltonBailey/formal-snarks-project) | Cryptographic certificates |
| Claude Code CLI | Default LLM agent for `eval_timeline.py` |
| Python 3.7+ | Evaluation harness scripts |

---

## License

The Lean verification code is licensed under the [Apache License 2.0](LICENSE-APACHE).

The `curve25519-dalek` Rust source (in `curve25519-dalek/`) is dual-licensed under [Apache 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT).
