#!/usr/bin/env python3
"""
Extract every sorry in a Lean project via the Lean REPL.

For each sorry the REPL returns an elaborated proof-state (goal) string.
We hash that string with SHA-256 to give each sorry a stable, content-based
identity that is independent of file location or line number.

Output
------
  <output_dir>/sorries.jsonl      – one JSON record per sorry
  <output_dir>/sorries_summary.json – aggregate counts

Record schema
-------------
  {
    "id":           "a3f1b2c4d5e6f789",   # first 16 hex chars of sha256(goal)
    "file":         "Curve25519Dalek/Funs.lean",
    "lean_version": "v4.28.0-rc1",
    "location": {
      "start_line": 42, "start_column": 4,
      "end_line":   42, "end_column":   9
    },
    "goal": "⊢ a + b = b + a"
  }

Usage
-----
  python scripts/extract_sorries_repl.py [PROJECT_DIR] [OUTPUT_DIR]

  PROJECT_DIR  defaults to full_proof_recovery_benchmark/ (sibling of scripts/)
  OUTPUT_DIR   defaults to PROJECT_DIR
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT = REPO_ROOT / "full_proof_recovery_benchmark"
REPL_REPO_URL = "https://github.com/leanprover-community/repl"

# Tactic that logs the parent type of the current goal (Prop vs Sort/Type/…)
_PARENT_TYPE_TACTIC = (
    "run_tac (do let t ← Lean.Meta.inferType (← Lean.Elab.Tactic.getMainTarget);"
    ' Lean.logInfo m!"Goal parent type: {t}")'
)


# ---------------------------------------------------------------------------
# Toolchain helpers
# ---------------------------------------------------------------------------

def read_lean_version(project_dir: Path) -> str:
    """Return the version tag from lean-toolchain, e.g. 'v4.28.0-rc1'."""
    tc = project_dir / "lean-toolchain"
    if not tc.exists():
        raise FileNotFoundError(f"lean-toolchain not found in {project_dir}")
    content = tc.read_text().strip()
    # format: leanprover/lean4:vX.Y.Z
    if ":" in content:
        return content.split(":", 1)[1]
    raise ValueError(f"Unexpected lean-toolchain format: {content!r}")


# ---------------------------------------------------------------------------
# REPL setup
# ---------------------------------------------------------------------------

def setup_repl(lean_data: Path, version_tag: str) -> Path:
    """
    Clone and build the Lean REPL at *version_tag* into *lean_data*.

    On subsequent calls the existing binary is returned immediately.
    """
    sanitized = version_tag.replace(".", "_").replace("-", "_")
    repl_dir = lean_data / f"repl_{sanitized}"
    repl_bin = repl_dir / ".lake" / "build" / "bin" / "repl"

    if repl_bin.exists():
        logger.info("Reusing existing REPL binary at %s", repl_bin)
        return repl_bin

    logger.info("Cloning REPL repository into %s …", repl_dir)
    subprocess.run(["git", "clone", REPL_REPO_URL, str(repl_dir)], check=True)

    logger.info("Checking out REPL at tag %s …", version_tag)
    subprocess.run(["git", "checkout", version_tag], cwd=repl_dir, check=True)

    logger.info("Building REPL (this may take a few minutes) …")
    result = subprocess.run(["lake", "build"], cwd=repl_dir)
    if result.returncode != 0:
        raise RuntimeError("lake build failed for REPL")

    if not repl_bin.exists():
        raise FileNotFoundError(f"REPL binary not found at {repl_bin} after build")

    repl_bin.chmod(0o755)
    logger.info("REPL binary ready: %s", repl_bin)
    return repl_bin


# ---------------------------------------------------------------------------
# LeanRepl – thin wrapper around the REPL subprocess
# ---------------------------------------------------------------------------

class LeanRepl:
    """
    Thin wrapper around a `lake env <repl>` subprocess.

    The REPL speaks a simple line-delimited JSON protocol:
      - send a JSON command followed by a blank line
      - read lines until a blank line signals end-of-response
    """

    def __init__(self, project_dir: Path, repl_bin: Path):
        import io
        cmd = ["lake", "env", str(repl_bin.absolute())]
        logger.debug("Starting REPL: %s (cwd=%s)", " ".join(cmd), project_dir)
        proc = subprocess.Popen(
            cmd,
            cwd=project_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # stdin/stdout/stderr are non-None because we passed PIPE above.
        # Store them as typed attributes so type-checkers don't complain in
        # other methods.
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise RuntimeError("Popen did not open expected streams")
        self._proc = proc
        self._stdin: io.TextIOWrapper = proc.stdin   # type: ignore[assignment]
        self._stdout: io.TextIOWrapper = proc.stdout  # type: ignore[assignment]
        self._stderr: io.TextIOWrapper = proc.stderr  # type: ignore[assignment]
        if proc.poll() is not None:
            raise RuntimeError(f"REPL failed to start: {self._stderr.read()}")

    # context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            self._proc.wait()

    def _send(self, command: dict) -> dict:
        """Send *command* and return the parsed JSON response."""
        payload = json.dumps(command) + "\n\n"
        logger.debug("→ REPL: %s", payload.rstrip())

        self._stdin.write(payload)
        self._stdin.flush()

        lines = []
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"REPL died mid-response: {self._stderr.read()}"
                )
            line = self._stdout.readline()
            if not line.strip():
                break
            lines.append(line)

        raw = "".join(lines)
        logger.debug("← REPL: %s", raw.rstrip())
        return json.loads(raw)

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------

    def read_file(self, relative_path: Path) -> list[dict]:
        """
        Ask the REPL to elaborate *relative_path* and return all sorries.

        Each entry:
          {
            "proof_state_id": <int>,
            "location": {start_line, start_column, end_line, end_column},
            "goal": <str>
          }
        """
        resp = self._send({"path": str(relative_path), "allTactics": True})
        sorries = resp.get("sorries", [])
        out = []
        for s in sorries:
            out.append(
                {
                    "proof_state_id": s["proofState"],
                    "location": {
                        "start_line":   s["pos"]["line"],
                        "start_column": s["pos"]["column"],
                        "end_line":     s["endPos"]["line"],
                        "end_column":   s["endPos"]["column"],
                    },
                    "goal": s["goal"],
                }
            )
        return out

    def get_goal_parent_type(self, proof_state_id: int) -> str | None:
        """
        Return 'Prop', 'Type', 'Sort', … for the goal at *proof_state_id*,
        or None if the tactic fails.
        """
        try:
            resp = self._send(
                {"tactic": _PARENT_TYPE_TACTIC, "proofState": proof_state_id}
            )
        except Exception as exc:
            logger.debug("parent-type tactic failed: %s", exc)
            return None

        for msg in resp.get("messages", []):
            if msg.get("severity") == "info" and "Goal parent type:" in msg.get("data", ""):
                return msg["data"].split("Goal parent type:", 1)[1].strip()
        return None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_goal(goal: str, length: int = 16) -> str:
    """SHA-256 of the goal string, first *length* hex chars."""
    return hashlib.sha256(goal.encode()).hexdigest()[:length]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_candidate_files(project_dir: Path) -> list[Path]:
    """
    Return relative paths of .lean files (outside .lake/) that contain 'sorry'.
    """
    candidates = []
    for lean_file in sorted(project_dir.rglob("*.lean")):
        if ".lake" in lean_file.parts:
            continue
        try:
            text = lean_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "sorry" in text:
            candidates.append(lean_file.relative_to(project_dir))
    return candidates


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def extract_all_sorries(
    project_dir: Path,
    repl_bin: Path,
    lean_version: str,
    prop_only: bool = True,
) -> list[dict]:
    """
    Run the REPL over every candidate file and collect sorry records.

    If *prop_only* is True, sorries whose goal is not of type Prop are skipped
    (e.g. term-mode holes inside `def` bodies).
    """
    candidate_files = find_candidate_files(project_dir)
    logger.info(
        "Found %d candidate .lean file(s) containing 'sorry'", len(candidate_files)
    )

    records: list[dict] = []

    for rel_path in candidate_files:
        logger.info("Processing %s …", rel_path)
        try:
            with LeanRepl(project_dir, repl_bin) as repl:
                sorries = repl.read_file(rel_path)

                for sorry in sorries:
                    psid = sorry["proof_state_id"]
                    goal = sorry["goal"]

                    if prop_only:
                        parent_type = repl.get_goal_parent_type(psid)
                        if parent_type != "Prop":
                            logger.debug(
                                "  skipping non-Prop sorry (parent_type=%r) at %s:%d",
                                parent_type,
                                rel_path,
                                sorry["location"]["start_line"],
                            )
                            continue

                    record = {
                        "id":           hash_goal(goal),
                        "file":         str(rel_path),
                        "lean_version": lean_version,
                        "location":     sorry["location"],
                        "goal":         goal,
                    }
                    records.append(record)
                    logger.info(
                        "  sorry id=%s  line=%d  goal=%s",
                        record["id"],
                        sorry["location"]["start_line"],
                        goal[:60].replace("\n", " "),
                    )

        except Exception as exc:
            logger.warning("Error processing %s: %s", rel_path, exc)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "project_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_PROJECT,
        help="Root of the Lean project to scan (default: full_proof_recovery_benchmark/)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Directory for output files (default: same as project_dir)",
    )
    parser.add_argument(
        "--lean-data",
        type=Path,
        default=REPO_ROOT / "lean_data",
        help="Directory where the REPL is cloned/built (default: <repo_root>/lean_data)",
    )
    parser.add_argument(
        "--all",
        dest="all_sorries",
        action="store_true",
        help="Include non-Prop sorries (term-mode holes, etc.)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip 'lake build' of the project (use when oleans are already up to date)",
    )
    parser.add_argument(
        "--skip-repl-setup",
        action="store_true",
        help="Skip cloning/building the REPL; fail if the binary is not already present",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    project_dir: Path = args.project_dir.resolve()
    output_dir: Path = (args.output_dir or project_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not project_dir.exists():
        logger.error("Project directory not found: %s", project_dir)
        sys.exit(1)

    # 1. Read Lean version
    lean_version = read_lean_version(project_dir)
    logger.info("Lean version: %s", lean_version)

    # 2. Set up (or locate) the REPL binary
    args.lean_data.mkdir(parents=True, exist_ok=True)
    if args.skip_repl_setup:
        sanitized = lean_version.replace(".", "_").replace("-", "_")
        repl_bin = args.lean_data / f"repl_{sanitized}" / ".lake" / "build" / "bin" / "repl"
        if not repl_bin.exists():
            logger.error("REPL binary not found at %s", repl_bin)
            sys.exit(1)
    else:
        repl_bin = setup_repl(args.lean_data, lean_version)

    # 3. Build the project so oleans are fresh
    if not args.no_build:
        logger.info("Running 'lake build' in %s …", project_dir)
        result = subprocess.run(["lake", "build"], cwd=project_dir)
        if result.returncode != 0:
            logger.error("lake build failed — aborting")
            sys.exit(1)

    # 4. Extract sorries via REPL
    records = extract_all_sorries(
        project_dir,
        repl_bin,
        lean_version,
        prop_only=not args.all_sorries,
    )

    # 5. Write JSONL
    jsonl_path = output_dir / "sorries.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d sorry record(s) to %s", len(records), jsonl_path)

    # 6. Write summary
    per_file: dict[str, int] = {}
    for rec in records:
        per_file[rec["file"]] = per_file.get(rec["file"], 0) + 1

    unique_ids = len({rec["id"] for rec in records})

    summary = {
        "lean_version": lean_version,
        "total_sorries": len(records),
        "unique_goal_hashes": unique_ids,
        "per_file": per_file,
    }
    summary_path = output_dir / "sorries_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Summary written to %s", summary_path)
    logger.info(
        "Done. %d sorries found (%d unique goals).", len(records), unique_ids
    )


if __name__ == "__main__":
    main()
