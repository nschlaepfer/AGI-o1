#!/usr/bin/env python3
"""
Dual Commit helper that partners with GPT-5-Codex to draft changes, apply diffs, and run validations.

Usage:
    python3 scripts/dual_commit.py "Add retry logging to fetch_data"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Directories
REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "dual_commit_runs"
LOG_DIR.mkdir(exist_ok=True)


def load_environment() -> Dict[str, Any]:
    load_dotenv(REPO_ROOT / ".env")

    env = os.getenv("OPENAI_ENVIRONMENT", "sandbox").strip().lower()
    key_map = {
        "sandbox": os.getenv("OPENAI_API_KEY_SANDBOX"),
        "staging": os.getenv("OPENAI_API_KEY_STAGING"),
        "production": os.getenv("OPENAI_API_KEY_PRODUCTION"),
    }
    api_key = key_map.get(env) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key configured. Populate OPENAI_API_KEY_* or OPENAI_API_KEY."
        )

    return {
        "environment": env,
        "api_key": api_key,
        "model_code": os.getenv("OPENAI_MODEL_CODE", "gpt-5-codex"),
    }


def build_prompt(task: str) -> str:
    schema = textwrap.dedent(
        """
        {
          "status": "ok" | "no_change" | "needs_input",
          "summary": "One sentence overview of the proposed change.",
          "patch": "Unified diff with filenames relative to repo root (no Markdown fences).",
          "validation_commands": ["shell command 1", "..."],
          "notes": ["Additional bullet point context or follow-ups."]
        }
        """
    ).strip()

    guidance = textwrap.dedent(
        f"""
        You are GPT-5-Codex acting as the autonomous leg of a Dual Commit workflow.
        Task: {task}

        Respond in STRICT JSON conforming to this schema (no Markdown, no commentary):
        {schema}

        Requirements:
        - If no code change is needed set status to "no_change", leave patch empty, and explain in notes.
        - Use a single unified diff covering all modified files. Omit code fences.
        - When adding files include the full diff with `+++ b/<path>` and `--- /dev/null` entries.
        - Choose validation_commands that can run locally (e.g., ["pytest -q"]). Use an empty array if nothing applies.
        - Keep notes concise; highlight manual follow-ups or risk items.
        """
    ).strip()
    return guidance


def call_codex(client: OpenAI, model: str, task: str) -> Dict[str, Any]:
    payload = build_prompt(task)
    response = client.responses.create(
        model=model,
        input=payload,
        max_output_tokens=1500,
        temperature=0.2,
    )
    raw_output = response.output_text.strip()

    json_start = raw_output.find("{")
    json_end = raw_output.rfind("}")
    if json_start == -1 or json_end == -1:
        raise ValueError(f"Model response did not contain JSON: {raw_output}")

    json_blob = raw_output[json_start : json_end + 1]
    return json.loads(json_blob)


def apply_patch(patch: str) -> Optional[str]:
    if not patch.strip():
        return None

    process = subprocess.run(
        ["git", "apply", "--whitespace=fix"],
        input=patch.encode("utf-8"),
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"git apply failed:\nSTDOUT:\n{process.stdout.decode()}\nSTDERR:\n{process.stderr.decode()}"
        )
    return process.stdout.decode() or None


def run_validations(commands: List[str]) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for command in commands:
        if not command.strip():
            continue
        completed = subprocess.run(
            command,
            shell=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "command": command,
                "returncode": str(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    return results


def save_run_log(run_data: Dict[str, Any]) -> Path:
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"{timestamp}_dual_commit.json"
    with path.open("w") as handle:
        json.dump(run_data, handle, indent=2)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dual Commit helper powered by GPT-5-Codex.")
    parser.add_argument("task", help="High-level description of the desired change.")
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Skip applying the generated patch (useful for preview).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip running validation commands returned by the model.",
    )
    args = parser.parse_args(argv)

    settings = load_environment()
    client = OpenAI(api_key=settings["api_key"])

    run_record: Dict[str, Any] = {
        "environment": settings["environment"],
        "task": args.task,
        "model": settings["model_code"],
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
    }

    try:
        model_response = call_codex(client, settings["model_code"], args.task)
    except Exception as exc:
        run_record["status"] = "model_error"
        run_record["error"] = str(exc)
        save_path = save_run_log(run_record)
        print(f"Model call failed. Details saved to {save_path}")
        return 1

    run_record["model_response"] = model_response

    status = model_response.get("status")
    patch = model_response.get("patch", "")
    validations = model_response.get("validation_commands", [])

    if status == "no_change":
        run_record["status"] = "no_change"
        save_path = save_run_log(run_record)
        print("Model indicated no change is required.")
        print(f"Details saved to {save_path}")
        return 0

    if not args.no_apply:
        try:
            apply_patch(patch)
            run_record["patch_applied"] = True
            print("Patch applied successfully.")
        except Exception as exc:
            run_record["patch_applied"] = False
            run_record["apply_error"] = str(exc)
            save_path = save_run_log(run_record)
            print("Failed to apply patch; see log for details.")
            print(f"Details saved to {save_path}")
            return 1
    else:
        run_record["patch_applied"] = False
        print("Patch application skipped (--no-apply).")

    validation_results: List[Dict[str, Any]] = []
    if not args.no_validate and isinstance(validations, list):
        validation_results = run_validations(validations)
        run_record["validation_results"] = validation_results
        for result in validation_results:
            command = result["command"]
            code = result["returncode"]
            print(f"[validation] {command} -> exit {code}")
            if result["stdout"]:
                print(result["stdout"])
            if result["stderr"]:
                print(result["stderr"], file=sys.stderr)
    else:
        run_record["validation_results"] = []
        if args.no_validate:
            print("Validation run skipped (--no-validate).")

    run_record["status"] = status or "unknown"
    save_path = save_run_log(run_record)
    print(f"Dual Commit run logged at {save_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
