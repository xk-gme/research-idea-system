"""Bridge from the CLI pipeline to the open-coscientist framework.

open-coscientist (jataware, MIT + Commons Clause) lives in its own
ocs-venv (langgraph/litellm stack), so the main CLI — which must keep
running on project/venv with zero extra dependencies — spawns
ocs_runner.py with the ocs-venv interpreter and adapts its output into
this system's coscientist_output.json format. Their Hypothesis objects
already carry elo_rating / reviews / literature_grounding from the
framework's own tournament; mapping is lossless at the field level.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import SYSTEM_ROOT

OCS_VENV_PYTHON = SYSTEM_ROOT / "ocs-venv" / "Scripts" / "python.exe"
OCS_RUNNER = SYSTEM_ROOT / "ocs_runner.py"


def ocs_available() -> bool:
    return OCS_VENV_PYTHON.exists() and OCS_RUNNER.exists()


def map_ocs_hypotheses(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map open-coscientist Hypothesis dicts into this system's pool shape."""
    mapped: list[dict[str, Any]] = []
    for index, hyp in enumerate(raw, start=1):
        text = str(hyp.get("text", "")).strip()
        if not text:
            continue
        reviews = []
        for review in hyp.get("reviews", []) or []:
            reviews.append({
                "round": None,
                "scores": review.get("scores", {}) or {},
                "verdict": "",
                "comment": review.get("review_summary", ""),
                "overall_score": review.get("overall_score"),
            })
        origin = "ocs:" + str(hyp.get("generation_method") or "generated")
        if hyp.get("evolution_history"):
            origin += "+evolved"
        mapped.append({
            "id": f"H{index}",
            "title": text.split("\n")[0][:100],
            "hypothesis": text,
            "explanation": hyp.get("explanation"),
            "literature_grounding": hyp.get("literature_grounding"),
            "experiment": hyp.get("experiment"),
            "origin": origin,
            "parent_ids": [],
            "elo": hyp.get("elo_rating", 1200),
            "score": hyp.get("score", 0.0),
            "win_count": hyp.get("win_count"),
            "loss_count": hyp.get("loss_count"),
            "total_matches": hyp.get("total_matches"),
            "similarity_cluster_id": hyp.get("similarity_cluster_id"),
            "reviews": reviews,
        })
    return mapped


def run_ocs_coscientist(
    task_text: str,
    config: dict[str, Any],
    run_dir: Path,
    manifest: Any,
    seed_ideas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Spawn ocs_runner.py in ocs-venv and register its pool in the run."""
    if not ocs_available():
        raise RuntimeError(
            "open-coscientist engine unavailable: build ocs-venv first "
            "(see README)."
        )
    ocs_config = config.get("ocs", {})
    deepseek = config["deepseek"]
    model = ocs_config.get(
        "model", f"deepseek/{deepseek.get('model', 'deepseek-flash')}")

    goal = task_text
    if seed_ideas:
        lines = [
            f"{i}. {idea.get('title', '')}: "
            f"{str(idea.get('idea', idea.get('hypothesis', '')))[:500]}"
            for i, idea in enumerate(seed_ideas, start=1)
        ]
        goal += (
            "\n\nThe following ideas already exist from a prior deep-research "
            "stage. Generate hypotheses that go beyond them or recombine "
            "them in genuinely new mechanisms, not restatements:\n"
            + "\n".join(lines)
        )

    goal_file = run_dir / "ocs_goal.txt"
    run_dir.mkdir(parents=True, exist_ok=True)
    goal_file.write_text(goal, encoding="utf-8")
    out_file = run_dir / "ocs_raw_output.json"

    command = [
        str(OCS_VENV_PYTHON), str(OCS_RUNNER),
        "--goal-file", str(goal_file),
        "--out", str(out_file),
        "--key-file", str(deepseek["secret_path"]),
        "--model", model,
        "--max-iterations", str(ocs_config.get("max_iterations", 2)),
        "--initial-count", str(ocs_config.get("initial_hypotheses_count", 7)),
        "--evolve-count", str(ocs_config.get("evolution_max_count", 4)),
    ]
    timeout = int(ocs_config.get("runner_timeout_seconds", 3600))
    manifest.note(f"ocs engine spawn: model={model} timeout={timeout}s")
    manifest.save()
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace", cwd=str(SYSTEM_ROOT),
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-2000:]
        # never echo credentials; the runner only prints sanitized errors
        raise RuntimeError(
            f"ocs_runner failed (exit {completed.returncode}):\n{tail}"
        )

    raw = json.loads(out_file.read_text(encoding="utf-8-sig"))
    hypotheses = map_ocs_hypotheses(raw.get("hypotheses", []))
    if not hypotheses:
        raise RuntimeError("ocs engine produced no hypotheses")
    hypotheses.sort(key=lambda h: h["elo"], reverse=True)

    output = {
        "hypotheses": hypotheses,
        "engine": "open-coscientist",
        "engine_details": {
            "model": raw.get("model"),
            "nodes_completed": raw.get("nodes_completed", []),
            "elapsed_seconds": raw.get("elapsed_seconds"),
        },
        "round_reports": [],
    }
    with open(run_dir / "coscientist_output.json", "w",
              encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    with open(run_dir / "coscientist_report.md", "w", encoding="utf-8") as handle:
        handle.write("# Co-scientist tournament (open-coscientist engine)\n\n")
        handle.write(
            f"- model: {raw.get('model')} | elapsed: "
            f"{raw.get('elapsed_seconds')}s\n"
            f"- nodes: {' -> '.join(raw.get('nodes_completed', []))}\n\n"
        )
        handle.write("| id | title | elo | score | origin | wins |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for hyp in hypotheses:
            handle.write(
                f"| {hyp['id']} | {hyp['title'][:60]} | {hyp['elo']} | "
                f"{hyp['score']} | {hyp['origin']} | "
                f"{hyp.get('win_count')}/{hyp.get('total_matches')} |\n"
            )
    manifest.data["engine"] = "open-coscientist"
    manifest.data["ocs_details"] = output["engine_details"]
    manifest.note(
        f"open-coscientist done: {len(hypotheses)} hypotheses; per-call "
        "usage not centrally tracked by this engine (framework-managed "
        "LiteLLM calls), see ocs_raw_output.json nodes/elapsed"
    )
    manifest.save()
    print(completed.stdout.strip()[-500:] if completed.stdout else "")
    return output
