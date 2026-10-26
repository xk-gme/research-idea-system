"""Task background loading.

A task file is the verbatim S15-style background prompt for one scientific
task (problem description, principles, constraints, SUPERHUMAN request).
The paper supplied "background information from the main problem
description" to both Deep Research and AI co-scientist (Methods); these
files are that background.
"""

from __future__ import annotations

from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "任务背景"


def available_tasks() -> list[str]:
    if not TASKS_DIR.is_dir():
        return []
    return sorted(p.stem for p in TASKS_DIR.glob("*.md"))


def load_task(name: str) -> tuple[str, Path]:
    path = TASKS_DIR / f"{name}.md"
    if not path.exists():
        known = ", ".join(available_tasks()) or "(none)"
        raise FileNotFoundError(
            f"Unknown task '{name}'. Available tasks: {known}"
        )
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Task file is empty: {path}")
    return text, path
