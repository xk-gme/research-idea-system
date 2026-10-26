"""Stage C: S16 formatting + human filtering helpers.

The paper's Supplementary Table S16 prompt formats each kept idea into the
<description>/<steps>/<notes> structure used by baseline method
descriptions; these cards are then injected as {method} into the ERA tree
search prompt (Supplementary Table S10) instead of the Gemini-formatted
cards.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

S16_SYSTEM = (
    "You format research ideas into structured method cards. Output ONLY "
    "the card, with no extra prose."
)

S16_PROMPT = """Structure the given idea into the following format:
<description>
Your description about the method goes here.
</description>
<steps>
Your list of steps to implement the method goes here.
</steps>
<notes>
Strengths and weaknesses of the idea goes here.
</notes>

Idea:
"""

TAG_PATTERN = re.compile(
    r"<description>(.*?)</description>\s*"
    r"<steps>(.*?)</steps>\s*"
    r"<notes>(.*?)</notes>",
    flags=re.DOTALL,
)


def load_pools(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load idea pools present in a run directory."""
    pools: dict[str, list[dict[str, Any]]] = {}
    ideas_path = run_dir / "ideas.json"
    if ideas_path.exists():
        data = json.loads(ideas_path.read_text(encoding="utf-8-sig"))
        pools["research"] = data.get("ideas", [])
    cos_path = run_dir / "coscientist_output.json"
    if cos_path.exists():
        data = json.loads(cos_path.read_text(encoding="utf-8-sig"))
        pools["coscientist"] = data.get("hypotheses", [])
    return pools


def write_selection_template(run_dir: Path) -> Path:
    pools = load_pools(run_dir)
    template = {
        "_instructions": (
            "Keep only the ideas to carry forward. Set decided_by to your "
            "name/handle, then run: python -m idea_system.pipeline format "
            f"--run {run_dir}"
        ),
        "keep": [
            {"source": source, "id": idea.get("id", "")}
            for source, ideas in pools.items()
            for idea in ideas
        ],
        "decided_by": "human",
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    path = run_dir / "selection.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(template, handle, ensure_ascii=False, indent=2)
    return path


def read_selection(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "selection.json"
    if not path.exists():
        raise FileNotFoundError(
            "selection.json not found. Run the filter command first, edit "
            "the keep list, then format."
        )
    selection = json.loads(path.read_text(encoding="utf-8-sig"))
    if not selection.get("keep"):
        raise ValueError("selection.json keeps nothing.")
    return selection


def _normalize_card(match: re.Match) -> str:
    description = match.group(1).strip()
    steps = match.group(2).strip()
    notes = match.group(3).strip()
    return (
        f"<description>\n{description}\n</description>\n"
        f"<steps>\n{steps}\n</steps>\n"
        f"<notes>\n{notes}\n</notes>"
    )


def format_selected(
    client: Any,
    run_dir: Path,
    manifest: Any,
    max_repair_attempts: int = 1,
) -> list[dict[str, Any]]:
    selection = read_selection(run_dir)
    pools = load_pools(run_dir)
    by_key = {
        (source, str(idea.get("id", ""))): idea
        for source, ideas in pools.items()
        for idea in ideas
    }
    cards: list[dict[str, Any]] = []
    cards_dir = run_dir / "s16"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for entry in selection["keep"]:
        source = entry.get("source", "")
        idea_id = str(entry.get("id", ""))
        idea = by_key.get((source, idea_id))
        if idea is None:
            raise KeyError(
                f"Selected idea not found: source={source} id={idea_id}"
            )
        idea_text = (
            f"Title: {idea.get('title', '')}\n"
            f"Hypothesis/idea: {idea.get('hypothesis', idea.get('idea', ''))}\n"
            f"Implementation notes: {idea.get('implementation_notes', '')}"
        )
        content = client.chat(
            S16_SYSTEM, S16_PROMPT + idea_text,
            stage=f"s16_{source}_{idea_id}",
        )
        match = TAG_PATTERN.search(content)
        repairs = 0
        while match is None and repairs < max_repair_attempts:
            repairs += 1
            content = client.chat(
                S16_SYSTEM,
                S16_PROMPT + idea_text
                + "\n\nYour previous reply did not contain the required "
                "<description>/<steps>/<notes> tags. Output ONLY the card.",
                stage=f"s16_{source}_{idea_id}#repair{repairs}",
            )
            match = TAG_PATTERN.search(content)
        card = {
            "source": source,
            "id": idea_id,
            "title": idea.get("title", ""),
            "card_valid": match is not None,
            "card": content if match is None else _normalize_card(match),
        }
        cards.append(card)
        manifest.add_llm_history(client.history)
        manifest.save()

    with open(cards_dir / "methods.json", "w", encoding="utf-8") as handle:
        json.dump(cards, handle, ensure_ascii=False, indent=2)
    for index, card in enumerate(cards, start=1):
        safe_title = re.sub(
            r"[^0-9A-Za-z_-]+", "_", card["title"]
        )[:40].strip("_") or "idea"
        path = cards_dir / f"方法卡_{index:02d}_{safe_title}.md"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                f"# {card['title']}\n\n"
                f"- source: {card['source']} / {card['id']}\n"
                f"- card_valid: {card['card_valid']}\n\n"
                "```\n" + card["card"] + "\n```\n"
            )
    manifest.note(f"S16 formatting done: {len(cards)} cards")
    manifest.save()
    return cards
