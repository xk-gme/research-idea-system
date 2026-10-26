"""Stage B: open co-scientist tournament (replacement for AI co-scientist).

Bounded re-implementation of the multi-agent loop described in
"Towards an AI co-scientist" (arXiv:2502.18864) as adapted by the open
jataware/open-coscientist project: generate distinct hypotheses, review
each against fixed criteria, rank, run an Elo tournament of pairwise
LLM judgements, meta-review the reviews, then evolve the top hypotheses
(mutations + recombinations) while pruning the bottom. Elo updates are
computed deterministically in code; the LLM only picks pairwise winners.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .provenance import RunManifest

GENERATE_SYSTEM = (
    "You are a creative research scientist generating candidate method "
    "hypotheses for a scientific modeling task. Hypotheses must be clearly "
    "distinct from each other in mechanism. Reply with ONLY a JSON array "
    'of objects with keys "title" and "hypothesis".'
)

REFLECT_SYSTEM = (
    "You are a skeptical peer comparing proposed hypotheses against the "
    "published literature findings provided. Identify overlaps and state "
    "what is genuinely new. Reply with ONLY a JSON array of objects with "
    'keys "id", "novelty_assessment" (string), "overlaps_with_literature" '
    '(string).'
)

REVIEW_SYSTEM = (
    "You are a rigorous grant reviewer. Score the hypothesis on each "
    "criterion from 1 to 10 and give a one-line verdict. Reply with ONLY "
    'a JSON object with one key per criterion (integer score) plus '
    '"verdict" (one of accept/refine/reject) and "comment" (string).'
)

RANK_SYSTEM = (
    "You are a committee chair ranking research hypotheses by promise "
    "for the research goal, using the reviews. Reply with ONLY a JSON "
    'array of hypothesis ids, best first.'
)

JUDGE_SYSTEM = (
    "You are a tournament judge. Given the research goal and two "
    "hypotheses with their reviews, decide which hypothesis is more "
    "promising to pursue. Reply with ONLY 'WINNER: A' or 'WINNER: B'."
)

META_SYSTEM = (
    "You are a meta-reviewer synthesizing many hypothesis reviews into "
    "common strengths, common weaknesses, and strategic directions for "
    "the next evolution round. Reply with ONLY a JSON object with keys "
    '"common_strengths", "common_weaknesses", "strategic_directions" '
    "(arrays of strings)."
)

EVOLVE_SYSTEM = (
    "You are an evolutionary research strategist. Improve the given top "
    "hypotheses using the meta-review: create mutations that fix "
    "weaknesses and recombinations that merge complementary mechanisms. "
    "Keep each output clearly distinct and preserve the best ideas. "
    "Reply with ONLY a JSON array of objects with keys \"title\", "
    "\"hypothesis\", \"origin\" (one of mutation/recombination), "
    "\"parent_ids\" (array)."
)

DEFAULT_CRITERIA = [
    "significance", "novelty", "feasibility",
    "effectiveness", "clarity", "task_fit",
]


def elo_update(rating_a: float, rating_b: float, score_a: float,
               k: float = 32.0) -> tuple[float, float]:
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - expected_a))
    return new_a, new_b


class _Guard:
    def __init__(self, client: Any, max_calls: int):
        self.client = client
        self.max_calls = max_calls

    def check(self) -> None:
        calls = len(self.client.history.records)
        if calls >= self.max_calls:
            raise RuntimeError(
                f"LLM call budget exhausted ({calls}/{self.max_calls}). "
                "Raise runtime.max_llm_calls or narrow the run."
            )


def _hypotheses_block(hypotheses: list[dict[str, Any]]) -> str:
    lines = []
    for hyp in hypotheses:
        reviews = hyp.get("reviews", [])
        scores = reviews[-1].get("scores", {}) if reviews else {}
        lines.append(
            f"[{hyp['id']}] {hyp.get('title', '')}\n"
            f"{hyp.get('hypothesis', hyp.get('text', ''))}\n"
            f"scores: {scores or 'n/a'} origin: {hyp.get('origin', 'base')}"
        )
    return "\n\n".join(lines)


def _judge_winner(content: str) -> float | None:
    text = content.upper()
    has_a = "WINNER: A" in text or "WINNER:A" in text
    has_b = "WINNER: B" in text or "WINNER:B" in text
    if has_a and not has_b:
        return 1.0
    if has_b and not has_a:
        return 0.0
    return None


def run_coscientist(
    client: Any,
    task_text: str,
    config: dict[str, Any],
    run_dir: Path,
    manifest: RunManifest,
    literature: dict[str, Any] | None = None,
    seed_ideas: list[dict[str, Any]] | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    cs_config = config["coscientist"]
    rounds = int(cs_config["rounds"])
    initial_count = int(cs_config["initial_hypotheses"])
    elo_base = float(cs_config["elo_base"])
    elo_k = float(cs_config["elo_k"])
    matches = int(cs_config["elo_matches_per_round"])
    top_k = int(cs_config["evolve_top_k"])
    judge_temperature = float(cs_config.get("judge_temperature", 0.2))
    criteria = DEFAULT_CRITERIA
    rng = rng or random.Random(int(config["runtime"]["seed"]))
    guard = _Guard(client, int(config["runtime"]["max_llm_calls"]))

    hypotheses: list[dict[str, Any]] = []
    next_number = 1

    def assign(idea: dict[str, Any], origin: str) -> dict[str, Any]:
        nonlocal next_number
        entry = {
            "id": f"H{next_number}",
            "title": idea.get("title", ""),
            "hypothesis": idea.get(
                "hypothesis", idea.get("idea", idea.get("text", ""))
            ),
            "origin": idea.get("origin", origin),
            "parent_ids": idea.get("parent_ids", []),
            "elo": elo_base,
            "reviews": [],
        }
        next_number += 1
        return entry

    literature_block = ""
    if literature:
        import json
        literature_block = (
            "\n\nPublished literature findings:\n"
            + json.dumps(literature, ensure_ascii=False, indent=1)[:8000]
        )

    if seed_ideas:
        for idea in seed_ideas:
            entry = assign(idea, idea.get("origin", "deep_research"))
            hypotheses.append(entry)

    round_reports: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        if round_index == 1:
            guard.check()
            generated = client.chat_json(
                GENERATE_SYSTEM,
                task_text
                + (f"\n\nGenerate {initial_count} distinct hypotheses."
                   if not seed_ideas else
                   "\n\nGenerate additional distinct hypotheses that "
                   "differ mechanistically from the provided ones."
                   f" Produce {max(2, initial_count - len(seed_ideas))}.")
                + "\n\nExisting hypotheses to differ from:\n"
                + (_hypotheses_block(hypotheses) or "(none)")
                + literature_block,
                stage=f"generate_R{round_index}",
            )
            for idea in generated:
                hypotheses.append(assign(idea, "generated"))
        else:
            guard.check()
            meta = round_reports[-1]["meta_review"]
            top = sorted(
                hypotheses, key=lambda h: h["elo"], reverse=True
            )[:top_k]
            guard.check()
            evolved = client.chat_json(
                EVOLVE_SYSTEM,
                task_text
                + "\n\nTop hypotheses to evolve:\n"
                + _hypotheses_block(top)
                + "\n\nMeta-review:\n"
                + _meta_block(meta)
                + f"\n\nProduce {top_k} evolved hypotheses.",
                stage=f"evolve_R{round_index}",
            )
            ranked_ids = [h["id"] for h in sorted(
                hypotheses, key=lambda h: h["elo"])]
            for idea in evolved:
                hypotheses.append(assign(idea, idea.get(
                    "origin", "mutation")))
            # prune the bottom to keep the pool bounded
            keep = set(ranked_ids[: max(initial_count, len(seed_ideas or []))])
            keep.update(h["id"] for h in top)
            hypotheses = [h for h in hypotheses if h["id"] in keep]

        manifest.add_llm_history(client.history)
        manifest.save()

        # Reflect against literature when available.
        reflections = None
        if literature_block:
            guard.check()
            reflections = client.chat_json(
                REFLECT_SYSTEM,
                task_text
                + "\n\nHypotheses:\n" + _hypotheses_block(hypotheses)
                + literature_block,
                stage=f"reflect_R{round_index}",
            )
            manifest.add_llm_history(client.history)
            manifest.save()

        # Review each hypothesis.
        for hyp in hypotheses:
            guard.check()
            review = client.chat_json(
                REVIEW_SYSTEM,
                task_text
                + "\n\nHypothesis:\n"
                + f"[{hyp['id']}] {hyp['title']}\n{hyp['hypothesis']}\n\n"
                + "Criteria: " + ", ".join(criteria) + ".",
                stage=f"review_{hyp['id']}_R{round_index}",
            )
            scores = {
                key: value for key, value in review.items()
                if key in criteria and isinstance(value, (int, float))
            }
            hyp["reviews"].append({
                "round": round_index,
                "scores": scores,
                "verdict": review.get("verdict", ""),
                "comment": review.get("comment", ""),
            })
        manifest.add_llm_history(client.history)
        manifest.save()

        # Holistic ranking (recorded; Elo below drives selection).
        guard.check()
        ranking = client.chat_json(
            RANK_SYSTEM,
            task_text
            + "\n\nHypotheses with latest reviews:\n"
            + _hypotheses_block(hypotheses)
            + "\n\nRank all ids best first.",
            stage=f"rank_R{round_index}",
        )
        ranking = [str(r) for r in ranking]
        manifest.add_llm_history(client.history)
        manifest.save()

        # Elo tournament.
        for match_index in range(matches):
            if len(hypotheses) < 2:
                break
            a, b = rng.sample(range(len(hypotheses)), 2)
            hyp_a, hyp_b = hypotheses[a], hypotheses[b]
            guard.check()
            content = client.chat(
                JUDGE_SYSTEM,
                task_text
                + "\n\nHypothesis A:\n"
                + f"[{hyp_a['id']}] {hyp_a['title']}\n{hyp_a['hypothesis']}\n"
                + f"reviews: {hyp_a['reviews'][-1] if hyp_a['reviews'] else 'n/a'}"
                + "\n\nHypothesis B:\n"
                + f"[{hyp_b['id']}] {hyp_b['title']}\n{hyp_b['hypothesis']}\n"
                + f"reviews: {hyp_b['reviews'][-1] if hyp_b['reviews'] else 'n/a'}"
                + "\n\nWhich is more promising?",
                stage=f"judge_R{round_index}_M{match_index + 1}",
                temperature=judge_temperature,
            )
            score_a = _judge_winner(content)
            if score_a is None:
                score_a = 0.5  # unparseable verdict counts as a tie
            hyp_a["elo"], hyp_b["elo"] = elo_update(
                hyp_a["elo"], hyp_b["elo"], score_a, k=elo_k
            )
        manifest.add_llm_history(client.history)
        manifest.save()

        # Meta-review.
        guard.check()
        meta = client.chat_json(
            META_SYSTEM,
            task_text
            + "\n\nAll latest reviews:\n"
            + "\n".join(
                f"[{h['id']}] {h['reviews'][-1] if h['reviews'] else 'n/a'}"
                for h in hypotheses
            )
            + ("\n\nReflections vs literature:\n" + _refl_block(reflections)
               if reflections else ""),
            stage=f"meta_R{round_index}",
        )
        manifest.add_llm_history(client.history)
        manifest.save()

        round_reports.append({
            "round": round_index,
            "ranking": ranking,
            "reflections": reflections,
            "meta_review": meta,
        })

    hypotheses.sort(key=lambda h: h["elo"], reverse=True)
    output = {
        "hypotheses": hypotheses,
        "round_reports": round_reports,
    }
    manifest.note(f"co-scientist done: {len(hypotheses)} hypotheses, "
                  f"{rounds} rounds")
    manifest.add_llm_history(client.history)
    manifest.save()

    import json
    with open(run_dir / "coscientist_output.json", "w",
              encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    with open(run_dir / "coscientist_report.md", "w", encoding="utf-8") as handle:
        handle.write("# Co-scientist tournament\n\n")
        handle.write("| id | title | elo | verdict | origin |\n")
        handle.write("|---|---|---|---|---|\n")
        for hyp in hypotheses:
            verdict = (hyp["reviews"][-1]["verdict"]
                       if hyp["reviews"] else "")
            handle.write(
                f"| {hyp['id']} | {hyp['title'][:60]} | "
                f"{hyp['elo']:.0f} | {verdict} | {hyp['origin']} |\n"
            )
    return output


def _meta_block(meta: dict[str, Any]) -> str:
    import json
    return json.dumps(meta, ensure_ascii=False, indent=1)


def _refl_block(reflections: list[dict[str, Any]] | None) -> str:
    if not reflections:
        return ""
    import json
    return json.dumps(reflections, ensure_ascii=False, indent=1)
