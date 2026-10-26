"""Stage A: open deep research (replacement for Gemini Deep Research).

Recursive literature-research loop in the style of the open
dzhng/deep-research architecture, closing with the paper's S15 request:
given the task background and everything learned, produce 10 highly
novel ideas with detailed implementation notes. All sources come from the
plugged search backends (arXiv by default), and every idea carries
citations back to the source list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import RunManifest
from .websearch import SearchLog, SearchResult

PLANNER_SYSTEM = (
    "You are a research planning assistant. You design literature search "
    "queries that will surface methods relevant to a scientific modeling "
    "task. Reply with ONLY a JSON array of query strings."
)

LEARNINGS_SYSTEM = (
    "You are a research analyst. You extract concrete methodological "
    "findings from retrieved literature records. Each finding must cite "
    "the record ids it rests on. Reply with ONLY a JSON array of objects "
    'with keys "finding" (string) and "citations" (array of record id '
    'strings like "S1").'
)

FOLLOWUP_SYSTEM = (
    "You are a research planner. Given what has already been learned, "
    "propose follow-up literature search queries that cover still-missing "
    "method families. Reply with ONLY a JSON array of query strings."
)

IDEATOR_SYSTEM = (
    "You are an elite research scientist aiming to design breakthrough "
    "methods. You ground every idea in the cited literature findings but "
    "push well beyond them. Reply with ONLY a JSON array of objects, each "
    'with keys "title", "idea", "implementation_notes", "references" '
    '(array of record id strings).'
)

S15_CLOSING = (
    "This task aims to develop a SUPERHUMAN METHOD for solving this "
    "problem.\n"
    "Please give me 10 highly novel and creative ideas with detailed "
    "implementation notes for the set of methods I should explore for "
    "solving this task. I aim to create the best method for solving this "
    "problem, preferably creating the best ever method."
)


def _records_block(records: dict[str, SearchResult]) -> str:
    lines = []
    for rid, record in records.items():
        lines.append(
            f"[{rid}] {record.title} ({record.source}, {record.published})\n"
            f"    {record.url}\n"
            f"    {record.abstract[:1200] if record.abstract else '(no abstract)'}"
        )
    return "\n".join(lines)


def run_deep_research(
    client: Any,
    backends: list[Any],
    task_text: str,
    config: dict[str, Any],
    run_dir: Path,
    manifest: RunManifest,
    corpus: Any | None = None,
) -> dict[str, Any]:
    research_config = config["research"]
    n_queries = int(research_config["initial_queries"])
    depth_levels = int(research_config["depth_levels"])
    ideas_count = int(research_config["ideas_count"])

    search_log = SearchLog()
    records: dict[str, SearchResult] = {}
    learnings: list[dict[str, Any]] = []
    queries_by_level: list[list[str]] = []
    search_counter = 0

    # Optional browser-harvested corpus (form 1): C-numbered records join
    # the literature pool before any search happens.
    if corpus is not None:
        records.update(corpus.records)
        for entry in corpus.provenance:
            if entry.get("status") == "loaded":
                source_file = corpus.directory / entry.get("file", "")
                if source_file.exists():
                    manifest.add_file_hash(
                        source_file, f"corpus:{entry.get('file', '')}")
        manifest.data["corpus"] = corpus.provenance
        manifest.note(
            f"harvest corpus loaded: {len(corpus.records)} records from "
            f"{corpus.directory}"
        )
        manifest.save()

    # Level 0: plan initial queries from the task background alone.
    queries = client.chat_json(
        PLANNER_SYSTEM,
        task_text
        + f"\n\nGenerate {n_queries} diverse literature search queries "
        "(arXiv-style English queries, each targeting a different method "
        "family or technique relevant to the task).",
        stage="plan_queries",
    )
    queries = [str(q) for q in queries][:n_queries]
    queries_by_level.append(queries)
    manifest.save()

    for level in range(depth_levels):
        for query in queries:
            for backend in backends:
                try:
                    results = backend.search(query)
                    search_log.add(backend.name, query, len(results))
                except Exception as error:  # best-effort per backend
                    search_log.add(backend.name, query, 0, str(error)[:120])
                    continue
                for result in results:
                    if result.url and result.url not in {
                        r.url for r in records.values()
                    }:
                        search_counter += 1
                        records[f"S{search_counter}"] = result
        manifest.add_search_log(search_log)
        manifest.save()

        learnings = client.chat_json(
            LEARNINGS_SYSTEM,
            task_text
            + "\n\nRetrieved literature records:\n"
            + _records_block(records)
            + "\n\nExtract 8-14 methodological findings useful for "
            "designing a superior method for the task.",
            stage=f"learnings_L{level}",
        )
        manifest.add_llm_history(client.history)
        manifest.save()

        if level < depth_levels - 1:
            queries = client.chat_json(
                FOLLOWUP_SYSTEM,
                task_text
                + "\n\nLearnings so far:\n"
                + json.dumps(learnings, ensure_ascii=False, indent=1)
                + f"\n\nPropose {n_queries} follow-up queries.",
                stage=f"followup_L{level}",
            )[:n_queries]
            queries = [str(q) for q in queries]
            queries_by_level.append(queries)
            manifest.save()

    # Final S15 call: task background + accumulated findings -> ideas.
    ideas = client.chat_json(
        IDEATOR_SYSTEM,
        task_text
        + "\n\nLiterature findings from my research so far:\n"
        + json.dumps(learnings, ensure_ascii=False, indent=1)
        + "\n\nRetrieved literature records (for references):\n"
        + _records_block(records)
        + "\n\n"
        + S15_CLOSING.replace("10 ", f"{ideas_count} ", 1),
        stage="s15_ideas",
    )
    ideas = ideas[:ideas_count]
    for index, idea in enumerate(ideas, start=1):
        idea.setdefault("id", f"R{index}")
        idea["origin"] = "deep_research"
    manifest.add_llm_history(client.history)
    manifest.add_search_log(search_log)
    manifest.note(f"deep research done: {len(ideas)} ideas, "
                  f"{len(records)} literature records")
    manifest.save()

    output = {
        "ideas": ideas,
        "learnings": learnings,
        "records": {
            rid: {
                "title": r.title, "url": r.url, "source": r.source,
                "published": r.published,
                "abstract": r.abstract,
            }
            for rid, r in records.items()
        },
        "queries_by_level": queries_by_level,
    }
    with open(run_dir / "ideas.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    with open(run_dir / "research_notes.md", "w", encoding="utf-8") as handle:
        handle.write("# Deep research notes\n\n## Queries\n\n")
        for level, queries in enumerate(queries_by_level):
            handle.write(f"- L{level}: " + " | ".join(queries) + "\n")
        handle.write("\n## Literature records\n\n")
        for rid, record in output["records"].items():
            handle.write(
                f"- **[{rid}]** {record['title']} — {record['url']}\n"
            )
        handle.write("\n## Learnings\n\n")
        for item in learnings:
            finding = item.get("finding", "")
            citations = ",".join(item.get("citations", []))
            handle.write(f"- {finding} [{citations}]\n")
    return output
