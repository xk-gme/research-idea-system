"""Harvested-corpus loading (form 1: browser harvest pre-pass).

A harvest corpus is produced by an agent session with browser/search
tools following prompts/收割协议_浏览器研究.md: each source becomes one
Markdown file with YAML front-matter (url/title/accessed_at/agent/tool)
plus excerpt text, and harvest_manifest.json lists every file. The
loader turns these into the same record shape as search results (ids
"C1, C2, ...") so the research loop can cite them side by side with
arXiv records. URL-level dedup keeps one entry per source.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .websearch import SearchResult

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", flags=re.DOTALL)

REQUIRED_FIELDS = ("url", "title", "accessed_at")


class CorpusError(ValueError):
    """Invalid harvest corpus layout or front-matter."""


@dataclass
class Corpus:
    records: dict[str, SearchResult]
    provenance: list[dict[str, Any]]
    directory: Path


def load_corpus(corpus_dir: Path | str) -> Corpus:
    """Load a harvest directory into C-numbered records.

    corpus_dir must contain harvest_manifest.json and at least one
    source .md file. Every .md must appear in the manifest; front-matter
    must carry url/title/accessed_at. Duplicate urls are dropped (first
    wins, in manifest order).
    """
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / "harvest_manifest.json"
    if not manifest_path.exists():
        raise CorpusError(
            f"{corpus_dir} is not a harvest corpus: harvest_manifest.json "
            "missing. Run harvest-scaffold first, then harvest sources per "
            "prompts/收割协议_浏览器研究.md."
        )
    entries = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list) or not entries:
        raise CorpusError("harvest_manifest.json must be a non-empty list.")

    records: dict[str, SearchResult] = {}
    provenance: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        file_name = entry.get("file", "")
        md_path = corpus_dir / file_name
        if not md_path.exists():
            raise CorpusError(f"Manifest entry '{file_name}' has no file.")
        text = md_path.read_text(encoding="utf-8-sig")
        match = FRONT_MATTER.match(text)
        if not match:
            raise CorpusError(
                f"{md_path.name}: missing YAML front-matter block."
            )
        meta = yaml.safe_load(match.group(1)) or {}
        missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
        if missing:
            raise CorpusError(
                f"{md_path.name}: front-matter missing {missing}."
            )
        body = text[match.end():].strip()
        url = str(meta["url"]).strip()
        if url in seen_urls:
            provenance.append({**entry, "status": "duplicate_url_skipped"})
            continue
        seen_urls.add(url)
        records[f"C{len(records) + 1}"] = SearchResult(
            title=str(meta["title"]).strip(),
            url=url,
            abstract=re.sub(r"\s+", " ", body)[:4000] or "(no excerpt)",
            source="harvest",
            published=str(meta.get("accessed_at", ""))[:10],
        )
        provenance.append({
            "file": file_name,
            "url": url,
            "title": str(meta.get("title", "")),
            "accessed_at": str(meta.get("accessed_at", "")),
            "agent": str(meta.get("agent", entry.get("agent", ""))),
            "tool": str(meta.get("tool", entry.get("tool", ""))),
            "status": "loaded",
        })
    if not records:
        raise CorpusError("Harvest corpus has no loadable records.")
    return Corpus(records=records, provenance=provenance, directory=corpus_dir)


def scaffold_harvest(harvest_root: Path, task_name: str, task_text: str,
                     protocol_text: str) -> Path:
    """Create runs/<stamp>_harvest_<task>/harvest/ with a manifest template
    and a harvesting guide customized with the task background."""
    from .provenance import new_run_dir
    run_dir = new_run_dir(harvest_root, task_name, "harvest")
    harvest_dir = run_dir / "harvest"
    harvest_dir.mkdir(parents=True, exist_ok=True)
    manifest_template = [
        {
            "file": "source_01.md",
            "url": "",
            "agent": "",
            "tool": "",
        }
    ]
    with open(harvest_dir / "harvest_manifest.json", "w",
              encoding="utf-8") as handle:
        json.dump(manifest_template, handle, ensure_ascii=False, indent=2)
    with open(harvest_dir / "source_01.md", "w", encoding="utf-8") as handle:
        handle.write(
            "---\n"
            "url: \n"
            "title: \n"
            "accessed_at: \n"
            "agent: \n"
            "tool: \n"
            "---\n"
            "(Replace with 5-20 lines of key excerpts from the source.)\n"
        )
    guide = (
        f"# 收割指引：{task_name}\n\n"
        "按《收割协议》执行，完成后删除本指引文件不影响加载。"
        "协议全文如下：\n\n---\n\n" + protocol_text
        + "\n\n---\n\n## 本任务的背景（检索与摘录以此为纲）\n\n"
        + task_text + "\n"
    )
    with open(run_dir / "收割指引.md", "w", encoding="utf-8") as handle:
        handle.write(guide)
    return harvest_dir
