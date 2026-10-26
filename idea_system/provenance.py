"""Run manifests: evidence trail for every run (workspace convention)."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunManifest:
    """Incrementally persisted manifest.json inside a run directory."""

    def __init__(self, run_dir: Path, stage: str, task: str, config: dict[str, Any]):
        self.run_dir = run_dir
        self.path = run_dir / "manifest.json"
        self.data: dict[str, Any] = {
            "system_version": __version__,
            "stage": stage,
            "task": task,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "config": config,
            "file_hashes": {},
            "stages": [],
            "llm_calls": [],
            "search_log": [],
            "usage_totals": {},
            "notes": [],
        }
        self._seen_records: set[int] = set()
        self._seen_search_entries: set[int] = set()

    def begin_stage(self, stage: str, dry_run: bool | None = None) -> None:
        """Append a stage marker (multi-stage runs share one manifest)."""
        self.data["stage"] = stage
        self.data["stages"].append({"stage": stage, "dry_run": dry_run})
        if dry_run is not None and "dry_run" not in self.data:
            self.data["dry_run"] = dry_run

    def note(self, text: str) -> None:
        self.data["notes"].append(
            f"{datetime.now(timezone.utc).isoformat()} {text}"
        )

    def add_file_hash(self, path: Path, label: str | None = None) -> None:
        key = label or str(path.name)
        self.data["file_hashes"][key] = sha256_file(path)

    def add_llm_history(self, history) -> None:
        """Append record objects not yet ingested by this manifest. Within
        one process a growing client history is re-reported safely; each
        pipeline stage uses a fresh client, and a restored manifest starts
        with an empty seen-set so the new stage's calls all append."""
        records = [
            record for record in history.records
            if id(record) not in self._seen_records
        ]
        for record in records:
            self._seen_records.add(id(record))
        for record in records:
            self.data["llm_calls"].append({
                "stage": record.stage, "model": record.model, "ok": record.ok,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "latency_seconds": record.latency_seconds,
                "error": record.error, "repair_attempts": record.repair_attempts,
            })
        # totals are summed over newly appended records only, so repeated
        # ingestion of one growing history never double counts
        previous = self.data.get("usage_totals", {})
        self.data["usage_totals"] = {
            "calls": previous.get("calls", 0) + len(records),
            "failed_calls": previous.get("failed_calls", 0)
            + sum(1 for r in records if not r.ok),
            "prompt_tokens": previous.get("prompt_tokens", 0)
            + sum(r.prompt_tokens or 0 for r in records),
            "completion_tokens": previous.get("completion_tokens", 0)
            + sum(r.completion_tokens or 0 for r in records),
            "total_tokens": None,  # filled below
        }
        totals = self.data["usage_totals"]
        totals["total_tokens"] = (
            totals["prompt_tokens"] + totals["completion_tokens"]
        )

    def add_search_log(self, search_log) -> None:
        new_queries = [
            entry for entry in search_log.queries
            if id(entry) not in self._seen_search_entries
        ]
        for entry in new_queries:
            self._seen_search_entries.add(id(entry))
        self.data["search_log"].extend(new_queries)

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)


def new_run_dir(runs_root: Path, task: str, stage: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return runs_root / f"{stamp}_{stage}_{task}"
