"""Command-line entry point for the research idea system.

Pipeline (mirrors the ERA paper's Methods "Adding research ideas to the
code mutation system"):

  1. research    - open deep research -> S15-style idea pool (ideas.json)
  2. coscientist - multi-agent Elo tournament -> hypothesis pool
                   (can continue the same run dir and reuse the research
                   learnings as literature context)
  3. filter      - write selection.json for the human to prune (manual
                   filtering step of the paper)
  4. format      - S16 formatting of kept ideas into method cards ready
                   for {method} injection into the ERA tree-search prompt

Every stage writes an incremental manifest.json with prompt-file hashes,
per-call usage and the search log. --dry-run replaces the LLM and the
search backends with deterministic fakes (zero API cost, no network).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from .config import SYSTEM_ROOT, load_config
from .provenance import RunManifest, new_run_dir
from .tasks import available_tasks, load_task


def _print_pools(run_dir: Path) -> None:
    from .format_s16 import load_pools
    pools = load_pools(run_dir)
    if not pools:
        print("No idea pools found in run dir (run research/coscientist first).")
        return
    for source, ideas in pools.items():
        print(f"\n[{source}]")
        for idea in ideas:
            title = idea.get("title", "")[:80]
            print(f"  {idea.get('id', '?'):>4}  {title}")


def _fake_responder(system: str, user: str) -> str:
    """Deterministic dry-run LLM: emits plausible JSON per stage."""
    if "research planning assistant" in system:
        return json.dumps([
            "probabilistic forecasting calibration methods",
            "ensemble time series model combination",
        ])
    if "research analyst" in system:
        return json.dumps([
            {"finding": "Quantile regression ensembles improve WIS.",
             "citations": ["S1"]},
            {"finding": "Growth-rate scaling stabilizes short horizons.",
             "citations": ["S2"]},
        ])
    if "research planner" in system:  # follow-up planner
        return json.dumps([
            "hierarchical reconciliation of quantile forecasts",
            "robust bayesian structural time series",
        ])
    if "elite research scientist" in system:
        return json.dumps([
            {
                "title": f"Dry-run idea {i}",
                "idea": "Deterministic placeholder idea for offline runs.",
                "implementation_notes": "Placeholder notes.",
                "references": ["S1"],
            }
            for i in range(1, 11)
        ])
    if "creative research scientist" in system:
        return json.dumps([
            {"title": "Dry-run hypothesis", "hypothesis": "Placeholder."},
        ])
    if "skeptical peer" in system:
        return json.dumps([])
    if "rigorous grant reviewer" in system:
        return json.dumps({
            "significance": 5, "novelty": 5, "feasibility": 5,
            "effectiveness": 5, "clarity": 5, "task_fit": 5,
            "verdict": "refine", "comment": "dry run",
        })
    if "committee chair" in system:
        return json.dumps([])
    if "tournament judge" in system:
        return "WINNER: A"
    if "meta-reviewer" in system:
        return json.dumps({
            "common_strengths": ["dry run"],
            "common_weaknesses": ["dry run"],
            "strategic_directions": ["dry run"],
        })
    if "evolutionary research strategist" in system:
        return json.dumps([
            {"title": "Dry-run evolution", "hypothesis": "Placeholder.",
             "origin": "mutation", "parent_ids": []},
        ])
    if "format research ideas" in system:
        return (
            "<description>\nDry-run description.\n</description>\n"
            "<steps>\n1. placeholder\n</steps>\n"
            "<notes>\nDry-run notes.\n</notes>"
        )
    return "dry-run"


def _load_or_create_manifest(run_dir: Path, stage: str, task: str,
                             config: dict, dry_run: bool) -> RunManifest:
    """Restore an existing manifest when continuing a run dir (provenance
    is append-only across stages); create a fresh one otherwise."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = RunManifest.__new__(RunManifest)
        manifest.run_dir = run_dir
        manifest.path = manifest_path
        manifest.data = json.loads(
            manifest_path.read_text(encoding="utf-8-sig"))
        manifest._seen_records = set()
        manifest._seen_search_entries = set()
    else:
        manifest = RunManifest(run_dir, stage, task, config)
        # Snapshot the controller code that will produce this run's outputs.
        for source in sorted((SYSTEM_ROOT / "idea_system").glob("*.py")):
            manifest.add_file_hash(source, f"src:{source.name}")
    manifest.begin_stage(stage, dry_run)
    return manifest


def _make_client(config: dict, dry_run: bool):
    if dry_run:
        from .deepseek import FakeLLM
        return FakeLLM(_fake_responder)
    from .deepseek import DeepSeekClient
    deepseek = config["deepseek"]
    return DeepSeekClient(
        model_name=deepseek["model"],
        secret_path=deepseek["secret_path"],
        timeout_seconds=deepseek["timeout_seconds"],
        transient_retries=deepseek["transient_retries"],
        max_tokens=deepseek["max_tokens"],
        temperature=deepseek["temperature"],
        thinking=deepseek["thinking"],
    )


def _make_backends(config: dict, dry_run: bool):
    from .websearch import SearchResult, build_backends
    if dry_run:
        from .websearch import FixtureBackend
        fixture = FixtureBackend({
            "forecast": [
                SearchResult(
                    title="Dry-run arXiv paper on quantile forecasting",
                    url="https://arxiv.org/abs/0000.0000",
                    abstract="Placeholder abstract about WIS calibration.",
                    source="arxiv", published="2025-01-01",
                ),
                SearchResult(
                    title="Dry-run arXiv paper on ensembles",
                    url="https://arxiv.org/abs/0000.0001",
                    abstract="Placeholder abstract about ensembling.",
                    source="arxiv", published="2025-01-02",
                ),
            ],
        })
        return [fixture]
    return build_backends(config["search"]["backends"], config["search"])


def _budget_report(config: dict) -> str:
    r = config["research"]
    c = config["coscientist"]
    research_calls = (
        1 + int(r["depth_levels"]) + max(0, int(r["depth_levels"]) - 1) + 1
    )
    initial = int(c["initial_hypotheses"])
    rounds = int(c["rounds"])
    matches = int(c["elo_matches_per_round"])
    per_round = initial + 1 + matches + 1 + 1  # reviews + rank + judges + meta
    coscientist_calls = rounds * per_round + (rounds - 1)  # + evolve calls
    return (
        f"research stage : ~{research_calls} LLM calls + "
        f"{int(r['initial_queries']) * int(r['depth_levels'])} searches\n"
        f"coscientist    : ~{coscientist_calls} LLM calls\n"
        f"format stage   : 1 LLM call per kept idea (+1 repair max)\n"
        f"hard guard     : runtime.max_llm_calls = "
        f"{config['runtime']['max_llm_calls']}"
    )


def cmd_plan(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    print(_budget_report(config))
    print("\nTasks:", ", ".join(available_tasks()))
    print("Dry run any stage with --dry-run (zero API cost).")
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    task_text, task_path = load_task(args.task)
    if args.run:
        run_dir = Path(args.run)
        stage = "research_cont"
    else:
        run_dir = new_run_dir(SYSTEM_ROOT / "runs", args.task, "research")
        stage = "research"
    corpus = _load_corpus_arg(args, run_dir)
    manifest = _load_or_create_manifest(
        run_dir, stage, args.task, config, args.dry_run)
    manifest.add_file_hash(task_path, f"task:{args.task}")
    manifest.save()
    client = _make_client(config, args.dry_run)
    backends = _make_backends(config, args.dry_run)
    from .deep_research import run_deep_research
    output = run_deep_research(
        client, backends, task_text, config, run_dir, manifest, corpus=corpus
    )
    print(f"research done: {len(output['ideas'])} ideas -> {run_dir}")
    print("next: python -m idea_system.pipeline filter --run", run_dir)
    return 0


def _load_corpus_arg(args: argparse.Namespace, run_dir: Path):
    """Resolve the harvest corpus: explicit --corpus wins, else auto-discover
    a harvest/ dir inside the continued run dir."""
    from .corpus import CorpusError, load_corpus
    candidates = []
    if getattr(args, "corpus", None):
        candidates.append(Path(args.corpus))
    elif getattr(args, "run", None):
        candidates.append(run_dir / "harvest")
    for candidate in candidates:
        if (candidate / "harvest_manifest.json").exists():
            try:
                return load_corpus(candidate)
            except CorpusError as error:
                print(f"corpus load failed, continuing without it: {error}")
                return None
    return None


def cmd_coscientist(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    task_text, task_path = load_task(args.task)
    literature = None
    seed_ideas = None
    if args.run:
        run_dir = Path(args.run)
        ideas_path = run_dir / "ideas.json"
        if not ideas_path.exists():
            print(f"no ideas.json in {run_dir}; starting standalone run")
            run_dir = new_run_dir(
                SYSTEM_ROOT / "runs", args.task, "coscientist")
        else:
            data = json.loads(ideas_path.read_text(encoding="utf-8-sig"))
            seed_ideas = data.get("ideas", [])
            literature = data.get("learnings")
    else:
        run_dir = new_run_dir(SYSTEM_ROOT / "runs", args.task, "coscientist")
    manifest = _load_or_create_manifest(
        run_dir, "coscientist", args.task, config, args.dry_run)
    manifest.add_file_hash(task_path, f"task:{args.task}")
    if seed_ideas:
        manifest.note(
            f"seeded with {len(seed_ideas)} deep research ideas + "
            "their literature learnings"
        )
    manifest.save()

    # Engine choice: ocs = jataware/open-coscientist framework (default
    # when its ocs-venv exists); self = built-in bounded tournament.
    from .ocs_bridge import ocs_available, run_ocs_coscientist
    engine = getattr(args, "engine", "auto")
    use_ocs = (
        engine in ("ocs", "auto")
        and not args.dry_run
        and ocs_available()
    )
    if engine == "ocs" and (args.dry_run or not ocs_available()):
        print("ocs engine requested but unavailable (dry-run or missing "
              "ocs-venv); falling back to self engine")
        use_ocs = False
    if engine == "self":
        use_ocs = False

    if use_ocs:
        output = run_ocs_coscientist(
            task_text, config, run_dir, manifest, seed_ideas=seed_ideas,
        )
    else:
        client = _make_client(config, args.dry_run)
        rng = random.Random(int(config["runtime"]["seed"]))
        from .coscientist import run_coscientist
        output = run_coscientist(
            client, task_text, config, run_dir, manifest,
            literature=literature, seed_ideas=seed_ideas, rng=rng,
        )
    engine_name = output.get("engine", "self")
    print(f"coscientist done (engine={engine_name}): "
          f"{len(output['hypotheses'])} hypotheses -> {run_dir}")
    print("next: python -m idea_system.pipeline filter --run", run_dir)
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    from .format_s16 import write_selection_template
    path = write_selection_template(run_dir)
    _print_pools(run_dir)
    print(f"\nselection template written: {path}")
    print("Edit its keep list (manual filtering), then run: "
          "python -m idea_system.pipeline format "
          f"--run {run_dir}")
    return 0


def cmd_format(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    config = load_config(Path(args.config) if args.config else None)
    manifest = _load_or_create_manifest(
        run_dir, "format", "?", config, args.dry_run)
    client = _make_client(config, args.dry_run)
    from .format_s16 import format_selected
    cards = format_selected(
        client, run_dir, manifest,
        max_repair_attempts=int(config["format"]["max_repair_attempts"]),
    )
    valid = sum(1 for card in cards if card["card_valid"])
    print(f"format done: {len(cards)} cards ({valid} valid tags) "
          f"-> {run_dir / 's16'}")
    print("Inject each card as {method} in the S10-style tree-search "
          "prompt (two slots), then run ERA scoring.")
    return 0


def cmd_harvest_scaffold(args: argparse.Namespace) -> int:
    task_text, _ = load_task(args.task)
    protocol_path = SYSTEM_ROOT / "prompts" / "收割协议_浏览器研究.md"
    protocol_text = protocol_path.read_text(encoding="utf-8-sig")
    from .corpus import scaffold_harvest
    harvest_dir = scaffold_harvest(
        SYSTEM_ROOT / "runs", args.task, task_text, protocol_text)
    guide = harvest_dir.parent / "收割指引.md"
    print(f"harvest workspace ready: {harvest_dir.parent}")
    print(f"guide for the harvesting agent: {guide}")
    print(
        "After the agent fills harvest/ per the protocol, run:\n"
        f"  python -m idea_system.pipeline research --task {args.task} "
        f"--corpus {harvest_dir}"
    )
    return 0


def cmd_takeover(args: argparse.Namespace) -> int:
    """Form 3: validate an agent-authored idea pool (ideas.json) and
    register it as a run. Lowest evidence tier: no bounded call log, the
    agent asserts its own sources; manifest records the tier explicitly."""
    config = load_config(Path(args.config) if args.config else None)
    task_text, task_path = load_task(args.task)
    source_file = Path(args.file)
    if not source_file.exists():
        print(f"file not found: {source_file}")
        return 2
    try:
        pool = json.loads(source_file.read_text(encoding="utf-8-sig"))
        ideas, records = _validate_takeover_pool(pool)
    except (ValueError, json.JSONDecodeError) as error:
        print(f"takeover validation failed: {error}")
        return 2
    if args.run:
        run_dir = Path(args.run)
    else:
        run_dir = new_run_dir(SYSTEM_ROOT / "runs", args.task, "takeover")
    manifest = _load_or_create_manifest(
        run_dir, "takeover", args.task, config, True)
    manifest.data["provenance_tier"] = "agent_takeover"
    manifest.add_file_hash(task_path, f"task:{args.task}")
    manifest.add_file_hash(source_file, "takeover:input")
    manifest.note(
        f"agent takeover registered: {len(ideas)} ideas, "
        f"{len(records)} distinct source urls; lowest evidence tier, not "
        "a formal reproduction path"
    )
    manifest.save()
    output = {
        "ideas": ideas,
        "learnings": [],
        "records": records,
        "queries_by_level": [],
    }
    with open(run_dir / "ideas.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(f"takeover registered: {len(ideas)} ideas -> {run_dir}")
    print("next: python -m idea_system.pipeline filter --run", run_dir)
    return 0


def _validate_takeover_pool(pool: Any) -> tuple[list[dict], dict]:
    if not isinstance(pool, dict) or not isinstance(
            pool.get("ideas"), list) or not pool["ideas"]:
        raise ValueError("input must be a JSON object with a non-empty 'ideas' list")
    ideas: list[dict] = []
    records: dict[str, dict] = {}
    titles: set[str] = set()
    url_to_sid: dict[str, str] = {}
    for index, raw in enumerate(pool["ideas"], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"idea #{index} is not an object")
        title = str(raw.get("title", "")).strip()
        body = str(raw.get("idea", raw.get("hypothesis", ""))).strip()
        if not title:
            raise ValueError(f"idea #{index}: missing title")
        if not body:
            raise ValueError(f"idea #{index}: missing idea/hypothesis text")
        if title.lower() in titles:
            raise ValueError(f"idea #{index}: duplicate title '{title}'")
        titles.add(title.lower())
        sources = raw.get("sources", [])
        if (not isinstance(sources, list) or not sources or not all(
                isinstance(s, str) and s.startswith(("http://", "https://"))
                for s in sources)):
            raise ValueError(
                f"idea #{index}: 'sources' must be a non-empty list of "
                "http(s) URLs (agent takeover requires real sources)"
            )
        references = []
        for url in sources:
            if url not in url_to_sid:
                sid = f"S{len(url_to_sid) + 1}"
                url_to_sid[url] = sid
                records[sid] = {
                    "title": url, "url": url, "source": "agent_claimed",
                    "published": "", "abstract": "",
                }
            references.append(url_to_sid[url])
        ideas.append({
            "id": f"R{index}",
            "title": title,
            "idea": body,
            "implementation_notes": str(raw.get("implementation_notes", "")),
            "references": references,
            "sources": sources,
            "origin": "agent_takeover",
        })
    return ideas, records


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="idea_system.pipeline",
        description="Open DeepSeek replacement for the ERA paper's "
                    "Gemini Deep Research / AI co-scientist idea chain.",
    )
    parser.add_argument("--config", help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="show call budget and available tasks")

    p_research = sub.add_parser(
        "research", help="stage A: deep research -> ideas.json")
    p_research.add_argument("--task", required=True)
    p_research.add_argument("--run", help="continue an existing run dir")
    p_research.add_argument(
        "--corpus",
        help="harvest corpus dir (harvest_manifest.json inside); defaults "
             "to auto-discovering harvest/ inside --run")
    p_research.add_argument("--dry-run", action="store_true")
    p_research.set_defaults(func=cmd_research)

    p_harvest = sub.add_parser(
        "harvest-scaffold",
        help="form 1: create a harvest workspace + guide for a "
             "browser-capable agent")
    p_harvest.add_argument("--task", required=True)
    p_harvest.set_defaults(func=cmd_harvest_scaffold)

    p_take = sub.add_parser(
        "takeover",
        help="form 3: validate & register an agent-authored ideas.json "
             "(lowest evidence tier)")
    p_take.add_argument("--task", required=True)
    p_take.add_argument("--file", required=True,
                        help="agent-produced ideas.json to validate")
    p_take.add_argument("--run", help="existing run dir to register into")
    p_take.set_defaults(func=cmd_takeover)

    p_cos = sub.add_parser(
        "coscientist", help="stage B: multi-agent Elo tournament")
    p_cos.add_argument("--task", required=True)
    p_cos.add_argument("--run", help="continue a research run dir (seeds "
                                     "hypotheses + literature)")
    p_cos.add_argument(
        "--engine", choices=["auto", "ocs", "self"], default="auto",
        help="ocs = jataware/open-coscientist framework (default when "
             "ocs-venv exists); self = built-in bounded tournament "
             "(dry-run always uses self)")
    p_cos.add_argument("--dry-run", action="store_true")
    p_cos.set_defaults(func=cmd_coscientist)

    p_filter = sub.add_parser(
        "filter", help="stage C: write selection.json for manual pruning")
    p_filter.add_argument("--run", required=True)
    p_filter.set_defaults(func=cmd_filter)

    p_format = sub.add_parser(
        "format", help="stage D: S16 method cards for kept ideas")
    p_format.add_argument("--run", required=True)
    p_format.add_argument("--dry-run", action="store_true")
    p_format.set_defaults(func=cmd_format)

    parser.set_defaults(func=lambda args: cmd_plan(args))
    args = parser.parse_args(argv)
    if args.command == "plan":
        return cmd_plan(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
