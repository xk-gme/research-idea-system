"""Offline tests: zero API calls, zero network (fixture search backend)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idea_system.config import load_config
from idea_system.coscientist import elo_update, run_coscientist
from idea_system.deep_research import run_deep_research
from idea_system.deepseek import FakeLLM, extract_json, read_deepseek_key
from idea_system.format_s16 import (
    format_selected, load_pools, write_selection_template,
)
from idea_system.provenance import RunManifest
from idea_system.tasks import available_tasks, load_task
from idea_system.websearch import FixtureBackend, SearchResult


def make_client() -> FakeLLM:
    def responder(system: str, user: str) -> str:
        if "research planning assistant" in system:
            return json.dumps(["quantile forecast calibration ensemble"])
        if "research analyst" in system:
            return json.dumps([
                {"finding": "Quantile ensembles improve WIS.",
                 "citations": ["S1"]},
            ])
        if "research planner" in system:
            return json.dumps(["hierarchical reconciliation"])
        if "elite research scientist" in system:
            return json.dumps([
                {"title": f"Idea {i}", "idea": f"method {i}",
                 "implementation_notes": "notes", "references": ["S1"]}
                for i in range(1, 11)
            ])
        if "creative research scientist" in system:
            return json.dumps([
                {"title": "Gen", "hypothesis": "hypo"},
            ])
        if "skeptical peer" in system:
            return json.dumps([])
        if "rigorous grant reviewer" in system:
            return json.dumps({
                "significance": 6, "novelty": 7, "feasibility": 5,
                "effectiveness": 6, "clarity": 8, "task_fit": 7,
                "verdict": "refine", "comment": "ok",
            })
        if "committee chair" in system:
            return json.dumps([])
        if "tournament judge" in system:
            return "WINNER: A"
        if "meta-reviewer" in system:
            return json.dumps({
                "common_strengths": ["s"], "common_weaknesses": ["w"],
                "strategic_directions": ["d"],
            })
        if "evolutionary research strategist" in system:
            return json.dumps([
                {"title": "Evo", "hypothesis": "evolved",
                 "origin": "mutation", "parent_ids": []},
            ])
        if "format research ideas" in system:
            return ("<description>\nd\n</description>\n<steps>\n1. s\n"
                    "</steps>\n<notes>\nn\n</notes>")
        return "fallback"

    return FakeLLM(responder)


def make_backend() -> FixtureBackend:
    return FixtureBackend({
        "quantile": [
            SearchResult(
                title="Paper on quantile calibration",
                url="https://arxiv.org/abs/0000.0001",
                abstract="Calibration improves WIS.",
                source="arxiv", published="2025-01-01",
            )
        ],
    })


def test_read_key_handles_bom_and_whitespace(tmp_path: Path) -> None:
    key_file = tmp_path / "secret.txt"
    key_file.write_bytes("  sk-test-123  \r\n".encode("utf-8-sig"))
    assert read_deepseek_key(key_file) == "sk-test-123"


def test_extract_json_fences_and_prose() -> None:
    assert extract_json('```json\n[1, 2]\n```') == [1, 2]
    assert extract_json('Here you go: {"a": 1} thanks') == {"a": 1}
    try:
        extract_json("no json here")
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_task_files_present_and_s15_structure() -> None:
    tasks = available_tasks()
    assert "COVID19_住院预测" in tasks
    assert "单细胞批次整合" in tasks
    text, path = load_task("单细胞批次整合")
    assert "SUPERHUMAN METHOD" in text
    assert ".obsm['X_emb']" in text
    covid_text, _ = load_task("COVID19_住院预测")
    # 2026-10-10: rewritten to the restored paper-protocol interface.
    assert "fit_and_predict_fn(train_x, train_y, test_x, config)" in covid_text
    assert "forecast_fn" not in covid_text
    assert "lightgbm" in covid_text and "PyTorch" in covid_text
    assert "SUPERHUMAN METHOD" in covid_text


def test_deep_research_offline(tmp_path: Path) -> None:
    config = load_config()
    task_text, task_path = load_task("COVID19_住院预测")
    manifest = RunManifest(tmp_path, "research", "covid", config)
    manifest.add_file_hash(task_path, "task:covid")
    client = make_client()
    output = run_deep_research(
        client, [make_backend()], task_text, config, tmp_path, manifest
    )
    assert len(output["ideas"]) == 10
    assert all(idea["origin"] == "deep_research" for idea in output["ideas"])
    assert (tmp_path / "ideas.json").exists()
    assert (tmp_path / "research_notes.md").exists()
    assert (tmp_path / "manifest.json").exists()
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["usage_totals"]["calls"] == len(client.history.records)
    assert saved["usage_totals"]["calls"] >= 4
    assert any(q["backend"] == "fixture" for q in saved["search_log"])


def test_coscientist_offline_seeded(tmp_path: Path) -> None:
    config = load_config()
    # shrink for test speed
    config["coscientist"]["rounds"] = 2
    config["coscientist"]["elo_matches_per_round"] = 4
    task_text, _ = load_task("COVID19_住院预测")
    manifest = RunManifest(tmp_path, "coscientist", "covid", config)
    client = make_client()
    seed_ideas = [
        {"id": "R1", "title": "Seeded", "idea": "seed idea",
         "origin": "deep_research"},
    ]
    import random
    output = run_coscientist(
        client, task_text, config, tmp_path, manifest,
        literature=[{"finding": "f", "citations": ["S1"]}],
        seed_ideas=seed_ideas, rng=random.Random(7),
    )
    hyps = output["hypotheses"]
    assert hyps, "no hypotheses produced"
    ids = [h["id"] for h in hyps]
    assert len(ids) == len(set(ids)), "duplicate ids"
    elos = [h["elo"] for h in hyps]
    assert elos == sorted(elos, reverse=True), "not sorted by elo"
    assert (tmp_path / "coscientist_output.json").exists()
    assert (tmp_path / "coscientist_report.md").exists()


def test_elo_math() -> None:
    # equal ratings, A wins: A gains k/2, B loses k/2
    a, b = elo_update(1200.0, 1200.0, 1.0, k=32.0)
    assert abs(a - 1216.0) < 1e-9
    assert abs(b - 1184.0) < 1e-9
    # tie keeps ratings
    a, b = elo_update(1200.0, 1200.0, 0.5, k=32.0)
    assert abs(a - 1200.0) < 1e-9 and abs(b - 1200.0) < 1e-9


def test_filter_and_format_offline(tmp_path: Path) -> None:
    # stage a research pool
    config = load_config()
    task_text, _ = load_task("COVID19_住院预测")
    manifest = RunManifest(tmp_path, "research", "covid", config)
    run_deep_research(
        make_client(), [make_backend()], task_text, config, tmp_path, manifest
    )
    template = write_selection_template(tmp_path)
    selection = json.loads(template.read_text(encoding="utf-8-sig"))
    assert selection["keep"], "template keeps nothing"
    selection["keep"] = selection["keep"][:2]  # simulate human pruning
    template.write_text(json.dumps(selection, ensure_ascii=False), encoding="utf-8")

    pools = load_pools(tmp_path)
    assert set(pools) == {"research"}

    cards = format_selected(make_client(), tmp_path, manifest)
    assert len(cards) == 2
    assert all(card["card_valid"] for card in cards)
    methods = json.loads(
        (tmp_path / "s16" / "methods.json").read_text(encoding="utf-8-sig"))
    assert len(methods) == 2
    assert "<description>" in methods[0]["card"]
    card_files = list((tmp_path / "s16").glob("方法卡_*.md"))
    assert len(card_files) == 2


def test_multistage_manifest_accumulates(tmp_path: Path) -> None:
    """Regression: continuing a run dir must extend, not overwrite, the
    manifest (research -> coscientist -> format provenance chain)."""
    import random

    from idea_system.pipeline import _load_or_create_manifest

    config = load_config()
    config["coscientist"]["rounds"] = 1
    config["coscientist"]["elo_matches_per_round"] = 2
    task_text, _ = load_task("COVID19_住院预测")

    manifest = _load_or_create_manifest(
        tmp_path, "research", "covid", config, True)
    research_client = make_client()
    run_deep_research(
        research_client, [make_backend()], task_text, config,
        tmp_path, manifest,
    )
    cos_manifest = _load_or_create_manifest(
        tmp_path, "coscientist", "covid", config, True)
    cos_client = make_client()
    run_coscientist(
        cos_client, task_text, config, tmp_path, cos_manifest,
        rng=random.Random(1),
    )
    saved = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8-sig"))
    assert [s["stage"] for s in saved["stages"]] == [
        "research", "coscientist"]
    research_calls = sum(
        1 for r in saved["llm_calls"] if r["stage"].startswith(
            ("plan_", "learnings", "followup", "s15"))
    )
    cos_calls = sum(
        1 for r in saved["llm_calls"] if r["stage"].startswith(
            ("generate_", "review_", "rank_", "judge_", "meta_", "evolve_",
             "reflect_"))
    )
    assert research_calls > 0 and cos_calls > 0
    assert saved["usage_totals"]["calls"] == research_calls + cos_calls
    assert saved["search_log"], "research search log lost after coscientist"


def _make_harvest_dir(tmp_path: Path) -> Path:
    from idea_system.corpus import CorpusError, load_corpus  # noqa: F401
    harvest = tmp_path / "harvest"
    harvest.mkdir()
    entries = []
    sources = [
        ("source_01.md", "https://example.org/paper-1", "Harvest paper one",
         "Excerpt about growth-rate anchoring."),
        ("source_02.md", "https://example.org/paper-2", "Harvest paper two",
         "Excerpt about quantile ensembles."),
        ("source_03.md", "https://example.org/paper-1", "Duplicate URL",
         "Should be skipped."),
    ]
    for file_name, url, title, body in sources:
        (harvest / file_name).write_text(
            "---\n"
            f"url: {url}\n"
            f"title: {title}\n"
            "accessed_at: 2026-10-19T00:00:00Z\n"
            "agent: test-agent\n"
            "tool: test\n"
            "---\n"
            f"{body}\n",
            encoding="utf-8",
        )
        entries.append({"file": file_name, "url": url,
                        "agent": "test-agent", "tool": "test"})
    with open(harvest / "harvest_manifest.json", "w", encoding="utf-8") as h:
        json.dump(entries, h, ensure_ascii=False)
    return harvest


def test_corpus_loader_dedupes_and_validates(tmp_path: Path) -> None:
    from idea_system.corpus import CorpusError, load_corpus
    harvest = _make_harvest_dir(tmp_path)
    corpus = load_corpus(harvest)
    assert list(corpus.records) == ["C1", "C2"]
    assert corpus.records["C1"].source == "harvest"
    assert len(corpus.provenance) == 3
    assert sum(1 for p in corpus.provenance
               if p["status"] == "duplicate_url_skipped") == 1
    # missing front-matter url fails loudly
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "source_01.md").write_text("---\ntitle: x\n---\nbody",
                                      encoding="utf-8")
    with open(bad / "harvest_manifest.json", "w", encoding="utf-8") as h:
        json.dump([{"file": "source_01.md"}], h)
    try:
        load_corpus(bad)
        raise AssertionError("should have raised")
    except CorpusError:
        pass


def test_research_with_corpus(tmp_path: Path) -> None:
    from idea_system.corpus import load_corpus
    config = load_config()
    task_text, task_path = load_task("COVID19_住院预测")
    harvest = _make_harvest_dir(tmp_path)
    corpus = load_corpus(harvest)
    manifest = RunManifest(tmp_path, "research", "covid", config)
    manifest.add_file_hash(task_path, "task:covid")
    output = run_deep_research(
        make_client(), [make_backend()], task_text, config, tmp_path,
        manifest, corpus=corpus,
    )
    assert len(output["ideas"]) == 10
    record_ids = list(output["records"])
    assert record_ids[:2] == ["C1", "C2"], "corpus records must come first"
    assert record_ids[2].startswith("S"), "search records numbered after corpus"
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8-sig"))
    assert len(saved["corpus"]) == 3
    assert any(k.startswith("corpus:") for k in saved["file_hashes"])
    # corpus url must not re-enter via search results
    urls = [r["url"] for r in output["records"].values()]
    assert len(urls) == len(set(urls))


def test_takeover_validation(tmp_path: Path) -> None:
    from idea_system.pipeline import _validate_takeover_pool
    good = {
        "ideas": [
            {"title": "Idea A", "idea": "body a",
             "sources": ["https://arxiv.org/abs/1"]},
            {"title": "Idea B", "hypothesis": "body b",
             "sources": ["https://arxiv.org/abs/1", "https://x.org/y"]},
        ]
    }
    ideas, records = _validate_takeover_pool(good)
    assert [i["id"] for i in ideas] == ["R1", "R2"]
    assert all(i["origin"] == "agent_takeover" for i in ideas)
    assert len(records) == 2  # shared url collapsed to one record
    bad_cases = [
        {"ideas": []},
        {"ideas": [{"title": "x", "idea": "y"}]},  # no sources
        {"ideas": [{"title": "x", "idea": "y", "sources": ["not-a-url"]}]},
        {"ideas": [{"title": "x", "idea": "y", "sources": ["https://a.b"]},
                   {"title": "x", "idea": "z", "sources": ["https://a.b"]}]},
    ]
    for bad in bad_cases:
        try:
            _validate_takeover_pool(bad)
            raise AssertionError(f"should have raised: {bad}")
        except ValueError:
            pass


def test_harvest_scaffold_and_cli(tmp_path: Path, monkeypatch) -> None:
    import idea_system.pipeline as pipeline_mod
    # keep test scaffolds out of the real runs/ directory
    monkeypatch.setattr(pipeline_mod, "SYSTEM_ROOT", tmp_path)
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "收割协议_浏览器研究.md").write_text(
        "protocol stub", encoding="utf-8")
    code = pipeline_mod.main([
        "harvest-scaffold", "--task", "COVID19_住院预测"])
    assert code == 0
    harvest_dirs = list((tmp_path / "runs").glob("*_harvest_*/harvest"))
    assert harvest_dirs, "scaffold produced no harvest dir"
    assert (harvest_dirs[0] / "harvest_manifest.json").exists()
    guide = harvest_dirs[0].parent / "收割指引.md"
    assert guide.exists()
    assert "SUPERHUMAN METHOD" in guide.read_text(encoding="utf-8-sig")


def test_arxiv_backend_retries_transient(monkeypatch) -> None:
    """429s are retried with backoff; non-transient 4xx raises immediately."""
    import requests as requests_mod
    from idea_system.websearch import ArxivBackend

    backend = ArxivBackend(politeness_seconds=0.0)
    backend.RETRY_DELAYS = (0.0, 0.0, 0.0)  # no sleeping in tests
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None, headers=None):
        calls["n"] += 1
        if calls["n"] == 1:
            response = requests_mod.Response()
            response.status_code = 429
            raise requests_mod.HTTPError("429", response=response)
        response = requests_mod.Response()
        response.status_code = 200
        response._content = (
            '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            "<title>Retried paper</title>"
            "<id>http://arxiv.org/abs/9999.9999</id>"
            "<summary>It worked after retry.</summary>"
            "<published>2026-01-01T00:00:00Z</published>"
            "</entry></feed>"
        ).encode()
        return response

    monkeypatch.setattr(requests_mod, "get", fake_get)
    results = backend.search("quantile forecasting")
    assert calls["n"] == 2, "429 should be retried once then succeed"
    assert results[0].title == "Retried paper"

    def bad_get(url, params=None, timeout=None, headers=None):
        calls["n"] += 1
        response = requests_mod.Response()
        response.status_code = 400
        raise requests_mod.HTTPError("400", response=response)

    monkeypatch.setattr(requests_mod, "get", bad_get)
    calls["n"] = 0
    try:
        backend.search("bad query")
        raise AssertionError("should have raised")
    except requests_mod.HTTPError:
        assert calls["n"] == 1, "non-transient errors must not retry"
