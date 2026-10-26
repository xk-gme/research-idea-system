"""open-coscientist engine tests: mapping, fallback, construct smoke.

All zero-API. The construct smoke requires ocs-venv and is skipped
otherwise (same pattern as the MCP smoke wrapper).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idea_system.ocs_bridge import map_ocs_hypotheses  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OCS_PYTHON = ROOT / "ocs-venv" / "Scripts" / "python.exe"


def test_map_ocs_hypotheses() -> None:
    raw = [
        {
            "text": "Growth-rate anchored quantile ensemble\nwith horizon scaling",
            "elo_rating": 1287,
            "score": 82.5,
            "generation_method": "literature",
            "evolution_history": [],
            "win_count": 5, "loss_count": 1, "total_matches": 6,
            "reviews": [{
                "review_summary": "sound",
                "scores": {"novelty": 8, "feasibility": 7},
                "overall_score": 7.5,
            }],
            "explanation": "layman text",
            "literature_grounding": "grounded [P1]",
            "experiment": "design",
        },
        {
            "text": "Evolved hybrid method",
            "elo_rating": 1240,
            "generation_method": "debate",
            "evolution_history": ["refined once"],
        },
        {"text": "   "},  # empty -> dropped
    ]
    mapped = map_ocs_hypotheses(raw)
    assert [h["id"] for h in mapped] == ["H1", "H2"]
    assert mapped[0]["elo"] == 1287
    assert mapped[0]["origin"] == "ocs:literature"
    assert mapped[1]["origin"] == "ocs:debate+evolved"
    assert mapped[0]["reviews"][0]["scores"] == {
        "novelty": 8, "feasibility": 7}
    assert mapped[0]["title"].startswith("Growth-rate anchored")
    assert mapped[0]["explanation"] == "layman text"


def test_coscientist_dry_run_falls_back_from_ocs(tmp_path: Path,
                                                 monkeypatch) -> None:
    """--engine ocs in dry-run must fall back to the self engine and still
    produce a pool (offline, zero API)."""
    import idea_system.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "SYSTEM_ROOT", tmp_path)
    code = pipeline_mod.main([
        "coscientist", "--task", "COVID19_住院预测",
        "--engine", "ocs", "--dry-run"])
    assert code == 0
    runs = list((tmp_path / "runs").glob("*_coscientist_*"))
    assert runs, "no coscientist run dir created"
    out = json.loads(
        (runs[0] / "coscientist_output.json").read_text(encoding="utf-8-sig"))
    assert out["hypotheses"], "self-engine fallback produced nothing"
    assert out.get("engine", "self") == "self"


@pytest.mark.skipif(not OCS_PYTHON.exists(), reason="ocs-venv not built")
def test_ocs_construct_smoke_zero_api() -> None:
    """Construct HypothesisGenerator inside ocs-venv (no API call)."""
    code = subprocess.run(
        [str(OCS_PYTHON), "-c",
         "from open_coscientist import HypothesisGenerator; "
         "HypothesisGenerator(model_name='deepseek/deepseek-flash', "
         "max_iterations=1, initial_hypotheses_count=3, "
         "evolution_max_count=2); print('construct ok')"],
        capture_output=True, text=True, timeout=120, cwd=str(ROOT),
    )
    assert code.returncode == 0, code.stdout + code.stderr
    assert "construct ok" in code.stdout
