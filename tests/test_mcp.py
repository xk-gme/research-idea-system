"""MCP server stdio smoke test wrapper (runs only when mcp-venv exists)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / "mcp-venv" / "Scripts" / "python.exe"
SMOKE = ROOT / "mcp_server" / "smoke_test.py"

pytestmark = pytest.mark.skipif(
    not VENV_PYTHON.exists() or not SMOKE.exists(),
    reason="mcp-venv not built (see README: pip install 'mcp<2')",
)


def test_mcp_stdio_smoke() -> None:
    result = subprocess.run(
        [str(VENV_PYTHON), str(SMOKE)],
        capture_output=True, text=True, timeout=180,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"smoke failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "SMOKE OK" in result.stdout
