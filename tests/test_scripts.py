"""scripts/run_live.sh guards (no LLM call is made): it refuses to start without the provider's
key or with an unknown provider, and rejects an unknown --from step or option."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_live.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _run(*args: str, **env: str) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items()
         if k not in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "CAMPUSFLEX_PROVIDER")}
    e.update(env)
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=e, timeout=30)


def test_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_refuses_without_key_or_with_an_unknown_provider():
    r = _run()
    assert r.returncode == 2 and "DEEPSEEK_API_KEY is not set" in r.stderr
    r = _run(CAMPUSFLEX_PROVIDER="anthropic")
    assert r.returncode == 2 and "ANTHROPIC_API_KEY is not set" in r.stderr
    r = _run(CAMPUSFLEX_PROVIDER="sim")
    assert r.returncode == 2 and "must be deepseek or anthropic" in r.stderr


def test_rejects_unknown_step_and_option():
    assert _run("--from", "nope").returncode == 2
    assert _run("--bogus").returncode == 2
    h = _run("--help")
    assert h.returncode == 0 and "llm_check" in h.stdout
