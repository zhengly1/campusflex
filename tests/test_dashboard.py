"""Dashboard (Table 11) rendered headlessly with Streamlit's AppTest; skipped without streamlit.

Every step of the demo episode renders without an exception, every metric shows the value of
the summary field named in its label, and a missing episode file gives a warning, not a crash.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

# Streamlit's own chart code triggers a pandas deprecation warning; not ours to fix.
pytestmark = pytest.mark.filterwarnings("ignore:The inplace keyword:Warning")

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "campusflex" / "dashboard" / "app.py")


@pytest.fixture()
def episode(tmp_path: Path) -> Path:
    from campusflex.config import PlannerConfig
    from campusflex.episode import run_episode
    from campusflex.policies import make_policy
    from campusflex.scenarios import make_scenario

    cfg = PlannerConfig()
    p = tmp_path / "ep.json"
    run_episode(make_scenario("fault", 0), make_policy("planner", cfg), cfg, keep_decisions=True).save(str(p))
    return p


def _app(path: Path) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(path)).run()
    return at


def test_every_step_renders_and_metrics_are_the_logged_values(episode: Path):
    at = _app(episode)
    assert not at.exception
    summ = json.loads(episode.read_text())["summary"]
    assert at.metric, "no summary metrics shown"
    for m in at.metric:
        field = m.label.removeprefix("summary.")
        assert m.label.startswith("summary.") and m.value == str(summ[field])
    assert any(h.value.startswith("candidate ranking") for h in at.subheader)
    clocks = at.select_slider[0].options
    assert len(clocks) == 24
    for c in clocks:
        at.select_slider[0].set_value(c).run()
        assert not at.exception, (c, [e.value for e in at.exception])
        assert any(h.value == f"rooms at {c}" for h in at.subheader)


def test_missing_episode_file_warns_instead_of_crashing(tmp_path: Path):
    at = _app(tmp_path / "nope.json")
    assert not at.exception
    assert any("not found" in w.value for w in at.warning)
