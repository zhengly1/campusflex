"""Dashboard (Table 11) rendered headlessly with Streamlit's AppTest; skipped without streamlit.

Every step of the demo episode renders without an exception, every metric shows the value of
the summary field named in its label, and a missing episode file gives a warning, not a crash.
The smart devices page does the same for a small smart experiment.
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


def test_replayed_llm_episode_shows_provenance_and_transcripts(tmp_path: Path):
    from campusflex.dashboard.replay_llm_demo import TRACES, load_trace, replay

    if not TRACES.exists():
        pytest.skip("no recorded live run")
    out = tmp_path / "llm.json"
    rep = replay(load_trace("arm_C", "fault", 10, 0), out)
    assert rep["verified"] and rep["replay_misses"] == 0       # the recorded run still reproduces exactly
    at = _app(out)
    assert not at.exception and any("provenance: replay" in i.value for i in at.info)
    d = json.loads(out.read_text())
    k = next(x["step"] for x in d["decisions"] if x.get("llm"))
    at.select_slider[0].set_value(d["steps"][k]["clock"]).run()
    assert not at.exception and any(h.value.startswith("LLM transcripts") for h in at.subheader)


@pytest.fixture(scope="module")
def smart_dir(tmp_path_factory) -> Path:
    from campusflex.experiments.smart import run

    d = tmp_path_factory.mktemp("smart")
    run(train_days=4, eval_days=1, workers=4, out_dir=d)
    return d


def _smart_app(d: Path) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.switch_page("smart_page.py").run()
    at.sidebar.text_input[0].set_value(str(d)).run()
    return at


def test_smart_page_metrics_are_the_logged_values_and_every_view_renders(smart_dir: Path):
    at = _smart_app(smart_dir)
    assert not at.exception and at.title[0].value == "CampusFlex – smart devices"
    summ = json.loads((smart_dir / "summary.json").read_text())
    assert at.metric
    for m in at.metric:
        v = summ
        for key in m.label.split("."):
            v = v[key]
        assert m.value == str(v), m.label
    evg = json.loads((smart_dir / "evening.json").read_text())
    arm, room = at.selectbox[1], at.selectbox[2]
    assert arm.value == "smart" and room.value == evg["faults"][0]["room"]
    flags = [f for t in evg["arms"]["smart"]["ticks"] for f in t["flags"] if f["room"] == room.value]
    assert sum("ticks[*].flags" in w.value for w in at.warning) == len(flags)
    for field in at.selectbox[0].options:
        at.selectbox[0].set_value(field).run()
        assert not at.exception, field
    for r in at.selectbox[2].options:                              # every room under the smart arm …
        at.selectbox[2].set_value(r).run()
        assert not at.exception, r
    for a in at.selectbox[1].options:                              # … and every arm
        at.selectbox[1].set_value(a).run()
        assert not at.exception, a
    clocks = at.select_slider[0].options
    for c in clocks[::10] + clocks[-1:]:
        at.select_slider[0].set_value(c).run()
        assert not at.exception, c
        assert any(h.value.startswith(f"all rooms at {c}") for h in at.subheader)


def test_smart_page_without_results_warns(tmp_path: Path):
    at = _smart_app(tmp_path)
    assert not at.exception and any("not found" in w.value for w in at.warning)
