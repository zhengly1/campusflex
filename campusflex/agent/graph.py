"""LangGraph state graph of Figure 2 (Section 3.5).

LLM nodes (L3 interpret, L2 propose, L4 trade-off) only feed forecasts, proposals and
choices into the deterministic column. Only the ``commit`` node holds ``apply_plan``.
If an LLM node fails, the loop continues with masks and rule-staged plans.

    observe → l3_interpret → candidate_pool → l2_propose → compile_validate
            → simulate_score → tradeoff_check → commit → advance → (observe | END)

If ``langgraph`` is not installed, an equivalent minimal runner with the same nodes and
edges is used and reported in ``GraphRun.engine``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from ..config import PlannerConfig
from ..env import CampusEnv
from ..episode import commit
from ..metrics import episode_metrics
from ..policies import Decision, PlanContext, SimulationPlanner
from ..state import Snapshot

try:  # optional dependency
    from langgraph.graph import END, START, StateGraph

    HAVE_LANGGRAPH = True
except Exception:  # pragma: no cover - exercised when langgraph is missing
    HAVE_LANGGRAPH = False
    START, END = "__start__", "__end__"


class GraphState(TypedDict, total=False):
    snap: Snapshot | None
    ctx: PlanContext | None
    decision: Decision | None
    commit: dict[str, Any] | None
    done: bool


class _MiniGraph:
    """Tiny stand-in for ``langgraph.graph.StateGraph`` (sequential nodes, conditional edges)."""

    def __init__(self, _schema: Any):
        self.nodes: dict[str, Callable[[dict], dict]] = {}
        self.edges: dict[str, str] = {}
        self.cond: dict[str, tuple[Callable[[dict], str], dict[str, str]]] = {}

    def add_node(self, name: str, fn: Callable[[dict], dict]) -> None:
        self.nodes[name] = fn

    def add_edge(self, a: str, b: str) -> None:
        self.edges[a] = b

    def add_conditional_edges(self, a: str, fn: Callable[[dict], str], mapping: dict[str, str]) -> None:
        self.cond[a] = (fn, mapping)

    def compile(self) -> "_MiniGraph":
        return self

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        limit = (config or {}).get("recursion_limit", 10_000)
        node = self.edges[START]
        for _ in range(limit):
            if node == END:
                return state
            state = {**state, **(self.nodes[node](state) or {})}
            if node in self.cond:
                fn, mapping = self.cond[node]
                node = mapping[fn(state)]
            else:
                node = self.edges[node]
        raise RecursionError("graph recursion limit reached")


@dataclass
class GraphRun:
    engine: str
    metrics: dict[str, Any]
    steps: list[dict[str, Any]]
    records: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    policy_stats: dict[str, Any] = field(default_factory=dict)


class CampusFlexGraph:
    def __init__(
        self,
        env: CampusEnv,
        planner: SimulationPlanner,
        cfg: PlannerConfig,
        run_id: str = "graph",
        on_step: Callable[[CampusEnv, Snapshot, Decision, dict[str, Any]], None] | None = None,
        force_minigraph: bool = False,
        max_steps: int | None = None,
    ):
        self.env = env
        self.planner = planner
        self.cfg = cfg
        self.run_id = run_id
        self.on_step = on_step
        self.records: list[dict[str, Any]] = []
        self.max_steps = max_steps
        self._n_steps = 0
        self.engine = "langgraph" if (HAVE_LANGGRAPH and not force_minigraph) else "minigraph"
        self.app = self._build()

    # --------------------------------------------------------------- nodes
    def _observe(self, s: GraphState) -> GraphState:
        # Reuse the observation returned by advance_time (it carries the recheck results);
        # a second get_state() would drop them.
        return {"snap": s.get("snap") or self.env.get_state()}

    def _l3(self, s: GraphState) -> GraphState:              # LLM node (via forecaster's interpreter)
        return {"ctx": self.planner.stage_forecast(s["snap"])}

    def _pool(self, s: GraphState) -> GraphState:
        return {"ctx": self.planner.stage_pool(s["ctx"])}

    def _l2(self, s: GraphState) -> GraphState:              # LLM node: proposals only
        return {"ctx": self.planner.stage_l2(s["ctx"])}

    def _compile(self, s: GraphState) -> GraphState:
        return {"ctx": self.planner.stage_compile(s["ctx"])}

    def _score(self, s: GraphState) -> GraphState:
        return {"ctx": self.planner.stage_score(s["ctx"])}

    def _tradeoff(self, s: GraphState) -> GraphState:        # L4 ask/choose (optional)
        ctx = self.planner.stage_tradeoff(s["ctx"])
        return {"ctx": ctx, "decision": self.planner.stage_decision(ctx)}

    def _commit(self, s: GraphState) -> GraphState:          # the only node with apply_plan
        snap, dec = s["snap"], s["decision"]
        key = f"{self.run_id}/t{snap.step}"
        c = commit(self.env, snap, dec, key, self.cfg)
        for r in (dec.notes or {}).get("recheck", []):
            self.env.request_recheck(r)
        if self.on_step is not None:
            self.on_step(self.env, snap, dec, c)
        self.records.append(
            {"step": snap.step, "plan_id": dec.plan_id, "source": dec.source, "trigger": dec.trigger,
             "ranking": dec.ranking[:3], "llm": dec.llm, "commit": c}
        )
        return {"commit": c}

    def _advance(self, s: GraphState) -> GraphState:
        dec, c = s["decision"], s["commit"]
        p = dec.predicted
        extra = {
            "plan_id": c["committed"] if c.get("fallback") else dec.plan_id,
            "source": "fallback" if c.get("fallback") else dec.source,
            "ack": c["ack"],
            "pred_first_V": round(p.first_V, 6) if p else None,
            "pred_first_E": round(p.first_E, 6) if p else None,
            "pred_V": round(p.V, 6) if p else None,
            "pred_E": round(p.E, 6) if p else None,
            "pred_omega": round(p.Omega, 6) if p else None,
            "trigger": dec.trigger,
        }
        nxt = self.env.advance_time(extra)
        self._n_steps += 1
        done = nxt is None or (self.max_steps is not None and self._n_steps >= self.max_steps)
        return {"snap": nxt, "done": done, "ctx": None, "decision": None, "commit": None}

    def _build(self) -> Any:
        G = StateGraph(GraphState) if self.engine == "langgraph" else _MiniGraph(GraphState)
        nodes = [
            ("observe", self._observe),
            ("l3_interpret", self._l3),
            ("candidate_pool", self._pool),
            ("l2_propose", self._l2),
            ("compile_validate", self._compile),
            ("simulate_score", self._score),
            ("tradeoff_check", self._tradeoff),
            ("commit", self._commit),
            ("advance", self._advance),
        ]
        for name, fn in nodes:
            G.add_node(name, fn)
        G.add_edge(START, "observe")
        for (a, _), (b, _) in zip(nodes, nodes[1:]):
            G.add_edge(a, b)
        G.add_conditional_edges("advance", lambda s: "end" if s.get("done") else "replan",
                                {"end": END, "replan": "observe"})
        return G.compile()

    def run(self, reset_planner: bool = True, close: bool = True) -> GraphRun:
        if reset_planner:
            self.planner.reset()
        self._n_steps = 0
        self.app.invoke({"done": False}, config={"recursion_limit": 10_000})
        if close:
            self.env.device.close()
        return GraphRun(
            engine=self.engine,
            metrics=episode_metrics(self.env.step_logs, self.cfg),
            steps=self.env.step_logs,
            records=self.records,
            counters=dict(self.env.counters),
            policy_stats=dict(self.planner.stats),
        )


def run_graph_episode(scenario, planner: SimulationPlanner, cfg: PlannerConfig, device_layer=None,
                      run_id: str | None = None, **kw: Any) -> GraphRun:
    env = CampusEnv(scenario, cfg, device_layer)
    g = CampusFlexGraph(env, planner, cfg, run_id or f"{scenario.name}/s{scenario.seed}/{planner.name}/graph", **kw)
    return g.run()
