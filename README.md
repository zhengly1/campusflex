# CampusFlex v2.1 — LLM-in-the-loop campus space & energy management

Code implementing the *Revised Project Design (v2.1)* (`CampusFlex_v2.1_preview.pdf`, not included in this repository):
a simulation-checked replanning controller for a synthetic six-room campus (18:00–22:00,
decisions every 10 min), five LLM roles that **propose** (never commit), the deterministic
validator/simulator that **decides**, an MQTT device layer, a LangGraph agent, a Streamlit
dashboard and the pre-registered experiments E1–E6 with the evidence ledger.

> Design principle: **the LLM proposes, the simulator decides, the validator vetoes.**
> All energy and temperature values are simulated estimates from a synthetic campus, not
> measurements from a university building.

## Install

```bash
pip install -e .            # core: numpy, pydantic (enough for every non-LLM item)
pip install -e ".[all]"     # + openai (DeepSeek client), anthropic, langgraph, paho-mqtt, streamlit, pandas, pytest
```

Without the optional packages the code still runs: the agent graph uses a built-in
runner with the same nodes, and the device layer falls back to the in-process bus
(reported as a fallback, Section 3.7).

A local MQTT broker without root (for ledger item 2 and `t1`, whose default device is MQTT):

```bash
conda create -y -p ~/mosquitto-env -c conda-forge mosquitto=2.0.20
printf 'listener 1883 127.0.0.1\nallow_anonymous true\n' > ~/mosquitto.conf
~/mosquitto-env/sbin/mosquitto -c ~/mosquitto.conf &
```

## Run in ledger order (Section 4.8)

| # | Item | Command | Needs LLM |
|---|------|---------|-----------|
| – | Freeze gate (once, before any held-out run) | `python -m campusflex freeze` → `results/frozen_config.json`: `config_hash`, per-role prompt hashes (L1–L5 and L4_explain for E4), model, held-out `notices_hash` | no |
| – | LLM readiness check (after setting the key, before items 5–8) | `python -m campusflex llm-check` → `results/llm_check.json` | yes |
| 0 | v1 core, 160 episodes, tests | `python -m campusflex v1` · `pytest -v -rs > results/test_results.txt` (item 2 is judged from the same file: run it with paho-mqtt and the broker up) | no |
| 1 | Predicted-vs-realized V log, Ω ablation | `python -m campusflex vgap` | no |
| 2 | MQTT device layer + tests | `pytest tests/test_iot.py` (the three real-broker tests need paho-mqtt and a broker on `localhost:1883`) | no |
| 3 | E5 scripted suite (120 fixtures) | `python -m campusflex e5` | no |
| 4 | E1 arms A, B (held-out seeds 10–19) | `python -m campusflex e1 --arms AB` | no |
| 5 | E1 arms C, D; E5 live sub-test | `python -m campusflex e1 --arms CD [--reduced]` · `python -m campusflex e5 --live` | yes |
| 6 | Joint trace T1 + counterfactuals | `python -m campusflex t1` (`--rehearsal` = non-LLM pipeline check) | yes |
| 7 | E2, E3, E4 | `python -m campusflex e2 / e3 / e4` (add `--llm` for the LLM arm) | partly |
| 8 | L5 review + E6 (stretch) | `python -m campusflex e6 review --kind random\|llm` · `python -m campusflex e6 evaluate` | partly |
| – | Tables / ledger | `python -m campusflex tables` · `python -m campusflex ledger` | no |
| – | Items 5–8 in one go (after `export DEEPSEEK_API_KEY=…`) | `scripts/run_live.sh [--reduced] [--prices IN HIT OUT]` · resume with `--from STEP` | yes |
| – | One episode / dashboard | `python -m campusflex run --scenario fault --seed 10 --policy arm_B --device bus --save results/demo_episode.json` · `streamlit run campusflex/dashboard/app.py -- --episode results/demo_episode.json` | no |

`scripts/run_live.sh` runs `llm-check` first (it must report `ready`: key, balance, latency and
an unchanged freeze record), then E1 C/D, E5 live, T1, E2–E4, the L5 review, E6, tables and the
ledger. It does not re-run the freeze gate, reads the key from the environment only, and stops at
the first step that fails or records `not_run`/`aborted` (e.g. HTTP 402), printing the
`--from` step to resume with. A resumed step runs again from its start; E1 C/D, E3, E5 live and
`llm-check` are uncached by design, so their calls are made (and billed) again.

`results/evidence_ledger.md` is computed from the files on disk: a result is cited only
when its artifact exists and passes the protocol checks (held-out seeds, expected episode
count, frozen config, the same `config_hash` in `e1_AB.json` and `e1_CD.json`, and for LLM
results the prompts and model of the freeze record — `prompts_frozen`/`notices_frozen = false`
makes an item Partial). Items 4–8 are also checked against the *current* freeze record, so a
later `freeze --force` (other config, prompts, model or notices) turns older artifacts Partial.
Items 6–7 check their own protocol: T1 on a held-out seed, E2 on all 40 requests, E3 on seeds
10–19 × every notice scenario with three uncached LLM repeats, E4 on fresh decisions of seeds
10–11; an E5 live variant that evaluated too few runs to claim its target makes item 5 Partial,
and a `review --quick` makes E6 reduced. Every LLM experiment first
makes one real preflight call; if it fails (no SDK / credentials) the arm is written as
`not_run`. After a run, an arm whose calls failed or fell back more than half the time is
marked `invalid_mostly_fallback` and is not compared — a masks-only fallback is never
reported as an LLM result. A fallback is a call that failed (error, timeout, unparseable or off-schema reply); an L2
reply with an empty plan list — which the L2 prompt allows when nothing staged would help — is
the model's answer, counted separately as `empty_reply_rate`, not as a fallback. In the E5 live
sub-test each with/without-injection pair replays the first reply for an identical prompt, so a
second sampling of the same prompt cannot be mistaken for injection-attributed divergence. If the account fails mid-run (HTTP 401 bad key / 402 insufficient balance), E1 C/D stops,
skips the remaining episodes and writes `status: aborted` with the reason (top up and rerun the
whole run); E5 live stops the same way. In E2, E3, E4 and the L5 review an account error marks the
LLM arm `aborted` (not `invalid_mostly_fallback`): failures caused by the key or balance say nothing
about the model. Calls abandoned at a step's deadline are still billed:
their tokens are collected when they complete and recorded as `meta.abandoned_calls` (E1 C/D),
and `llm-check` averages tokens over calls that reported usage. The first live attempt on 2026-11-04 ran out of balance after ~700 of ~5,200 L2
calls; that file is kept for audit in `results/failed_runs/` and is not ledger evidence. E1 C/D also writes its decisions and L2 transcripts to
`results/e1_CD_traces.jsonl.gz`; E3's LLM arm runs three uncached repeats per seed
(`--cache` to reuse the cache).

### LLM configuration

* **Default provider: DeepSeek**, model `deepseek-flash`, through its OpenAI-compatible API
  (`openai` SDK, `base_url=https://api.deepseek.com`). Set `DEEPSEEK_API_KEY`; optional
  `DEEPSEEK_BASE_URL`, `CAMPUSFLEX_MODEL` (the API listed `deepseek-flash` and `deepseek-v4-pro`
  on 2026-11-04; the legacy id `deepseek-chat` is still accepted and served as `deepseek-flash`).
  Changing the model after the freeze gate marks later LLM results `prompts_frozen=false`.
* **Cost (DeepSeek price page, 2026-11-04, USD per 1M tokens, `deepseek-flash`):** input
  cache hit 0.006 / miss 0.30, output 1.20 at peak (weekdays 01:00–04:00 and 06:00–10:00 UTC);
  half of that off-peak. All LLM experiments together ≈ 6,000 calls ≈ **$6 at peak, $3
  off-peak**, ~98 % of it E1 C/D (≈ 5,200 L2 calls, ~760 output tokens each). Check the price
  page before a run: `llm-check --price-in 0.30 --price-cache-hit 0.006 --price-out 1.20`
  (`scripts/run_live.sh --prices 0.30 0.006 1.20` passes the same values).
* Alternative provider: `CAMPUSFLEX_PROVIDER=anthropic` (or `--provider anthropic`) uses Claude
  (`claude-opus-5-5`) via the `anthropic` SDK with server-side JSON-schema output.
* Every LLM command accepts `--provider` and `--model`.
* DeepSeek runs in JSON mode (`response_format={"type": "json_object"}`) at **temperature 0**;
  the role's JSON schema is sent in the system prompt and every reply is checked against it
  locally (`llm.client.schema_errors`) before the role's own validator sees it. A reply that
  is empty, truncated, filtered or off-schema counts as a failed call → fallback.
* Prompt-hash cache in `.cache/llm/`, keyed by model, prompt, schema and generation settings;
  only successful replies are stored, with their original latency, so a replay makes the same
  budget decisions as the live call. Not used by E1 C/D, E5 live and `llm-check`, nor by E3's
  three repeats (`e3 --cache` reuses it); E2, E4, T1 and the L5 review use it.
* Per step: ≤ 3 LLM calls and 20 s in total (cumulative), enforced as a wall-clock deadline
  on each call; SDK retries off; then fallback to masks + rule-staged plans. Output caps per
  role (DeepSeek): L1 1200, L2 2500, L3 800, L4 600, L5 1500 tokens.

```bash
export DEEPSEEK_API_KEY=sk-...
python -m campusflex llm-check             # ~1 min: per-role latency vs the 20 s budget, tokens, E1 C/D projection
python -m campusflex e1 --arms CD          # ledger 5 (add --reduced for the reduced protocol)
python -m campusflex e5 --live             # ledger 5
python -m campusflex t1                    # ledger 6
python -m campusflex e2 --llm; python -m campusflex e3 --llm; python -m campusflex e4 --llm   # ledger 7
python -m campusflex e6 review --kind llm; python -m campusflex e6 evaluate                    # ledger 8
python -m campusflex tables; python -m campusflex ledger
```

`llm-check` sends each role's real prompt three times (uncached) on the E5 development-seed
state and reports success, usable outputs (L2: plans passing the validator), median/max
latency against the 20 s step budget, tokens incl. DeepSeek prefix-cache hits, and projects
E1 C/D (≈ 5,200 L2 calls over 480 episodes; PDF: "about 4,800"): wall time at `--threads`, and
cost if you pass `--price-in/--price-out` (USD per 1M tokens). It exits non-zero
(`check_warnings`) when half or more of a role's calls fail or reply unusably, on any account
error (HTTP 401/402), when the median latency of L2 or L3 exceeds 75 % of the 20 s budget, or
when the prompts or model differ from the freeze record; a single bad reply is only a note. It
touches no held-out seed and is not ledger evidence.

The freeze record also locks the prompts (Figure 4: "prompts, weight levels, λΩ and held-out
notices locked"): one hash per role over its system prompt, schema, user-prompt builder and
generation settings (`campusflex/llm/prompts.py`). Editing a prompt, switching the model or
rewriting the held-out notices after the gate labels later LLM results `prompts_frozen=false` /
`notices_frozen=false`. When a teammate rewrites the held-out notices (assumption 8 below), re-run
`python -m campusflex freeze --force` **before** any held-out run.

## Where each part of the PDF lives

| PDF | Module |
|-----|--------|
| §2.1 Table 4, §3.1 Table 6, Table 8, Table 9 | `campusflex/config.py` |
| §3.1 Eq. 1–3 (RC model, energy, feasibility) | `campusflex/model.py` |
| Scenarios (v1 + fault variants + Notice family, mismatch ±10%) | `campusflex/scenarios.py`, `data/notice_corpus.json` |
| Eq. 4 score with terminal term Ω, rollout | `campusflex/simulate.py` |
| Plan language + compiler (L2 example) | `campusflex/plans.py` |
| Validator (unknown/offline rooms, capacity, dwell, windows) | `campusflex/validator.py` |
| 64 masks + rule-staged generator (§3.3) | `campusflex/candidates.py` |
| Rule forecast + bounded notice adjustments (±80, confidence, later-notice-wins, grounding) | `campusflex/forecast.py` |
| Tool contract get_state/simulate/validate/apply/advance (versions, idempotency, ack) | `campusflex/env.py`, `campusflex/ledger.py`, `campusflex/devices.py` |
| Baselines, efficiency-ranked threshold, planner, arms A–D | `campusflex/policies.py` |
| Metrics Eq. 5, Table 15 | `campusflex/metrics.py` |
| Paired bootstrap | `campusflex/stats.py` |
| LLM guardrails (schemas, cache, budget, data-field delimiting) | `campusflex/llm/client.py` |
| L1 intent compiler · L2 proposer · L3 interpreter · L4 negotiator · L5 reviewer | `campusflex/roles/l1_intent.py` · `l2_proposer.py` · `l3_events.py` · `l4_negotiator.py` · `l5_reviewer.py` |
| Faithfulness checker (L4, L5 report, E4) | `campusflex/roles/faithfulness.py` |
| Figure 2 LangGraph state graph | `campusflex/agent/graph.py` |
| Table 11 MQTT layer (topics `campus/<room>/telemetry\|cmd\|ack`, QoS 1, seq, stale, missing ack) | `campusflex/iot/` |
| Experiments E1–E6, T1, v1, V-gap, tables, ledger | `campusflex/experiments/` |
| Table 14 fixtures | `tests/e5_fixtures/` (authored by `campusflex/experiments/e5_fixtures.py`) |
| §5.4 team records | `docs/reviews.md`, `docs/disagreements.md`, `docs/integration_log.md`, `docs/contrib.md` (templates) |

## Assumptions and deviations from the PDF (read before citing numbers)

1. **The v1 code was not in this repository.** The numerical core was reconstructed from
   Equations 1–5 and Tables 4, 6, 8. Unspecified details (demand profiles, outdoor
   temperature, heat timing, seat-assignment order, thermostat) are declared choices in
   `scenarios.py` / `model.py`, so **numbers do not reproduce Table 12**. The reconstructed v1
   run shows the same pattern: the planner saves ≈27% kWh against fixed opening, has higher
   deviation and relocations, and a worse full-evening score than the threshold rule.
2. AC actuation is a device-local thermostat that drives each room to its setpoint in one
   interval within `0 ≤ u ≤ h` (nominal parameters; the plant may be perturbed).
3. Fixed opening opens inventory-order rooms covering 180 planned seats (4 rooms, matching
   v1's 4.00 switches). Room dwell = 2 steps, student dwell = 3 steps, ≤ 2 moves per student,
   pre-cool targets in [22, 27] °C — declared settings.
4. The rule-staged generator is implemented literally ("idle room with enough seats"). With
   70 students in B201 no single idle room fits, so in the E1 fault scenarios it rarely emits
   plans; arm B ≈ arm A there. A split transfer is exactly what an LLM proposal can add.
5. **LLM provider.** The PDF names no vendor; DeepSeek (`deepseek-flash`) is the default here.
   It accepts the PDF's **temperature 0**, but temperature 0 is not guaranteed bit-identical
   across calls, so replay reproducibility still comes from the prompt-hash cache and model
   variance is measured by the uncached repeats. DeepSeek's JSON mode does not enforce a
   schema server-side, hence the local schema check. With the Claude provider temperature
   cannot be set (sampling parameters were removed on current models). The provider may
   report a model under another name than requested (the legacy `deepseek-chat` came back as
   `deepseek-flash`); the freeze gate compares the requested id, and E1 C/D records the
   reported names of successful calls as `served_models`.
6. Extra deterministic defences beyond the PDF:
   * **L3:** notices that look like instructions ("SYSTEM:", "ignore previous", "report
     delta", …) are withheld from the model; an adjustment must cite a released notice,
     quote it verbatim, take |delta| from a count phrase in it ("about 60 students"; clock
     times never count) and start at a clock time written in it; one adjustment per
     (notice, kind); deltas are clipped to ±80. Table 10's "two notices contradict → use the
     later notice" is applied to any overlapping adjustments of the same kind (so two
     independent overlapping events of one kind collapse to the later one).
   * **L1:** every number in a compiled request must be written in the request or equal a
     declared default (listed in `defaults_filled`); otherwise the field counts as missing
     and a clarifying question is asked. The default interval 18:00–22:00 needs a time
     reference ("tonight", "this evening", "usual hours") or a written clock time; a student
     count is grounded only by numbers that are not temperatures, kWh/kW, percentages or
     durations (or a sum of two such numbers). An inverted band or interval, or a target/band
     below the lowest AC setpoint (22 °C), is a conflict that triggers a question.
     Band/target must lie in 18–30 °C, and `apply_request` refuses a band with low ≥ high or a
     target outside [22 °C, band high] (no AC setpoint would be valid).
   * **L3 (LLM):** besides start and delta, the end time must be written in the notice or be
     the declared default (start + 2 h, at most 22:00), and the evidence quote may not be empty.
   * **Faithfulness checker:** a logged fraction may appear as a percentage only before
     "%"/"percent"; scientific notation and number words followed by a unit ("fifteen
     percent", "two moves") are checked; "Option 1.5 kWh" is not an option label. A number
     written with a unit must match a logged value of that unit: each value takes its unit
     from its field name (`kwh`/`E`, `mean_dev_c`/`dev_mean`, `moves`/`M`, `K`, `U`,
     `*_min`), so "6 kWh" is flagged when 6 is only a move count (listed in
     `unit_mismatch`). Values of unnamed fields (`extra_allowed`, parameter values) match
     any unit; a number without a unit is matched by value alone.
   * **Prompt hashes** cover the helpers that render each role's data (`Snapshot.summary` for L2,
     L4 option views, the E4 ranking view, the L5 digest), not only the prompt builders.
   * **E5 live** reports two variants on the same 20 injected notices: *deployed* (screen
     on) and *model_only* (screen off, so the model itself is tested). The screen patterns
     were written by the same author as the fixtures, so the deployed figure is optimistic.
7. The full-evening score always uses the v1 reference weights (λT=8, λM=0.06, λS=0.08) so
   configurations with different weights (L1, L5) are compared on one yardstick.
8. `data/e2_requests.json` holds 40 newly written labelled requests; the 30 v1 requests were
   not available and should be merged in before the freeze gate. Held-out notice wording in
   `data/notice_corpus.json` should be rewritten by a member who did not write the prompts.
9. The team-process files in `docs/` are templates only; no review records are fabricated.
10. **Runtime safety (declared choices where the PDF is silent):** the controller's belief
    about each room is corrected from telemetry taken after its last command (a lost ack is
    detected as `drift`, and an occupied room that is physically closed counts as a hard
    violation); room switches are make-before-break (closes are sent only after every newly
    opened room acknowledged, otherwise deferred); an `ac_offline` ack marks the unit offline
    until a newer reading; a room without a fresh reading is not opened; devices refuse
    commands older than the last one applied; runs sharing a broker ignore each other's
    messages (run id in every payload); setpoints must lie in [22 °C, band high].
11. **Planner details:** the rollout and Ω window stop at 22:00; explicit moves are applied
    before forced relocation (a transfer out of a closing room reaches its named target, and
    nobody is moved twice in one step); dwell rules are checked on every step of a staged
    plan, not only the first; an L1 energy budget is checked pro rata and ranks compliant
    plans before non-compliant ones at equal unserved demand (service stays first).
    Moves at one offset run in the order written; `"count": "all"` means whoever is still in
    the source room after the earlier moves out of it (as `model.assign` executes it), so a
    split transfer is "40 to B202, then all to A103". The validator checks this per offset and
    rejects a move that exceeds the destination's seats instead of trimming it; the L2 prompt
    says so. (Before this was written into the validator and the prompt, the first live
    `llm-check` showed 2 of 3 L2 replies with no valid plan: the model split the 70 students of
    B201 correctly but every "all" was counted as 70.)
    Later steps of a staged plan are checked against the state the rollout predicts for them
    with the commit-time rules (free seats, moves into closed rooms, closing an occupied room in
    a no-move window): a plan whose later step the validator would refuse when it comes due is
    rejected (and the reason returned to L2) instead of being scored on trimmed moves. Seats freed
    by an earlier move out of a room count for a later move into it at the same step.
12. **Comparisons (E1, E6) in Eq. 4 order:** unserved person-minutes decide when their paired
    interval excludes zero; when they are equal in every pair or their interval includes zero,
    the score decides (`decided_by: score_unserved_tied`). The V-gap log skips steps whose
    committed action was a fallback (the prediction belonged to another plan).
13. **E6 budget:** L5 stops early after two rounds without an accepted diff (Section 3.4); the
    random-search control always spends the full 18 proposals ("the same budget"), so it never
    has less evaluation budget than L5. `e6.json` records both budgets.

## Tests

`pytest` → 190 passed (`results/test_results.txt`, run with paho-mqtt, streamlit and a local
Mosquitto 2.0.20 broker; the MQTT tests also pass on amqtt 0.12.1). Without paho-mqtt/a broker
the three real-broker tests are skipped, and without streamlit the dashboard tests
(`tests/test_dashboard.py`, Streamlit AppTest: every step of an episode renders, each metric is
the logged value) are skipped: 185 passed, 4 skipped. `tests/test_core.py` holds the 13 core invariants; the others cover the plan
language/validator/graph, IoT layer (incl. drift, partial acks, out-of-order commands), L1/L3/L4
roles and their grounding, L5, the DeepSeek client (`tests/test_llm_client.py`, fake transport,
no network), the experiment protocol checks incl. the prompt/notice freeze (`tests/test_experiments.py`),
the readiness check (`tests/test_llm_check.py`), the budgeted call path (`tests/test_llm_call.py`:
step deadline, tokens of abandoned calls, client exceptions) and the guards of `scripts/run_live.sh` (`tests/test_scripts.py`, no LLM call).
