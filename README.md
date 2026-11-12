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
| – | Freeze gate (once, before any held-out run) | `python -m campusflex freeze` → `results/frozen_config.json`: `config_hash`, per-role prompt hashes (L1–L5 and L4_explain for E4), model, held-out `notices_hash`, `scenario_hash` (scenario generator inputs and code) | no |
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
| – | One episode / dashboard | `python -m campusflex run --scenario fault --seed 10 --policy arm_B --device bus --save results/demo_episode.json` · `streamlit run campusflex/dashboard/app.py -- --episode results/demo_episode.json [--smart-dir results/smart]` | no |

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

### Results of the live run (2026-11-10, `deepseek-flash`, frozen configuration)

All values are simulated estimates; full tables in `results/tables.md`, statuses in
`results/evidence_ledger.md` (items 0–8 Executed; item 9 needs the team's own records).

* **E1 (L2 staged plans):** arms C and D beat B on the Eq. 4 order (no unserved demand in any
  arm; score 147.0 → 76.3 and 75.0, C − B = −70.7 [−88.5, −53.9]); the gain comes from the
  three fault scenarios (−72 to −126 per scenario), where an LLM split transfer out of the
  failed 70-seat B201 is something the rule generator cannot express; heat: no difference.
  L2 failed or fell back in 0.9 % (C) and 1.5 % (D) of calls; 88 % of its plans passed the
  validator; LLM plans were committed in 7 % of steps.
* **E5 live:** 0 injection-attributed divergence in both variants (40/40 runs each). In the
  deployed variant the screen withheld every injected notice, so that figure tests the screen.
* **T1:** complete – at 20:00 the committed action depends jointly on the notice, the IoT
  fault message and the staged L2 plan (removing any one changes it).
* **E2:** LLM compiler field accuracy 0.99 (structured form 1.00), no hard constraint omitted,
  weight levels 92.5 % correct, clarifications 100 % correct.
* **E3:** success criterion not met – the LLM interpreter equals the keyword parser
  (30-minute forecast MAE 10.9 vs 10.5; oracle 5.6). Over 120 evenings with one real notice
  each it had 117 adjustments applied (the keyword parser 1 per evening); grounding refused its
  other 213 proposals, which is why its `valid_proposal_rate` is only 0.35. This run did not
  record which notices those were; E3 now records `dropped_by_source` and
  `true_notice_coverage` so that a rerun separates rightful refusals from missed notices. The
  held-out notices are still template sentences the keyword parser reads perfectly, and the
  gap to the oracle is the event end time, which no notice states; the notices should be
  rewritten by a member who did not write the prompts before this test can show an LLM gain.
* **E4:** 407 of 407 numbers in LLM explanations matched a logged value; none shown unverified.
* **E6:** success criterion not met – L5 made 6 proposals (2 rounds, early stop), random
  search 18; none passed the paired test, so both reviewed configurations equal the
  pre-review one. This is the pre-registered falsification outcome ("L5 ≈ random search").

### LLM configuration

* **Default provider: DeepSeek**, model `deepseek-flash`, through its OpenAI-compatible API
  (`openai` SDK, `base_url=https://api.deepseek.com`). Set `DEEPSEEK_API_KEY`; optional
  `DEEPSEEK_BASE_URL`, `CAMPUSFLEX_MODEL` (the API listed `deepseek-flash` and `deepseek-v4-pro`
  on 2026-11-04; the legacy id `deepseek-chat` is still accepted and served as `deepseek-flash`).
  Changing the model after the freeze gate marks later LLM results `prompts_frozen=false`.
* **Cost (DeepSeek price page, 2026-11-04, USD per 1M tokens, `deepseek-flash`):** input
  cache hit 0.006 / miss 0.30, output 1.20 at peak (weekdays 01:00–04:00 and 06:00–10:00 UTC);
  half of that off-peak. All LLM experiments together ≈ 6,000 calls ≈ **$6 at peak, $3
  off-peak**, ~98 % of it E1 C/D (≈ 5,200 L2 calls, ~760 output tokens each). Measured on
  2026-11-10 (thinking off): E1 C/D made 5,110 L2 calls with 9.1M input tokens (6.4M prefix-cache
  hits) and 3.9M output tokens; all experiments together ≈ $5.6 at those peak prices. Check the price
  page before a run: `llm-check --price-in 0.30 --price-cache-hit 0.006 --price-out 1.20`
  (`scripts/run_live.sh --prices 0.30 0.006 1.20` passes the same values).
* Alternative provider: `CAMPUSFLEX_PROVIDER=anthropic` (or `--provider anthropic`) uses Claude
  (`claude-opus-5-5`) via the `anthropic` SDK with server-side JSON-schema output.
* Every LLM command accepts `--provider` and `--model`.
* DeepSeek V4 models (`deepseek-flash`, `deepseek-v4-pro`) think before answering by default, and
  the thinking tokens count toward `max_tokens` and latency: on 2026-11-10 `llm-check` had L1, L2,
  L4 and L5 truncated (`max_tokens`) and L2 at a 16 s median. CampusFlex therefore sends
  `thinking: {"type": "disabled"}` (`CAMPUSFLEX_DEEPSEEK_THINKING=1` turns it on; the setting is
  part of the prompt hashes). With it off, every role answered 3/3 usably, L2 at a 3 s median.
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
`notices_frozen=false`. When a teammate rewrites the held-out notices (`data/notice_corpus.json`, `heldout`), re-run
`python -m campusflex freeze --force` **before** any held-out run.

## Demo

```bash
python -m venv --system-site-packages .venv && .venv/bin/pip install -e ".[iot,dashboard]"   # once
conda create -y -p ~/mosquitto-env -c conda-forge mosquitto=2.0.20                          # once (see Install)
scripts/demo.sh            # writes only to demo_out/; Ctrl-C stops the dashboard and the broker it started
```

1. **IoT** – one evening of the planner over MQTT (the script starts Mosquitto from
   `~/mosquitto-env` when nothing listens on 1883; `device layer: bus:mqtt` confirms the broker
   was used, a fallback to the in-process bus is printed as such).
2. **LLM** – the live E1 C/D fault episode (arm C, seed 10) rebuilt from the recorded
   `deepseek-flash` replies in `results/e1_CD_traces.jsonl.gz` with a client that cannot reach the
   network (`python -m campusflex.dashboard.replay_llm_demo`): every L2 prompt hashes to a
   recorded reply and each step must commit the same plan as the live run, otherwise nothing is
   written. With `DEEPSEEK_API_KEY` set the script also makes one live call per role
   (`llm-check --repeats 1`, about a cent).
3. **Smart devices** – the IoT · AI · DSA extension: one line per arm and the learned slot
   personas are printed, the full JSON goes to `demo_out/smart.log`.
4. **Dashboard** – Streamlit on `127.0.0.1:8501`, two pages. *Decision trace*: the replayed LLM
   episode (L2 plans in the ranking, the model's transcripts, a provenance banner); switch to
   `demo_out/episode_mqtt.json` in the sidebar. *Smart devices*: the extension's results from
   `demo_out/smart` (see below). From a laptop: `ssh -L 8501:127.0.0.1:8501 <user>@<server>`.

`results/demo_episode_llm.json` is the same verified replay, kept for the dashboard without
running the script.

## Smart device extension (IoT · AI · DSA)

`campusflex/smart/` runs every device of every room on top of the core: the core planner
still decides which rooms are used and who sits where; the extension decides how the
devices in those rooms run. It does not touch E1–E6, the freeze record or the ledger.

```bash
python -m campusflex smart            # 20 training + 10 evaluation evenings, ~5 s → results/smart/
python -m campusflex smart --profiles results/smart/profiles.json   # start from the remembered profiles
python -m campusflex smart --llm      # persona text written by the LLM (numbers checked); needs a key
```

* **IoT – per-device dispatch.** Six devices per room: AC (setpoint 22–28 °C + fan 0–3; fan
  speed limits cooling to 75/90/100 %), dimmable light (0–100 %), fresh air (4/7/10 L/s per
  seat, 70 % heat recovery), curtain (0–100 % open), projector, sockets. Commands go over MQTT
  topics `campus/<room>/<device>/cmd|ack` (in-process bus by default) with an idempotency key,
  per-device sequence numbers and retries; the firmware validates ranges, refuses cooling on
  an offline AC and older commands. Sensors report on `campus/<room>/sensors` every 2 minutes.
  A manual adjustment (wall panel, also over the bus) holds that device for 30 minutes, until
  the hourly session ends, or until the room has been empty for 10 minutes.
* **AI – usage profiles per room and hourly slot.** Learned from manual adjustments (weight 1)
  and from settings left untouched through an occupied slot (weight 0.2): preferred setpoint
  and fan, desk illuminance, curtain position, projector use. Room estimates are shrunk toward
  the slot's campus-wide estimate (so a rarely used room gets the slot's habits), evidence
  fades 3 % per evening, and the store persists as `results/smart/profiles.json`. Each slot
  gets a persona (e.g. "19:00-20:00: lecture (projector on), dim for the screen, prefers
  cool") in `results/smart/personas.md`. Within a slot the scheduler also keeps the latest
  manual choice instead of reverting it.
* **DSA – sensor-driven fine control.** Temperature, CO₂, illuminance, people counter, PIR and
  one power meter per device, with noise, dropouts and injected faults. Each reading is judged
  against an independent reference – CO₂ against an open-loop mass balance, temperature
  against the thermostat (while the AC holds its setpoint), illuminance against dimming +
  estimated daylight – to drop spikes and flag stuck or drifting sensors (the model then
  replaces the sensor); accepted readings are Kalman-filtered. A thermal model ΔT = a·q_known +
  b·(T_out − T) is identified online across evenings (R = a/b, C = Δt/a). The scheduler uses
  it for the pre-cooling lead, raises the fan when the AC cannot hold the setpoint (one level
  above a fan speed people accepted), stops the AC shortly before a room empties, dims for
  daylight, closes the curtain against screen glare, ventilates on predicted CO₂ (target 900
  ppm, hysteresis) and cuts projector/socket standby in empty rooms.

Results (`results/smart/summary.json`, also in `results/tables.md`; simulated occupants with
synthetic habits, 10 evaluation evenings, paired bootstrap): against the occupancy rule,
`smart` cuts manual adjustments from 54.8 to 21.9 per evening (−32.9 [−37.8, −27.6]) and
light shortfall by 1.7 points, with the same energy (−0.1 kWh [−1.2, 1.2]) and no significant
thermal difference. Profiles alone add −5.8 adjustments per evening over DSA alone; DSA alone
adds −15.8 points of thermal discomfort and −3.1 points of CO₂ above 1000 ppm over profiles on
raw readings (at +1.9 kWh, mostly pre-cooling). The static building uses 1.6 kWh less only by
under-ventilating (CO₂ above 1000 ppm for 78 % of occupied time). Learned profiles: setpoint
error 0.17 °C, illuminance error 30 lux; 13 of 16 stuck/drift faults detected (2 false alarms,
none in 20 fault-free evenings); identified R and C within 23 % and 11 % of Table 6 (median).
The extension saves no energy against a good occupancy rule: the AC dominates and runs near
its Table 6 capacity in full rooms.

The dashboard's *Smart devices* page (`campusflex/dashboard/smart_page.py`) shows these
results in four tabs: the evaluation table, energy by device and the paired comparisons; the
learning curve, the slot personas and what the memory recommends for every room and slot
(and whether from the room's own evidence, the slot pool or the default); fault detection and
thermal identification; and one evaluation evening traced every 2 minutes in every arm
(`results/smart/evening.json`, `--trace-day`, default seed 100 with a stuck temperature sensor
and a drifting CO₂ sensor in A101): true / sensor / estimated temperature, CO₂ and
illuminance, every device's state, each command by user or scheduler, the fault flags, and
all rooms at a chosen time. Recording the trace does not change the evening.

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
| Smart device extension (not in the PDF): devices, IoT, profiles, sensors, DSA, schedulers | `campusflex/smart/`, `campusflex/experiments/smart.py` |
| Table 14 fixtures | `tests/e5_fixtures/` (authored by `campusflex/experiments/e5_fixtures.py`) |
| §5.4 team records | `docs/reviews.md`, `docs/disagreements.md`, `docs/integration_log.md`, `docs/contrib.md` (templates) |

## Tests

`pytest` → 208 passed (`results/test_results.txt`, run with paho-mqtt, streamlit and a local
Mosquitto 2.0.20 broker; the MQTT tests also pass on amqtt 0.12.1). Without paho-mqtt/a broker
the three real-broker tests are skipped, and without streamlit the dashboard tests
(`tests/test_dashboard.py`, Streamlit AppTest: every step of an episode renders, each metric is
the logged value, on both pages) are skipped: 200 passed, 4 skipped. `tests/test_core.py` holds the 13 core invariants; the others cover the plan
language/validator/graph, IoT layer (incl. drift, partial acks, out-of-order commands), L1/L3/L4
roles and their grounding, L5, the DeepSeek client (`tests/test_llm_client.py`, fake transport,
no network), the experiment protocol checks incl. the prompt/notice freeze (`tests/test_experiments.py`),
the readiness check (`tests/test_llm_check.py`), the budgeted call path (`tests/test_llm_call.py`:
step deadline, tokens of abandoned calls, client exceptions) the guards of `scripts/run_live.sh` (`tests/test_scripts.py`, no LLM call) and the smart device
extension (`tests/test_smart.py`: firmware and bus faults, room physics, fault detection, thermal
identification, profile learning and persistence, scheduler rules, a small end-to-end run).
