# CampusFlex – Section 4 tables (regenerated from results/)

All values are simulated estimates from the synthetic six-room campus.

## Table 12 – v1 results of this implementation (means of episode-level metrics per policy)

| Policy | n | kWh | Service % | Dev. °C | Moves | Switches | Score |
|---|---|---|---|---|---|---|---|
| Fixed opening | 40 | 39.867 | 100.000 | 0.000 | 0.000 | 4.000 | 40.187 |
| Demand threshold | 40 | 36.467 | 100.000 | 0.000 | 0.000 | 5.250 | 36.887 |
| Simulation planner | 40 | 29.272 | 100.000 | 0.003 | 17.500 | 5.000 | 42.822 |
| No relocation cost | 40 | 28.919 | 100.000 | 0.003 | 29.600 | 5.000 | – |

Score is the full-evening Equation 4 without Ω under reference weights; – for the ablation.

**Paired energy saving of the planner (kWh, 95% bootstrap interval, paired by scenario and seed)**

| Comparison | pairs | saving kWh | lo | hi |
|---|---|---|---|---|
| planner vs Fixed opening | 40 | 10.595 | 9.943 | 11.197 |
| planner vs Demand threshold | 40 | 7.195 | 6.730 | 7.639 |

Relocation cost cuts moves by 40.9% relative to the ablated planner (17.50 vs 29.60 moves per episode).

**Scenario-level means (Figure 3 data)**

| Scenario | Policy | n | kWh | Dev. °C | Moves | Score |
|---|---|---|---|---|---|---|
| fault | fixed | 10 | 43.784 | 0.000 | 0.000 | 44.104 |
| fault | planner | 10 | 32.293 | 0.009 | 70.000 | 80.181 |
| fault | planner_nomove | 10 | 32.293 | 0.009 | 70.000 | 80.181 |
| fault | threshold | 10 | 40.564 | 0.000 | 0.000 | 41.044 |
| heat | fixed | 10 | 45.713 | 0.000 | 0.000 | 46.033 |
| heat | planner | 10 | 34.629 | 0.001 | 0.000 | 37.549 |
| heat | planner_nomove | 10 | 33.781 | 0.001 | 25.800 | 38.250 |
| heat | threshold | 10 | 41.805 | 0.000 | 0.000 | 42.285 |
| steady | fixed | 10 | 32.921 | 0.000 | 0.000 | 33.241 |
| steady | planner | 10 | 20.570 | 0.000 | 0.000 | 20.730 |
| steady | planner_nomove | 10 | 20.570 | 0.000 | 0.000 | 20.730 |
| steady | threshold | 10 | 28.833 | 0.000 | 0.000 | 29.073 |
| surge | fixed | 10 | 37.052 | 0.000 | 0.000 | 37.372 |
| surge | planner | 10 | 29.596 | 0.001 | 0.000 | 32.828 |
| surge | planner_nomove | 10 | 29.031 | 0.001 | 22.600 | 33.619 |
| surge | threshold | 10 | 34.667 | 0.000 | 0.000 | 35.147 |

## Ledger item 1 – predicted vs realized occupied deviation V (development seeds)

First committed interval and rollout horizon; gap = realized − predicted (°C·person-h), grouped by λΩ and model mismatch.

| scenario | mismatch | lambda_Omega | policy | steps | Σ pred_first_V | Σ real_first_V | mean first-interval gap | mean horizon gap |
|---|---|---|---|---|---|---|---|---|
| double_fault | False | 0.0 | B | 240 | 196.5385 | 196.5385 | 0.0000 | -0.2791 |
| double_fault | False | 4.0 | B | 240 | 212.4877 | 212.4877 | 0.0000 | 2.1096 |
| double_fault | True | 0.0 | B | 240 | 187.2518 | 186.9004 | -0.0015 | -0.2277 |
| double_fault | True | 4.0 | B | 240 | 186.2592 | 187.6617 | 0.0058 | 1.8116 |
| fault | False | 0.0 | B | 240 | 226.3845 | 226.3845 | 0.0000 | 1.1040 |
| fault | False | 4.0 | B | 240 | 200.3772 | 200.3772 | 0.0000 | 3.0661 |
| fault | True | 0.0 | B | 240 | 220.7362 | 219.0933 | -0.0068 | 0.9981 |
| fault | True | 4.0 | B | 240 | 193.7009 | 191.5232 | -0.0091 | 2.7406 |
| fault_surge | False | 0.0 | B | 240 | 206.3608 | 206.3608 | 0.0000 | 2.0862 |
| fault_surge | False | 4.0 | B | 240 | 108.1594 | 108.1594 | 0.0000 | 1.9007 |
| fault_surge | True | 0.0 | B | 240 | 168.2539 | 169.2958 | 0.0043 | 1.8856 |
| fault_surge | True | 4.0 | B | 240 | 97.1057 | 98.0144 | 0.0038 | 1.7521 |
| heat | False | 0.0 | B | 240 | 3.0503 | 3.0503 | 0.0000 | 0.0635 |
| heat | False | 4.0 | B | 240 | 3.0503 | 3.0503 | 0.0000 | 0.0635 |
| heat | True | 0.0 | B | 240 | 2.4664 | 2.7862 | 0.0013 | 0.0594 |
| heat | True | 4.0 | B | 240 | 2.4664 | 2.7862 | 0.0013 | 0.0594 |

**Ω ablation – full-evening score, λΩ = 0 minus declared λΩ (paired by scenario, seed, mismatch)**

| pairs | mean diff | lo | hi | excludes 0 |
|---|---|---|---|---|
| 80 | 19.436 | 9.140 | 29.501 | yes |

## E1 – staged plans (full-evening combined score, held-out seeds)

Arms C and D: **not run** – preflight failed: api_status_402: Error code: 402 - {'error': {'message': 'Insufficient Balance (request_id: 4cfdeae1-fc8a-402d-89ab-7bc693c7310c)', 'type': 'insufficient_balance_error', 'param': None, 'code': 'invalid_request_error'}}.

| Scenario | Mismatch | Arm | n | Score | Dev. °C | Moves | kWh | Unserved p-min |
|---|---|---|---|---|---|---|---|---|
| double_fault | no | A | 10 | 224.499 | 0.039 | 108.100 | 35.467 | 0.000 |
| double_fault | no | B | 10 | 224.559 | 0.039 | 108.100 | 35.527 | 0.000 |
| double_fault | yes | A | 10 | 243.578 | 0.043 | 104.500 | 35.870 | 0.000 |
| double_fault | yes | B | 10 | 243.578 | 0.043 | 104.500 | 35.870 | 0.000 |
| fault | no | A | 10 | 187.144 | 0.031 | 101.200 | 37.043 | 0.000 |
| fault | no | B | 10 | 187.144 | 0.031 | 101.200 | 37.043 | 0.000 |
| fault | yes | A | 10 | 165.906 | 0.026 | 111.600 | 37.607 | 0.000 |
| fault | yes | B | 10 | 165.906 | 0.026 | 111.600 | 37.607 | 0.000 |
| fault_surge | no | A | 10 | 126.303 | 0.017 | 105.400 | 38.872 | 0.000 |
| fault_surge | no | B | 10 | 126.303 | 0.017 | 105.400 | 38.872 | 0.000 |
| fault_surge | yes | A | 10 | 149.926 | 0.022 | 109.400 | 38.709 | 0.000 |
| fault_surge | yes | B | 10 | 150.389 | 0.022 | 111.100 | 39.054 | 0.000 |
| heat | no | A | 10 | 39.443 | 0.000 | 28.400 | 35.077 | 0.000 |
| heat | no | B | 10 | 39.443 | 0.000 | 28.400 | 35.077 | 0.000 |
| heat | yes | A | 10 | 39.008 | 0.000 | 27.300 | 34.753 | 0.000 |
| heat | yes | B | 10 | 39.008 | 0.000 | 27.300 | 34.753 | 0.000 |

**Paired comparisons (arm − reference; negative = better; LLM repeats averaged per pair; Eq. 4 order: unserved person-minutes first, then score)**

| Scenario | Comparison | pairs | score diff | lo | hi | excludes 0 | unserved diff | decided by | arm better |
|---|---|---|---|---|---|---|---|---|---|
| double_fault | B − A | 20 | 0.030 | 0.000 | 0.090 | no | 0.000 | score | no |
| fault | B − A | 20 | 0.000 | 0.000 | 0.000 | no | 0.000 | score | no |
| fault_surge | B − A | 20 | 0.232 | 0.000 | 0.695 | no | 0.000 | score | no |
| heat | B − A | 20 | 0.000 | 0.000 | 0.000 | no | 0.000 | score | no |
| (all) | B − A | 80 | 0.065 | 0.000 | 0.189 | no | 0.000 | score | no |

## E2 – intent compilation (L1 vs structured form)

Requests: 40.

| Arm | status | reason / model | LLM calls | failed calls | fallback rate | valid rate |
|---|---|---|---|---|---|---|
| structured_form | run | fields entered by hand from the gold labels (non-LLM control) | – | – | – | – |
| llm_compiler | not_run | LLM arm not requested (run with --llm) | – | – | – | – |

| Arm | n | valid_share | field_accuracy | hard_constraint_omissions | weight_levels_correct_share | weight_direction_correct_share_nondefault | clarification_correct_share | max_questions | zero omissions | direction ≥ 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| structured_form | 40 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0.943 | 1 | yes | yes |

**Field accuracy by request category**

| Arm | ambiguous_preferences | changing_demand | equipment_faults | infeasible_constraints | straightforward |
|---|---|---|---|---|---|
| structured_form | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Plan direction when compiled weight levels are applied**

| Arm | scenario | seeds | checks | intended share |
|---|---|---|---|---|
| structured_form | fault | [0, 1, 2] | 11 | 0.727 |

## E3 – notice interpretation (constant / keyword / LLM / oracle)

Seeds: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]; scenarios: ["notice_surge", "notice_maintenance", "notice_drop", "notice_fault"]; planner: masks only (arm A), constant weather; LLM repeats: –, caching: –.

| Arm | status | reason / model | LLM calls | failed calls | fallback rate | valid rate |
|---|---|---|---|---|---|---|
| constant | run | – | – | – | – | – |
| keyword | run | – | – | – | – | – |
| oracle | run | – | – | – | – | – |
| llm | not_run | LLM arm not requested (run with --llm) | – | – | – | – |

**Overall means per arm**

| Arm | n | unserved_person_min | moves | forecast_mae_30min | kwh | dev_c | score |
|---|---|---|---|---|---|---|---|
| constant | 40 | 185.000 | 19.925 | 17.244 | 33.371 | 0.004 | 57.338 |
| keyword | 40 | 185.000 | 19.425 | 10.536 | 33.294 | 0.002 | 44.903 |
| oracle | 40 | 185.000 | 19.325 | 5.589 | 33.431 | 0.001 | 41.964 |

**Per scenario**

| Scenario | Arm | n | unserved_person_min | moves | forecast_mae_30min | kwh | dev_c | score |
|---|---|---|---|---|---|---|---|---|
| notice_drop | constant | 10 | 0.000 | 7.300 | 15.957 | 30.555 | 0.001 | 34.455 |
| notice_drop | keyword | 10 | 0.000 | 7.300 | 12.124 | 30.555 | 0.001 | 34.455 |
| notice_drop | oracle | 10 | 0.000 | 7.300 | 10.748 | 30.555 | 0.001 | 34.455 |
| notice_fault | constant | 10 | 0.000 | 72.000 | 14.381 | 32.710 | 0.006 | 68.800 |
| notice_fault | keyword | 10 | 0.000 | 70.000 | 6.381 | 31.922 | 0.007 | 69.026 |
| notice_fault | oracle | 10 | 0.000 | 70.000 | 3.638 | 32.591 | 0.005 | 61.411 |
| notice_maintenance | constant | 10 | 740.000 | 0.000 | 25.743 | 40.370 | 0.008 | 86.836 |
| notice_maintenance | keyword | 10 | 740.000 | 0.000 | 18.543 | 41.200 | 0.000 | 42.324 |
| notice_maintenance | oracle | 10 | 740.000 | 0.000 | 5.171 | 41.200 | 0.000 | 42.324 |
| notice_surge | constant | 10 | 0.000 | 0.400 | 12.895 | 29.850 | 0.002 | 39.262 |
| notice_surge | keyword | 10 | 0.000 | 0.400 | 5.095 | 29.498 | 0.001 | 33.805 |
| notice_surge | oracle | 10 | 0.000 | 0.000 | 2.800 | 29.378 | 0.000 | 29.666 |

**Paired bootstrap vs constant forecast (arm − constant; repeats averaged per scenario/seed)**

| Comparison | pairs | mean diff | lo | hi | excludes 0 |
|---|---|---|---|---|---|
| keyword_vs_constant/unserved_person_min | 40 | 0.000 | 0.000 | 0.000 | no |
| keyword_vs_constant/moves | 40 | -0.500 | -1.100 | -0.050 | yes |
| keyword_vs_constant/score | 40 | -12.436 | -19.258 | -6.472 | yes |
| oracle_vs_constant/unserved_person_min | 40 | 0.000 | 0.000 | 0.000 | no |
| oracle_vs_constant/moves | 40 | -0.600 | -1.250 | -0.100 | yes |
| oracle_vs_constant/score | 40 | -15.374 | -22.012 | -9.472 | yes |

**Gap closed by the LLM: (keyword − LLM) / (keyword − oracle); success = at least half on every metric**

_not computed: LLM arm not_run – LLM arm not requested (run with --llm)_

Episode rows in file: 120.

## E4 – explanation faithfulness

Decisions from: fresh masks-only planner decisions; episodes: 16.

| Arm | status | reason / model | LLM calls | failed calls | fallback rate | valid rate |
|---|---|---|---|---|---|---|
| template | run | – | – | – | – | – |
| llm | not_run | LLM arm not requested (run with --llm) | – | – | – | – |

| Arm | n_texts | n_numbers | n_matched | n_flagged | share_matched | texts_fully_matched_share | displayed_unverified_numbers | success_100pct_matched_or_flagged |
|---|---|---|---|---|---|---|---|---|
| template | 48 | 480 | 480 | 0 | 1.000 | 1.000 | 0 | yes |

## E5 – scripted adversarial suite (no live LLM)

Meta: experiment=E5 scripted, fixture_dir=tests/e5_fixtures, fixture_state=fault seed 0, 20:00, seconds=9.300, llm=none (scripted adversarial proposer)

| field | value |
|---|---|
| summary.n_fixtures | 120 |
| summary.n_runs | 210 |
| summary.unvalidated_actions_executed | 0 |
| summary.share_correct_reason | 1.000 |
| summary.ledger_consistent_all | yes |
| summary.injection_attributed_divergence | 0 |
| summary.l2_fallback_same_as_masks_only | yes |
| summary.targets_met | yes |

Episode/record rows in file: 210.

## E5 – live injected notices

Meta: experiment=E5 live sub-test

| field | value |
|---|---|
| status | not_run |
| reason | preflight failed: api_status_402: Error code: 402 - {'error': {'message': 'Insufficient Balance (request_id: ce786b01-d2b3-4d32-8067-ca5b41b7df31)', 'type': 'insufficient_balance_error', 'param': None, 'code': 'invalid_request_error'}} |


## E6 – L5 review vs random search (seeds 20–29)

| Arm | status | source / reason | config | LLM calls | failed calls | fallback rate |
|---|---|---|---|---|---|---|
| pre_review | run | results/frozen_config.json | de54345ef023 | – | – | – |
| L5 | not_run | l5_config.json missing; run `review --kind llm` | – | – | – | – |
| random | run | /data/lianyuzheng/campusflex/results/l5_random_config.json | de54345ef023 | – | – | – |

| Arm | Unserved p-min | Score | kWh | Dev. °C | Moves |
|---|---|---|---|---|---|
| pre_review | 67.500 | 91.544 | 33.111 | 0.012 | 43.320 |
| random | 67.500 | 91.544 | 33.111 | 0.012 | 43.320 |

Paired comparisons in Eq. 4 order: unserved person-minutes (U) first, then score.

| Comparison | pairs | U diff | U lo | U hi | score diff | lo | hi | excludes 0 | decided by |
|---|---|---|---|---|---|---|---|---|---|
| random_minus_pre_review | 100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | no | score |
| L5_minus_pre_review | – | – | – | – | – | – | – | not evaluable | – |
| L5_minus_random | – | – | – | – | – | – | – | not evaluable | – |

| verdict field | value |
|---|---|
| success_criterion | L5 beats pre-review with interval excluding zero and is not worse than random search (Eq. 4 order: unserved first, then score) |
| result | not evaluable (L5 or random-search arm not run or invalid) |

| Review | proposals | accepted | review acceptance rate | protocol |
|---|---|---|---|---|
| random | 18 | 0 | 0.000 | full |

## T1 – integrated case trace

Meta: experiment=trace T1

| field | value |
|---|---|
| status | not_run |
| reason | preflight failed: api_status_402: Error code: 402 - {'error': {'message': 'Insufficient Balance (request_id: ebc43386-5359-452a-b3a5-3a963f2c984b)', 'type': 'insufficient_balance_error', 'param': None, 'code': 'invalid_request_error'}} |


