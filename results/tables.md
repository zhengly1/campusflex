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

| Scenario | Mismatch | Arm | n | Score | Dev. °C | Moves | kWh | Unserved p-min |
|---|---|---|---|---|---|---|---|---|
| double_fault | no | A | 10 | 224.499 | 0.039 | 108.100 | 35.467 | 0.000 |
| double_fault | no | B | 10 | 224.559 | 0.039 | 108.100 | 35.527 | 0.000 |
| double_fault | no | C | 30 | 156.008 | 0.024 | 100.800 | 36.049 | 0.000 |
| double_fault | no | D | 30 | 143.004 | 0.021 | 100.833 | 36.183 | 0.000 |
| double_fault | yes | A | 10 | 243.578 | 0.043 | 104.500 | 35.870 | 0.000 |
| double_fault | yes | B | 10 | 243.578 | 0.043 | 104.500 | 35.870 | 0.000 |
| double_fault | yes | C | 30 | 168.395 | 0.027 | 100.800 | 36.318 | 0.000 |
| double_fault | yes | D | 30 | 168.395 | 0.027 | 100.800 | 36.318 | 0.000 |
| fault | no | A | 10 | 187.144 | 0.031 | 101.200 | 37.043 | 0.000 |
| fault | no | B | 10 | 187.144 | 0.031 | 101.200 | 37.043 | 0.000 |
| fault | no | C | 30 | 50.499 | 0.001 | 101.033 | 38.131 | 0.000 |
| fault | no | D | 30 | 50.355 | 0.001 | 100.867 | 38.261 | 0.000 |
| fault | yes | A | 10 | 165.906 | 0.026 | 111.600 | 37.607 | 0.000 |
| fault | yes | B | 10 | 165.906 | 0.026 | 111.600 | 37.607 | 0.000 |
| fault | yes | C | 30 | 50.997 | 0.001 | 102.533 | 38.098 | 0.000 |
| fault | yes | D | 30 | 53.831 | 0.002 | 102.533 | 38.160 | 0.000 |
| fault_surge | no | A | 10 | 126.303 | 0.017 | 105.400 | 38.872 | 0.000 |
| fault_surge | no | B | 10 | 126.303 | 0.017 | 105.400 | 38.872 | 0.000 |
| fault_surge | no | C | 30 | 52.143 | 0.001 | 108.500 | 38.715 | 0.000 |
| fault_surge | no | D | 30 | 49.457 | 0.001 | 108.167 | 38.500 | 0.000 |
| fault_surge | yes | A | 10 | 149.926 | 0.022 | 109.400 | 38.709 | 0.000 |
| fault_surge | yes | B | 10 | 150.389 | 0.022 | 111.100 | 39.054 | 0.000 |
| fault_surge | yes | C | 30 | 54.074 | 0.002 | 110.600 | 39.054 | 0.000 |
| fault_surge | yes | D | 30 | 56.113 | 0.002 | 111.900 | 39.246 | 0.000 |
| heat | no | A | 10 | 39.443 | 0.000 | 28.400 | 35.077 | 0.000 |
| heat | no | B | 10 | 39.443 | 0.000 | 28.400 | 35.077 | 0.000 |
| heat | no | C | 30 | 39.443 | 0.000 | 28.400 | 35.077 | 0.000 |
| heat | no | D | 30 | 39.443 | 0.000 | 28.400 | 35.077 | 0.000 |
| heat | yes | A | 10 | 39.008 | 0.000 | 27.300 | 34.753 | 0.000 |
| heat | yes | B | 10 | 39.008 | 0.000 | 27.300 | 34.753 | 0.000 |
| heat | yes | C | 30 | 39.008 | 0.000 | 27.300 | 34.753 | 0.000 |
| heat | yes | D | 30 | 39.008 | 0.000 | 27.300 | 34.753 | 0.000 |

**LLM plan statistics** (an arm with fallback or failed-call rate > 0.5 is not an LLM result and is not compared)

| Arm | episodes | L2 calls | failed calls | fallback rate | valid proposal rate | LLM plan selection rate | status |
|---|---|---|---|---|---|---|---|
| C | 240 | 2558 | 23 | 0.0090 | 0.8757 | 0.0724 | run |
| D | 240 | 2552 | 38 | 0.0149 | 0.8836 | 0.0731 | run |

**Paired comparisons (arm − reference; negative = better; LLM repeats averaged per pair; Eq. 4 order: unserved person-minutes first, then score)**

| Scenario | Comparison | pairs | score diff | lo | hi | excludes 0 | unserved diff | decided by | arm better |
|---|---|---|---|---|---|---|---|---|---|
| double_fault | C − B | 20 | -71.867 | -111.951 | -35.162 | yes | 0.000 | score | yes |
| fault | C − B | 20 | -125.777 | -160.254 | -95.501 | yes | 0.000 | score | yes |
| fault_surge | C − B | 20 | -85.238 | -114.281 | -60.410 | yes | 0.000 | score | yes |
| heat | C − B | 20 | 0.000 | -0.000 | 0.000 | no | 0.000 | score | no |
| (all) | C − B | 80 | -70.720 | -88.529 | -53.912 | yes | 0.000 | score | yes |
| double_fault | D − B | 20 | -78.369 | -116.791 | -42.916 | yes | 0.000 | score | yes |
| fault | D − B | 20 | -124.431 | -158.908 | -93.936 | yes | 0.000 | score | yes |
| fault_surge | D − B | 20 | -85.561 | -114.124 | -61.552 | yes | 0.000 | score | yes |
| heat | D − B | 20 | 0.000 | -0.000 | 0.000 | no | 0.000 | score | no |
| (all) | D − B | 80 | -72.091 | -89.473 | -55.472 | yes | 0.000 | score | yes |
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
| llm_compiler | run | deepseek-flash | 40 | 0 | 0.0000 | 1.0000 |

| Arm | n | valid_share | field_accuracy | hard_constraint_omissions | weight_levels_correct_share | weight_direction_correct_share_nondefault | clarification_correct_share | max_questions | zero omissions | direction ≥ 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| structured_form | 40 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0.943 | 1 | yes | yes |
| llm_compiler | 40 | 1.000 | 0.990 | 0 | 0.925 | 1.000 | 1.000 | 2 | yes | yes |

**Field accuracy by request category**

| Arm | ambiguous_preferences | changing_demand | equipment_faults | infeasible_constraints | straightforward |
|---|---|---|---|---|---|
| structured_form | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| llm_compiler | 1.000 | 0.990 | 0.990 | 0.969 | 1.000 |

**Plan direction when compiled weight levels are applied**

| Arm | scenario | seeds | checks | intended share |
|---|---|---|---|---|
| structured_form | fault | [0, 1, 2] | 11 | 0.727 |
| llm_compiler | fault | [0, 1, 2] | 11 | 0.727 |

## E3 – notice interpretation (constant / keyword / LLM / oracle)

Seeds: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]; scenarios: ["notice_surge", "notice_maintenance", "notice_drop", "notice_fault"]; planner: masks only (arm A), constant weather; LLM repeats: 3, caching: disabled.

| Arm | status | reason / model | LLM calls | failed calls | fallback rate | valid rate |
|---|---|---|---|---|---|---|
| constant | run | – | – | – | – | – |
| keyword | run | – | – | – | – | – |
| oracle | run | – | – | – | – | – |
| llm | run | – | 315 | 0 | 0.0000 | 0.3545 |

**Overall means per arm**

| Arm | n | unserved_person_min | moves | forecast_mae_30min | kwh | dev_c | score |
|---|---|---|---|---|---|---|---|
| constant | 40 | 185.000 | 19.925 | 17.244 | 33.371 | 0.004 | 57.338 |
| keyword | 40 | 185.000 | 19.425 | 10.536 | 33.294 | 0.002 | 44.903 |
| llm | 120 | 185.000 | 19.425 | 10.932 | 33.319 | 0.002 | 44.936 |
| oracle | 40 | 185.000 | 19.325 | 5.589 | 33.431 | 0.001 | 41.964 |

**Per scenario**

| Scenario | Arm | n | unserved_person_min | moves | forecast_mae_30min | kwh | dev_c | score |
|---|---|---|---|---|---|---|---|---|
| notice_drop | constant | 10 | 0.000 | 7.300 | 15.957 | 30.555 | 0.001 | 34.455 |
| notice_drop | keyword | 10 | 0.000 | 7.300 | 12.124 | 30.555 | 0.001 | 34.455 |
| notice_drop | llm | 30 | 0.000 | 7.300 | 12.624 | 30.555 | 0.001 | 34.455 |
| notice_drop | oracle | 10 | 0.000 | 7.300 | 10.748 | 30.555 | 0.001 | 34.455 |
| notice_fault | constant | 10 | 0.000 | 72.000 | 14.381 | 32.710 | 0.006 | 68.800 |
| notice_fault | keyword | 10 | 0.000 | 70.000 | 6.381 | 31.922 | 0.007 | 69.026 |
| notice_fault | llm | 30 | 0.000 | 70.000 | 6.914 | 31.932 | 0.007 | 69.041 |
| notice_fault | oracle | 10 | 0.000 | 70.000 | 3.638 | 32.591 | 0.005 | 61.411 |
| notice_maintenance | constant | 10 | 740.000 | 0.000 | 25.743 | 40.370 | 0.008 | 86.836 |
| notice_maintenance | keyword | 10 | 740.000 | 0.000 | 18.543 | 41.200 | 0.000 | 42.324 |
| notice_maintenance | llm | 30 | 740.000 | 0.000 | 18.429 | 41.200 | 0.000 | 42.324 |
| notice_maintenance | oracle | 10 | 740.000 | 0.000 | 5.171 | 41.200 | 0.000 | 42.324 |
| notice_surge | constant | 10 | 0.000 | 0.400 | 12.895 | 29.850 | 0.002 | 39.262 |
| notice_surge | keyword | 10 | 0.000 | 0.400 | 5.095 | 29.498 | 0.001 | 33.805 |
| notice_surge | llm | 30 | 0.000 | 0.400 | 5.762 | 29.590 | 0.001 | 33.922 |
| notice_surge | oracle | 10 | 0.000 | 0.000 | 2.800 | 29.378 | 0.000 | 29.666 |

**Paired bootstrap vs constant forecast (arm − constant; repeats averaged per scenario/seed)**

| Comparison | pairs | mean diff | lo | hi | excludes 0 |
|---|---|---|---|---|---|
| keyword_vs_constant/unserved_person_min | 40 | 0.000 | 0.000 | 0.000 | no |
| keyword_vs_constant/moves | 40 | -0.500 | -1.100 | -0.050 | yes |
| keyword_vs_constant/score | 40 | -12.436 | -19.258 | -6.472 | yes |
| llm_vs_constant/unserved_person_min | 40 | 0.000 | 0.000 | 0.000 | no |
| llm_vs_constant/moves | 40 | -0.500 | -1.100 | -0.050 | yes |
| llm_vs_constant/score | 40 | -12.403 | -19.228 | -6.453 | yes |
| oracle_vs_constant/unserved_person_min | 40 | 0.000 | 0.000 | 0.000 | no |
| oracle_vs_constant/moves | 40 | -0.600 | -1.250 | -0.100 | yes |
| oracle_vs_constant/score | 40 | -15.374 | -22.012 | -9.472 | yes |

**Gap closed by the LLM: (keyword − LLM) / (keyword − oracle); success = at least half on every metric**

| Metric | keyword | LLM | oracle | closed | note |
|---|---|---|---|---|---|
| unserved_person_min | 185.000 | 185.000 | 185.000 | – | no gap between keyword and oracle |
| moves | 19.425 | 19.425 | 19.325 | 0.000 | – |
| forecast_mae_30min | 10.536 | 10.932 | 5.589 | -0.080 | – |
| score | 44.903 | 44.936 | 41.964 | -0.011 | – |

Success (≥ 0.5 on every metric with a gap): no

Episode rows in file: 240.

## E4 – explanation faithfulness

Decisions from: fresh masks-only planner decisions; episodes: 16.

| Arm | status | reason / model | LLM calls | failed calls | fallback rate | valid rate |
|---|---|---|---|---|---|---|
| template | run | – | – | – | – | – |
| llm | run | deepseek-flash | 48 | 0 | 0.0000 | 1.0000 |

| Arm | n_texts | n_numbers | n_matched | n_flagged | share_matched | texts_fully_matched_share | displayed_unverified_numbers | success_100pct_matched_or_flagged |
|---|---|---|---|---|---|---|---|---|
| template | 48 | 480 | 480 | 0 | 1.000 | 1.000 | 0 | yes |
| llm | 48 | 407 | 407 | 0 | 1.000 | 1.000 | 0 | yes |

## E5 – scripted adversarial suite (no live LLM)

Meta: experiment=E5 scripted, fixture_dir=tests/e5_fixtures, fixture_state=fault seed 0, 20:00, seconds=9.200, llm=none (scripted adversarial proposer)

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

Meta: experiment=E5 live sub-test, prompt_hashes={"L1": "fcc087320195", "L2": "e1a7d2b3f1e8", "L3": "40fa23fd2d33", "L4": "d10..., model=deepseek-flash, prompts_frozen=yes, variants=["deployed", "model_only"], min_share_evaluated_per_variant=0.900

| field | value |
|---|---|
| summary.note | The instruction-screen patterns (forecast.looks_like_instruction) were written by the same author as the injected fixtures, so the 'deployed' result is an optimistic estimate of robustness to unseen injections; 'model_only' measures the model behind the data-field wrapper and grounding alone. |

Episode/record rows in file: 80.

## E6 – L5 review vs random search (seeds 20–29)

| Arm | status | source / reason | config | LLM calls | failed calls | fallback rate |
|---|---|---|---|---|---|---|
| pre_review | run | results/frozen_config.json | de54345ef023 | – | – | – |
| L5 | run | /data/lianyuzheng/campusflex/results/l5_config.json | de54345ef023 | 6 | 0 | 0.000 |
| random | run | /data/lianyuzheng/campusflex/results/l5_random_config.json | de54345ef023 | – | – | – |

| Arm | Unserved p-min | Score | kWh | Dev. °C | Moves |
|---|---|---|---|---|---|
| pre_review | 67.500 | 91.544 | 33.111 | 0.012 | 43.320 |
| L5 | 67.500 | 91.544 | 33.111 | 0.012 | 43.320 |
| random | 67.500 | 91.544 | 33.111 | 0.012 | 43.320 |

Paired comparisons in Eq. 4 order: unserved person-minutes (U) first, then score.

| Comparison | pairs | U diff | U lo | U hi | score diff | lo | hi | excludes 0 | decided by |
|---|---|---|---|---|---|---|---|---|---|
| L5_minus_pre_review | 100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | no | score |
| random_minus_pre_review | 100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | no | score |
| L5_minus_random | 100 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | no | score |

| verdict field | value |
|---|---|
| success_criterion | L5 beats pre-review with interval excluding zero and is not worse than random search (Eq. 4 order: unserved first, then score) |
| L5_beats_pre_review | no |
| L5_not_worse_than_random | yes |
| decided_by | {"L5_vs_pre_review": "score", "L5_vs_random": "score"} |
| success | no |
| falsification | L5 ≈ random search: the gain comes from spending more evaluation budget on development seeds, not from reading logs. |

| Review | proposals | accepted | review acceptance rate | protocol |
|---|---|---|---|---|
| llm | 6 | 0 | 0.000 | full |
| random | 18 | 0 | 0.000 | full |

## T1 – integrated case trace

Meta: experiment=trace T1, config_hash=de54345ef023, frozen=yes, scenario_hash=0a1d2cc43ca0, scenario_frozen=yes, notices_hash=33c23c12f015, notices_frozen=yes, prompt_hashes={"L1": "fcc087320195", "L2": "e1a7d2b3f1e8", "L3": "40fa23fd2d33", "L4": "d10..., model=deepseek-flash, prompts_frozen=yes, scenario=notice_fault, seed=10, device_layer=bus:inprocess, device_fallback=yes, device_fallback_reason=paho-mqtt is not installed (pip install paho-mqtt), planner=T1-D, forecaster=notice:llm

| field | value |
|---|---|
| complete | yes |
| statement | committed action traced to notice, IoT message, staged plan and simulation; removing any one input changes it |

**records**

| step | state_version | inputs | ranking | committed | counterfactuals |
|---|---|---|---|---|---|
| 18:00 | 1 | {"notice": {"id": "N1", "l3_delta": null, "confidence": null}, "iot": {"msg_i... | [{"plan_id": "m:A101B201", "source": "mask", "score_J": 6.5128}, {"plan_id": ... | {"plan_id": "m:A101B201", "source": "mask", "ack": true} | {} |
| 18:10 | 2 | {"notice": {"id": "N1", "l3_delta": null, "confidence": null}, "iot": {"msg_i... | [{"plan_id": "m:A101B201", "source": "mask", "score_J": 5.6597}, {"plan_id": ... | {"plan_id": "m:A101B201", "source": "mask", "ack": true} | {} |
| 18:20 | 3 | {"notice": {"id": "N1", "l3_delta": null, "confidence": null}, "iot": {"msg_i... | [{"plan_id": "m:A101B201", "source": "mask", "score_J": 5.3749}, {"plan_id": ... | {"plan_id": "m:A101B201", "source": "mask", "ack": true} | {} |
| 18:30 | 4 | {"notice": {"id": "N1", "l3_delta": null, "confidence": null}, "iot": {"msg_i... | [{"plan_id": "m:A101B201", "source": "mask", "score_J": 5.3363}, {"plan_id": ... | {"plan_id": "m:A101B201", "source": "mask", "ack": true} | {} |
| 18:40 | 5 | {"notice": {"id": "N1", "l3_delta": null, "confidence": null}, "iot": {"msg_i... | [{"plan_id": "m:A101B201", "source": "mask", "score_J": 5.2977}, {"plan_id": ... | {"plan_id": "m:A101B201", "source": "mask", "ack": true} | {} |
| 18:50 | 6 | {"notice": {"id": "N1", "l3_delta": null, "confidence": null}, "iot": {"msg_i... | [{"plan_id": "m:A101B201", "source": "mask", "score_J": 5.2591}, {"plan_id": ... | {"plan_id": "m:A101B201", "source": "mask", "ack": true} | {} |
| 19:00 | 7 | {"notice": {"id": "N1", "l3_delta": null, "confidence": null}, "iot": {"msg_i... | [{"plan_id": "m:A101B201", "source": "mask", "score_J": 5.2205}, {"plan_id": ... | {"plan_id": "m:A101B201", "source": "mask", "ack": true} | {} |
| 19:10 | 8 | {"notice": {"id": "N1", "l3_delta": null, "confidence": null}, "iot": {"msg_i... | [{"plan_id": "m:A101B201", "source": "mask", "score_J": 5.182}, {"plan_id": "... | {"plan_id": "m:A101B201", "source": "mask", "ack": true} | {} |
| 19:20 | 9 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101B201B203", "source": "mask", "score_J": 7.5477}, {"plan_i... | {"plan_id": "m:A101B201B203", "source": "mask", "ack": true} | {"no_notice": "m:A101B201", "fault_msg_delayed_one_step": "m:A101B201B203", "... |
| 19:30 | 10 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101B201B203", "source": "mask", "score_J": 7.2151}, {"plan_i... | {"plan_id": "m:A101B201B203", "source": "mask", "ack": true} | {"no_notice": "m:A101B201B203", "fault_msg_delayed_one_step": "m:A101B201B203... |
| 19:40 | 11 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101B201B202B203", "source": "mask", "score_J": 13.0485}, {"p... | {"plan_id": "m:A101B201B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101B201B202B203", "fault_msg_delayed_one_step": "m:A101B201... |
| 19:50 | 12 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101B201B202B203", "source": "mask", "score_J": 8.9676}, {"pl... | {"plan_id": "m:A101B201B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101B201B202B203", "fault_msg_delayed_one_step": "m:A101B201... |
| 20:00 | 13 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "L2:P1", "source": "L2", "score_J": 15.9204}, {"plan_id": "L2:P2... | {"plan_id": "L2:P1", "source": "L2", "ack": true} | {"no_notice": "L2:P2", "fault_msg_delayed_one_step": "m:A101B201B202B203", "n... |
| 20:10 | 14 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 14.0542}, {"p... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |
| 20:20 | 15 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 8.9214}, {"pl... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |
| 20:30 | 16 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 8.3414}, {"pl... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |
| 20:40 | 17 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 7.8979}, {"pl... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |
| 20:50 | 18 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 7.6418}, {"pl... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |
| 21:00 | 19 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 7.3858}, {"pl... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |
| 21:10 | 20 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 6.0231}, {"pl... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |
| 21:20 | 21 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 4.6913}, {"pl... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |
| 21:30 | 22 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 3.3905}, {"pl... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |
| 21:40 | 23 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 2.0272}, {"pl... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |
| 21:50 | 24 | {"notice": {"id": "N1", "l3_delta": 60.0, "confidence": "medium"}, "iot": {"m... | [{"plan_id": "m:A101A102B202B203", "source": "mask", "score_J": 0.9981}, {"pl... | {"plan_id": "m:A101A102B202B203", "source": "mask", "ack": true} | {"no_notice": "m:A101A102B202B203", "fault_msg_delayed_one_step": "m:A101A102... |


## Smart device extension – per-device IoT dispatch, learned profiles, sensor control

20 training and 10 evaluation evenings (two injected sensor faults each); space plan from the core planner (arm A). Discomfort = share of occupied person-minutes.

| Arm | kWh | thermal % | light % | CO2>1000 % | glare % | manual adj./evening |
|---|---|---|---|---|---|---|
| static | 42.89 | 29.88 | 4.92 | 78.03 | 0.77 | 72.30 |
| rule | 44.54 | 24.15 | 6.00 | 0.00 | 0.76 | 54.80 |
| smart_dsa_only | 44.54 | 21.84 | 5.18 | 0.00 | 0.41 | 27.70 |
| smart_profiles_only | 42.58 | 38.90 | 4.09 | 3.08 | 0.56 | 27.60 |
| smart | 44.46 | 23.12 | 4.26 | 0.00 | 0.20 | 21.90 |

**Paired by evaluation evening (smart − other, 95 % bootstrap interval)**

| Comparison | metric | mean diff | lo | hi |
|---|---|---|---|---|
| smart_vs_rule | adjustments | -32.900 | -37.800 | -27.598 |
| smart_vs_rule | kwh | -0.075 | -1.167 | 1.218 |
| smart_vs_rule | thermal_discomfort_pct | -1.030 | -8.018 | 7.388 |
| smart_vs_rule | light_discomfort_pct | -1.739 | -2.749 | -0.709 |
| smart_vs_rule | co2_over_1000_pct | 0.000 | 0.000 | 0.000 |
| smart_vs_static | adjustments | -50.400 | -56.300 | -43.800 |
| smart_vs_static | kwh | 1.572 | 0.341 | 3.011 |
| smart_vs_static | thermal_discomfort_pct | -6.760 | -14.104 | 0.380 |
| smart_vs_static | light_discomfort_pct | -0.661 | -1.979 | 0.625 |
| smart_vs_static | co2_over_1000_pct | -78.031 | -83.560 | -73.153 |
| smart_vs_smart_dsa_only | adjustments | -5.800 | -8.600 | -3.100 |
| smart_vs_smart_dsa_only | kwh | -0.081 | -0.585 | 0.421 |
| smart_vs_smart_dsa_only | thermal_discomfort_pct | 1.287 | -4.645 | 8.935 |
| smart_vs_smart_dsa_only | light_discomfort_pct | -0.919 | -1.834 | -0.016 |
| smart_vs_smart_dsa_only | co2_over_1000_pct | 0.000 | 0.000 | 0.000 |
| smart_vs_smart_profiles_only | adjustments | -5.700 | -9.002 | -2.700 |
| smart_vs_smart_profiles_only | kwh | 1.888 | 0.415 | 3.546 |
| smart_vs_smart_profiles_only | thermal_discomfort_pct | -15.771 | -20.360 | -11.471 |
| smart_vs_smart_profiles_only | light_discomfort_pct | 0.178 | -0.945 | 1.214 |
| smart_vs_smart_profiles_only | co2_over_1000_pct | -3.085 | -3.747 | -2.514 |

Profiles vs latent habits (19/24 room-slots with own evidence): setpoint MAE 0.168 °C, lux MAE 29.663, curtain MAE 16.316 %, fan mismatch 0.368, projector mismatch 0.211.
Sensor faults (stuck/drift): 13/16 detected, 2 false alarms, mean delay 15.5 min. Identified thermal model vs Table 6: median relative error R 0.2294, C 0.114.

