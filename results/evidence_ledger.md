# Evidence ledger (Table 16) – statuses computed from files on disk

| # | Item | LLM | Artifact | Status | Evidence |
|---|---|---|---|---|---|
| 0 | v1 core, 160 episodes, 13 tests | No | results/v1/summary.json | Executed | tests: 193 passed |
| 1 | Predicted-vs-realized V log; Ω term | No | results/vgap_dev.csv | Executed |  |
| 2 | MQTT device layer and its tests | No | tests/test_iot.py | Executed | test_iot.py passed in results/test_results.txt, incl. 3 test(s) through a real MQTT broker |
| 3 | E5 scripted suite (120 fixtures) | No | results/e5_scripted.json | Executed |  |
| 4 | E1 arms A and B on held-out seeds | No | results/e1_AB.json | Executed |  |
| 5 | E1 arms C and D; E5 live sub-test | Yes | results/e1_CD.json, results/e5_live.json | Not run | results/e1_CD.json: status not_run (Insufficient Balance); results/e5_live.json: status not_run (Insufficient Balance) |
| 6 | Joint trace T1 with counterfactuals | Yes | results/trace_T1.json | Not run | results/trace_T1.json: status not_run (Insufficient Balance) |
| 7 | E2, E3, E4 | Yes | results/e2.json, results/e3.json, results/e4.json | Partial | results/e2.json: arm(s) not run or invalid: llm_compiler (not_run); results/e3.json: arm(s) not run or invalid: llm (not_run); results/e4.json: arm(s) not run or invalid: llm (not_run) |
| 8 | E6 and L5 review report (stretch) | Yes | results/e6.json, docs/l5_report.md | Partial | missing: docs/l5_report.md; results/e6.json: arm(s) not run or invalid: L5 (not_run) |
| 9 | Review records, disagreement record, contribution sign-off | No | docs/reviews.md, docs/contrib.md | Not started | 0 dated review records; 0 distinct reviewers; 0 disagreement records; 0/5 sign-offs |

The report cites a result only when its artifact exists; items 1–4 need no API access.
