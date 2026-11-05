<!-- TEMPLATE: shares are planned, not confirmed. -->
# Contribution confirmation and submission checklist (PDF Section 5.4)

Final shares combine three sources: (1) authorship of merged pull requests per module, (2) the
review records in `docs/reviews.md`, and (3) a peer assessment in which each member estimates every
other member's share. The team agrees final percentages that sum to 100%, and all five members sign
off below.

## Peer assessment (each row: assessor's estimate of every member's share, %)

| Assessor | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| M1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| M2 | [ ] | [ ] | [ ] | [ ] | [ ] |
| M3 | [ ] | [ ] | [ ] | [ ] | [ ] |
| M4 | [ ] | [ ] | [ ] | [ ] | [ ] |
| M5 | [ ] | [ ] | [ ] | [ ] | [ ] |

## Agreed shares and sign-off

Planned allocation is 20% each (cover table). Replace with the agreed values; the column must sum
to 100%. Set *Signature* to `signed` and add the real date when each member signs.

| Member | Name and student ID | Module | Merged PRs | Reviews given | Agreed share | Signature | Date |
|---|---|---|---|---|---|---|---|
| M1 | [name, ID] | L2 proposer, plan language, LangGraph graph | [ ] | [ ] | [20%] | pending | [YYYY-MM-DD] |
| M2 | [name, ID] | MQTT device layer, notice corpus, L3 | [ ] | [ ] | [20%] | pending | [YYYY-MM-DD] |
| M3 | [name, ID] | Thermal model, plan compiler, Ω term, L5 | [ ] | [ ] | [20%] | pending | [YYYY-MM-DD] |
| M4 | [name, ID] | Baselines, random search, E1–E6, statistics | [ ] | [ ] | [20%] | pending | [YYYY-MM-DD] |
| M5 | [name, ID] | L1, L4, faithfulness checker, dashboard | [ ] | [ ] | [20%] | pending | [YYYY-MM-DD] |

## Submission checklist

- [ ] Group ID, submission date, member names and student IDs on the cover
- [ ] Confirmed shares (table above, sums to 100%, five sign-offs)
- [ ] Review-record appendix (`docs/reviews.md`: ≥10 records, every member ≥2× as reviewer)
- [ ] Disagreement record (`docs/disagreements.md`)
- [ ] Integration log (`docs/integration_log.md`)
- [ ] Evidence ledger statuses updated from the repository (`python -m campusflex.experiments.evidence_ledger`)
- [ ] Section 4 tables regenerated from `results/` (`python -m campusflex.experiments.tables`)
