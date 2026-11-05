<!-- TEMPLATE: no real records yet. Replace the placeholder row; keep one row per review. -->
# Review records (PDF Section 5.4, Table 19)

Every cross-review is a pull-request review or a short written check, logged here **on the day it
happens**. Before submission there must be at least ten records, and each member must appear at
least twice as reviewer. A record that found nothing is still recorded.

Planned checks (Table 18): M3 checks M1 compiler physics; M4 checks M1 attribution; M5 checks M2
data flow; M1 checks M2 schemas; M4 reviews M3 scenario assumptions and the L5 acceptance rule;
M1 reviews M4 LLM comparison; M2 reviews M4 notice labels; M2 reviews M5 tool contracts; M3
reviews M5 L5-report numbers.

Format: `Date | Reviewer → owner | Artifact | Finding | Resolution | Status`. Dates must be real
ISO dates (`2026-11-10`); `results/evidence_ledger.md` counts only rows that start with one.

| Date | Reviewer → owner | Artifact | Finding | Resolution | Status |
|---|---|---|---|---|---|
| [YYYY-MM-DD] | [M3 → M1] | [PR or commit id] | [what was checked and what was found] | [change made, or why none] | [closed] |
