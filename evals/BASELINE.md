# Eval baseline

The committed baseline that `make eval` gates against. Update it only on an
intended, reviewed improvement (`python -m evals.run --update`).

| Metric | Value |
|---|---|
| Agent | prebuilt `create_agent` (gpt-4o) |
| Judge | openevals correctness (gpt-4o-mini) |
| Accuracy | **58.3%** (7 / 12) |
| Total tool calls | **59** |

Per-case (representative):

| Case | Correct | Notes |
|---|---|---|
| ex1 BlueHarbor proof plan | no | right customer; proof-plan specifics incomplete |
| ex2 Verdant Bay patch/rollback | no | window right; exact rollback step incomplete |
| ex3 MapleHarvest transform/workshop | no | mappings right; workshop output vague |
| ex4 Aureum SCIM fix | no | fields right; Jin's fix only half captured |
| ex5 defection risk / milestone | no | entity unstable run-to-run; milestone vague |
| ex6 NA-West taxonomy vs duplicate | yes | groups largely correct |
| ex7 Canada approval-bypass | yes | pattern + names correct |
| auth count customer calls | yes | 50 |
| auth BlueHarbor region | yes | North America West |
| auth products | yes | all four |
| auth cheap tactical competitor | yes | NoiseGuard |
| auth not-in-data | yes | honest refusal |

The misses are concentrated in the detail-heavy example queries (ex1–ex5): the
prebuilt agent answers from search snippets rather than the full artifact. The
grounding/citation graph is expected to raise accuracy here; when it does, this
baseline is re-recorded upward.

Per-run details and traces are in the LangSmith experiment for each run
(`make eval` prints the experiment URL).
