# Development labels

`development.jsonl` contains 200 **AI-authored draft fixtures**, not collected user
inputs or human-reviewed ground truth. Rankings against these labels are diagnostic
only. Sprint 2 adds `calibration.draft.jsonl` and `test.draft.jsonl`, each containing
200 additional review proposals. The three files contain 600 examples with disjoint
queries, category targets, and declared input families across splits.

The calibration proposals were used for an explicitly experimental policy fit.
The test proposals have not been scored. They are not an independently collected,
reviewed test set. Do not rename them or change their role metadata and claim that
an independent real-world holdout has been collected.

`development.audited.draft.jsonl` is a separate, still-unreviewed derivative of the
original development data. It proposes six label/ambiguity changes after a failure
audit. It does not replace the original primary benchmark. The
[audit](../../../reports/sprint_2/improvement/label_audit.json) and
[findings](../../../reports/sprint_2/improvement/FINDINGS.md) distinguish those
proposals from human review. Do not use this derivative as an independent split.

The [current review queue](../../../reports/sprint_2/revision_3/LABEL_REVIEW.md)
shows remaining development failures, the earlier audit proposals, and canonical
paths for comparison. It does not approve labels or cover all required review.

`user_examples_2.jsonl` contains five additional real store titles and legacy LAYA
outputs supplied by the user. Expected IDs/statuses remain AI proposals awaiting
review. They were inspected during development and are not independent holdout
inputs. The [review workflow](../reviews/README.md) provides pending packets and
source-bound import commands without approving any labels automatically.

Before using these records as trusted labels, a category reviewer must check each
query against the source taxonomy, correct acceptable IDs or expected status, and
set `review_status` to `approved`, with nonempty `reviewer` and `reviewed_at` fields.
Ambiguous inputs must remain ambiguous when the text cannot identify one category.
The same review fields are required to approve aliases in `../aliases.jsonl`.

Draft evaluation requires the explicit `--allow-draft` flag. Reports preserve the
dataset hash, review counts, method settings, and per-example results. A draft
report must never be described as validated real-world accuracy.

The initial development fixture counts are 40 exact, 60 paraphrase, 32 typo,
28 hierarchy, 20 ambiguous, 16 no-match, and 4 unsupported-language examples.
Some examples intentionally share category/input families; keep those families
together when collecting later calibration/test data. Canonical exact fixtures
cover all 21 top-level branches. Do not copy them into a holdout.

Use `category-matcher check-data` to check overlaps. Qualification requires reviewed
calibration and a separate reviewed dataset marked `independent_holdout`, along
with measured precision, coverage, retrieval recall, and non-exact model evidence.
The qualification command records holdout accesses so inspected data cannot be
silently reused to validate a different policy. The production statistical gate
still belongs to Sprint 5; a small pilot test cannot establish that claim.
