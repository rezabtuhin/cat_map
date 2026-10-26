# Development labels

`development.jsonl` contains 200 **AI-authored draft fixtures**, not collected user
inputs or human-reviewed ground truth. Rankings against these labels are diagnostic
only. No calibration or untouched test set has been created.

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
