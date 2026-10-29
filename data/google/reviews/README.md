# Review inputs and mapping proposals

The packets in this directory are ready for human review. Exporting them does not
approve any label. Original evaluation files remain unchanged.

| Packet | Inputs | Origin |
|---|---:|---|
| `development.pending.jsonl` | 200 | Original AI-authored development fixtures |
| `calibration.pending.jsonl` | 200 | AI-authored calibration proposals |
| `user_batch_2.pending.jsonl` | 5 | Real user-supplied store titles, with AI label proposals |

The five new titles were inspected during development. They cannot be used as an
independent test set. Existing test proposals have not been exported or scored by
this workflow. A separately collected, reviewed holdout is still needed.

## Review and apply a packet

1. Open a packet and read each `input` and `proposal`. The listed IDs and canonical
   paths are proposals. Check what products the category actually contains.
2. Edit only the `decision` object. Leave unresolved records in `pending` state.
   Use `matched` with one or more acceptable canonical IDs when the input justifies
   them. Use `needs_clarification`, `no_match`, or `unsupported_language` with an
   empty ID list when appropriate. Broad categories are valid; a collection or a
   recommendation title does not imply a particular product type.
3. For an approved decision, provide `reviewer_kind: "human"`, your reviewer name,
   a timestamp with timezone, and a reason. For example, the decision fields are:

```json
{
  "state": "approved",
  "expected_status": "needs_clarification",
  "acceptable_category_ids": [],
  "reviewer_kind": "human",
  "reviewer": "<actual reviewer name>",
  "reviewed_at": "<actual ISO timestamp with timezone>",
  "reason": "<what was checked and why the input supports this decision>"
}
```

4. Apply your decisions to a **new** output file:

```bash
uv run category-matcher review-apply \
  --dataset data/google/evaluation/user_examples_2.jsonl \
  --reviews data/google/reviews/user_batch_2.pending.jsonl \
  --output data/google/evaluation/user_examples_2.reviewed.jsonl
```

The command binds decisions to the original dataset, taxonomy, example cohort,
and input contents. It rejects missing/duplicate examples, stale inputs, unknown
IDs, contradictory parent scopes, missing human-review fields, invalid statuses,
and overwriting an existing output. Pending records retain their original review
state. An entirely pending packet cannot be applied as an approved dataset.

The tool records the supplied reviewer declaration; it does not authenticate the
reviewer's identity. It preserves original label origin and split metadata.
Approving AI-authored labels does not turn those fixtures into independently
collected real-world inputs or an independent holdout.

## Capture additional real inputs

Create a JSONL file with one input per line. `context` contains the product
description available to the matcher; `parent_category_id` provides an optional
explicit taxonomy constraint. Product descriptions must be available during
inference if a reviewed label depends on them.

```json
{"query":"Erin Recommends","context":"winter coats and jackets","parent_category_id":"1604","input_family":"store-recommendations"}
```

```bash
uv run category-matcher capture-inputs --inputs path/to/real-inputs.jsonl \
  --source-name '<collection source>' \
  --output data/google/evaluation/collected.unlabeled.jsonl
uv run category-matcher review-export \
  --dataset data/google/evaluation/collected.unlabeled.jsonl \
  --output data/google/reviews/collected.pending.jsonl
```

Capture creates **unlabeled** records (`expected_status: null`), rather than
inventing expected outcomes. They cannot pass evaluation until labeled. An optional
`product_description` is copied to `context` when no context is provided; when
both are supplied they must agree. The records start with an unassigned dataset
role and do not claim independent provenance. Decide collection/split membership
before tuning, keep related input families together, and preserve the original
collection history. Inputs inspected to change the matcher belong to development.

All three commands work without model weights or an active vector artifact.
