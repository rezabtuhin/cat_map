# HTTP API and client integration

Start `uv run category-matcher-api` from the repository root. The default base URL
is `http://127.0.0.1:8000`; interactive Swagger documentation is `/docs` and the
machine-readable schema is `/openapi.json`.

## Routes

| Method and path | Behavior |
|---|---|
| `GET /health/live` | Process liveness; available while initialization or inference is running |
| `GET /health/ready` | Service state, active versions, queue counts, capabilities, and memory information; 503 while unready |
| `GET /v1/taxonomies` | Currently available Google taxonomy/version information |
| `GET /v1/taxonomies/google/categories/{category_id}` | Canonical category, parent, depth, and leaf information |
| `POST /v1/categories/match` | Accepted match, provisional selection, or an abstention status |

The removed feedback route returns 404. Category lookup accepts an optional
`taxonomy_version` query parameter; a supplied version must match the selected
version. The service has no public-network authentication layer and its launcher
permits only loopback hosts.

## Match inputs

| Field | Meaning and validation |
|---|---|
| `taxonomy_key` | Required string `"google"` |
| `taxonomy_version` | Optional active taxonomy version; a mismatch is rejected |
| `category` | Store category text, up to 300 characters; required unless `category_id` is provided |
| `category_id` | Optional canonical Google ID, as a string; text supplied with it must identify that same category |
| `parent_category_id` | Optional canonical Google ID restricting the allowed subtree |
| `context` | Optional useful product description, up to 1,000 characters |
| `source_ancestors` | Optional store names, root to immediate parent; up to 20 nonblank labels, 300 characters per label, 1,000 total |
| `language` | Defaults to `"auto"`; optional enabled language hint such as `"fr"`, `"bn"`, or `"zh-TW"`; unsupported explicit hints yield `unsupported_language` |
| `limit` | Number of returned alternatives, 1–10, default 3; does not set the retrieval pool size |

The Pydantic contract rejects unknown fields and incorrect types with 422. IDs are
strings, and leading/trailing whitespace is trimmed on the text fields. Requests
must use `Content-Type: application/json` and fit the configured body-size limit.

`auto` passes the original multilingual text to the models; it does not run a
separate language detector or guarantee recognition of every language. Initial
enabled base hints are `ar`, `bn`, `de`, `en`, `es`, `fr`, `hi`, `id`, `it`, `ja`,
`ko`, `pt`, `ru`, `tr`, and `zh`. Regional tags use the enabled base language.
Mixed-language names are accepted. Quality across these languages remains unqualified.
Unicode combining marks are preserved; lexical spelling/plural heuristics still
primarily support English, while cross-lingual evidence comes from the models.

`GET /v1/taxonomies` retains `languages: ["en"]` for canonical taxonomy labels and
adds `input_languages` for the active model's enabled input hints. Responses keep
canonical Google IDs and English paths. Clients need not translate taxonomy IDs.

### Category name only

```bash
curl -X POST http://127.0.0.1:8000/v1/categories/match \
  -H 'Content-Type: application/json' \
  -d '{"taxonomy_key":"google","category":"Hoodies & Sweatshirts|Clothing"}'
```

### Authoritative store hierarchy

```json
{
  "taxonomy_key": "google",
  "category": "Collections",
  "source_ancestors": [
    "Clothing",
    "Men",
    "Tops",
    "Hoodies & Sweatshirts|Clothing"
  ],
  "limit": 3
}
```

Use actual WooCommerce parent relationships. A literal pipe in a store name is
preserved; the matcher can interpret the existing `label|context` syntax. Keep
structured ancestry separate rather than concatenating the whole tree into one
category string. Use `context` for descriptions available at inference time.

### Direct canonical lookup

```json
{"taxonomy_key":"google","category_id":"212"}
```

For a hard taxonomy constraint, use `parent_category_id`. A WooCommerce term ID
is not a Google ID and must not be sent as that constraint.

## Responses

| Status | Assigned `category` | Meaning |
|---|---|---|
| `matched` | Canonical record | Safe lookup or acceptance under the loaded qualified policy |
| `provisional` | `null` | Ranked candidate evidence that still requires review |
| `needs_clarification` | `null` | Insufficient or conflicting product meaning |
| `no_match` | `null` | No accepted match under the decision rules |
| `unsupported_language` | `null` | Input language unsupported by the current matcher |

An illustrative provisional response, with installation-specific IDs and timings
represented by example values:

```json
{
  "status": "provisional",
  "taxonomy_key": "google",
  "taxonomy_version": "example-taxonomy-version",
  "artifact_version": "example-artifact-version",
  "category": null,
  "alternatives": [
    {"id":"212","label":"Shirts & Tops","path":"Apparel & Accessories > Clothing > Shirts & Tops"}
  ],
  "provisional_category": {
    "id":"212",
    "label":"Shirts & Tops",
    "path":"Apparel & Accessories > Clothing > Shirts & Tops"
  },
  "requires_review": true,
  "method": "source_ancestor",
  "reason_code": "specific_product_ancestor_provisional",
  "clarification": null,
  "event_id": "example-request-id",
  "timings": {"queue_wait_ms":0.1,"inference_ms":1.0,"total_ms":1.5}
}
```

`provisional_category` must equal the first alternative only when the status is
`provisional` and alternatives exist; `requires_review` then equals `true`.
Other uncertain statuses have no provisional selection, even if clarification
alternatives are displayed. `category` is populated only for `matched`.
The `X-Request-ID` header matches the response `event_id` and structured log ID.
Successful match responses are written to the runtime database before returning.

CLI output is the engine's `MatchResult`, with `candidates` and richer diagnostic
fields. HTTP translates those records into the stable `alternatives` contract;
clients should not expect CLI JSON to have the HTTP response schema.

## Errors, overload, and timing

Errors use an envelope such as:

```json
{"error":{"code":"queue_full","message":"The inference queue is full"},"event_id":"example-request-id"}
```

| HTTP status | Typical cause |
|---|---|
| 404 | Unknown category/vendor route or removed feedback endpoint |
| 413 | Body exceeds the configured size limit |
| 415 | Match request is not JSON |
| 422 | Invalid fields, unknown/conflicting category IDs, unsupported requested version |
| 503 | Startup failure/unready state, full queue, or failed inference |
| 504 | Body reading, pending wait, or response waiting exceeds the deadline |

Service 503 errors carry `Retry-After: 1`. A retry is a new request/event; there is
no idempotency-key contract. One inference runs at a time; default pending capacity
is eight and the default deadline is 30 seconds. Timing fields are milliseconds.
A timeout does not stop an already-running native model call or release its slot
before it finishes. Health and canonical lookups do not join the inference queue.

## WooCommerce contract

The shared free plugin mapper supplies WooCommerce names and ancestry from term
IDs, including when Pro is active. PHP calls this API using
`WPFM_CATEGORY_MATCHER_API_URL` when configured, otherwise `http://127.0.0.1:8000`.
The URL must be reachable from the WordPress host; loopback refers to that host's
own network environment.

Accepted matches fill the merchant-category dropdown. A provisional selection may
also fill it but displays **Provisioning needed** until reviewed or replaced.
Saving a mapping remains a user action in the plugin. The API's removed feedback
feature is not the plugin's local review flag. The plugin checks readiness
capabilities for ancestry support before using this integration.
