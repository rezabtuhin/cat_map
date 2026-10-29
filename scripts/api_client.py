"""Minimal caller for the local development API."""

import argparse
import json
import sys

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category")
    parser.add_argument("--context")
    parser.add_argument("--parent-category-id")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    response = httpx.post(
        args.url.rstrip("/") + "/v1/categories/match",
        timeout=45,
        json={
            "taxonomy_key": "google",
            "category": args.category,
            "context": args.context,
            "parent_category_id": args.parent_category_id,
        },
    )
    print(json.dumps(response.json(), indent=2))
    # Consumers assign only response["category"] on an accepted matched result.
    # Alternatives and provisional suggestions never become automatic mappings.
    return 0 if response.is_success else 2


if __name__ == "__main__":
    sys.exit(main())
