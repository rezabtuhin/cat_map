"""Prepare pinned multilingual INT8 assets and metadata; downloads require --download."""

import argparse
import sys
from pathlib import Path

from huggingface_hub.errors import HfHubHTTPError

from category_matcher.model_assets import load_pins, prepare_models, write_runtime_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned files")
    args = parser.parse_args()
    models = load_pins(PROJECT_ROOT / "config/onnx_model_pins.json")
    root = PROJECT_ROOT / "models"
    prepare_models(models, root, download=args.download)
    write_runtime_metadata(models, root)
    print("ONNX models are ready. Build a new bundle before switching the API release.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, HfHubHTTPError) as error:
        print(f"Model preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
