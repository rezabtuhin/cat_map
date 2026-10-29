"""Prepare exactly the checked-in revisions; network access requires --download."""

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    pins = json.loads(Path("config/model_pins.json").read_text())
    for role, filename in (
        ("embedding", "embedding_model.json"),
        ("ranking", "nli_reranker_model.json"),
    ):
        metadata = pins[role]
        if re.fullmatch(r"[0-9a-f]{40}", metadata["revision"]) is None:
            raise ValueError("Model revision must be immutable")
        snapshot = snapshot_download(
            metadata["name"],
            revision=metadata["revision"],
            cache_dir="models/hub",
            local_files_only=not args.download,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "1_Pooling/*"],
        )
        output = Path("models") / filename
        value = {**metadata, "snapshot": str(Path(snapshot).resolve())}
        if output.exists() and json.loads(output.read_text()) != value:
            raise ValueError(f"Existing {output} differs from the pinned configuration")
        output.write_text(json.dumps(value, indent=2) + "\n")
        print(f"{role}: {metadata['name']} @ {metadata['revision']}")


if __name__ == "__main__":
    main()
