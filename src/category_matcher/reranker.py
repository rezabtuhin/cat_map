"""Optional early Sprint 2 ranking experiment; never an acceptance policy."""

import argparse
import json
from pathlib import Path

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def align_cpu_parameters(model) -> int:
    """Copy misaligned mmap-backed weights into PyTorch-owned aligned storage.

    On the observed M1/torch 2.14.1 combination, this checkpoint's pooler has
    addresses offset by two bytes and crashes inside CPU linear inference.
    Copying preserves tensor values and avoids that observed native failure.
    """
    import torch

    copied = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.device.type == "cpu" and parameter.data_ptr() % 64:
                parameter.data = parameter.detach().clone(memory_format=torch.contiguous_format)
                copied += 1
    return copied


def download_model(cache: Path) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(MODEL_NAME).sha
    snapshot = snapshot_download(
        MODEL_NAME,
        revision=revision,
        cache_dir=str(cache),
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )
    return {"name": MODEL_NAME, "revision": revision, "snapshot": str(Path(snapshot).resolve())}


class MiniLMReranker:
    def __init__(self, model: dict):
        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(4)
        self.model = CrossEncoder(
            model["snapshot"],
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
            max_length=256,
        )
        self.aligned_parameter_copies = align_cpu_parameters(self.model)

    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        import torch

        if not candidates:
            return []
        pairs = [(query, f"Category: {c['label']}. Path: {c['path']}.") for c in candidates]
        scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
        )
        ranked = [
            {**candidate, "retrieval_rank": candidate["rank"], "reranker_score": float(score)}
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda c: (-c["reranker_score"], c["id"]))
        return [{**candidate, "rank": i} for i, candidate in enumerate(ranked, 1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("models/hub"))
    parser.add_argument("--output", type=Path, default=Path("models/reranker_model.json"))
    args = parser.parse_args()
    metadata = download_model(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
