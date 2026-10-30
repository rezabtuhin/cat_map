"""Pinned local cross-encoder adapters; ranking never authorizes acceptance."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
SUPPORTED_MODELS = (MODEL_NAME, "BAAI/bge-reranker-base", NLI_MODEL)


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


def download_model(cache: Path, name: str = MODEL_NAME, revision: str = "main") -> dict:
    from huggingface_hub import HfApi, snapshot_download

    if name not in SUPPORTED_MODELS:
        raise ValueError("Unsupported reranker model")
    revision = HfApi().model_info(name, revision=revision).sha
    snapshot = snapshot_download(
        name,
        revision=revision,
        cache_dir=str(cache),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
    )
    metadata = {"name": name, "revision": revision, "snapshot": str(Path(snapshot).resolve())}
    if name == NLI_MODEL:
        metadata.update(
            {
                "score_profile": "category_entailment_v1",
                "entailment_index": 1,
                "premise_template": "These products are {query}.",
                "hypothesis_template": "These products are {label}.",
            }
        )
    return metadata


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text())
    metadata_path = Path(profile["model_metadata"])
    if not metadata_path.is_absolute():
        metadata_path = path.parent / metadata_path
    metadata = json.loads(metadata_path.read_text())
    if (metadata["name"], metadata["revision"]) != (
        profile["model_name"],
        profile["model_revision"],
    ):
        raise ValueError("Ranking profile model revision does not match the downloaded model")
    if metadata.get("score_profile") != profile["score_profile"]:
        raise ValueError("Ranking score profile mismatch")
    templates = {}
    for key, placeholder in (("premise_template", "{query}"), ("hypothesis_template", "{label}")):
        if key not in profile:
            continue
        template = profile[key]
        if (
            metadata.get("score_profile") != "category_entailment_v1"
            or not isinstance(template, str)
            or len(template) > 300
            or template.count(placeholder) != 1
            or any(char in template.replace(placeholder, "") for char in "{}")
        ):
            raise ValueError("Invalid category-entailment pair template")
        templates[key] = template
    if (
        type(profile["candidate_limit"]) is not int
        or type(profile["rank_constant"]) is not int
        or not 1 <= profile["candidate_limit"] <= 200
        or profile["rank_constant"] < 1
    ):
        raise ValueError("Invalid ranking profile limits")
    weight = profile["rank_fusion_weight"]
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0
    ):
        raise ValueError("Invalid ranking profile weight")
    return {
        **metadata,
        **templates,
        **{key: profile[key] for key in ("candidate_limit", "rank_fusion_weight", "rank_constant")},
    }


class CrossEncoderReranker:
    def __init__(self, model: dict):
        self.metadata = model
        if model.get("backend") == "onnx":
            from category_matcher.onnx_backend import ONNXNLIPredictor

            self.model = ONNXNLIPredictor(model)
            self.aligned_parameter_copies = 0
            return
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
        self.metadata = model

    def rank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        original = candidates
        limit = self.metadata.get("candidate_limit", len(candidates))
        candidates = candidates[:limit]
        entailment = self.metadata.get("score_profile") == "category_entailment_v1"
        if entailment:
            pairs = [
                (
                    self.metadata["premise_template"].format(query=query),
                    self.metadata["hypothesis_template"].format(label=c["label"]),
                )
                for c in candidates
            ]
        else:
            pairs = [(query, f"Category: {c['label']}. Path: {c['path']}.") for c in candidates]
        options = {}
        if self.metadata.get("backend") != "onnx":
            import torch

            options["activation_fn"] = torch.nn.Identity()
        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False, **options)
        if entailment:
            logits = np.asarray(scores, dtype=np.float32)
            if logits.ndim != 2 or logits.shape[1] != 3:
                raise ValueError("Expected contradiction/entailment/neutral model outputs")
            index = self.metadata["entailment_index"]
            other = [i for i in range(3) if i != index]
            if type(index) is not int or not 0 <= index < 3 or not np.isfinite(logits).all():
                raise ValueError("Invalid NLI entailment logits or label index")
            scores = logits[:, index] - np.logaddexp(logits[:, other[0]], logits[:, other[1]])
        ranked = [
            {**candidate, "retrieval_rank": candidate["rank"], "reranker_score": float(score)}
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda c: (-c["reranker_score"], c["id"]))
        if "rank_fusion_weight" in self.metadata:
            weight, constant = self.metadata["rank_fusion_weight"], self.metadata["rank_constant"]
            model_ranks = {c["id"]: i for i, c in enumerate(ranked, 1)}
            model_scores = {c["id"]: c["reranker_score"] for c in ranked}
            ranked = [
                {
                    **candidate,
                    "retrieval_rank": i,
                    "entailment_logit_margin": model_scores.get(candidate["id"]),
                    "reranker_score": 1 / (constant + i)
                    + (
                        weight / (constant + model_ranks[candidate["id"]])
                        if candidate["id"] in model_ranks
                        else 0
                    ),
                }
                for i, candidate in enumerate(original, 1)
            ]
            ranked.sort(key=lambda c: (-c["reranker_score"], c["id"]))
        else:
            ranked += original[limit:]
        return [{**candidate, "rank": i} for i, candidate in enumerate(ranked, 1)]


MiniLMReranker = CrossEncoderReranker  # Existing CLI/API import remains compatible.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("models/hub"))
    parser.add_argument("--output", type=Path, default=Path("models/reranker_model.json"))
    parser.add_argument("--model-name", choices=SUPPORTED_MODELS, default=MODEL_NAME)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    metadata = download_model(args.cache, args.model_name, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
