"""Pinned, local-only model inference; downloads are an explicit build step."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DIMENSION = 768
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def download_model(cache: Path, revision: str = "main") -> dict:
    from huggingface_hub import HfApi, snapshot_download

    resolved = HfApi().model_info(MODEL_NAME, revision=revision).sha
    snapshot = snapshot_download(
        MODEL_NAME,
        revision=resolved,
        cache_dir=str(cache),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "1_Pooling/*"],
    )
    return {
        "name": MODEL_NAME,
        "revision": resolved,
        "snapshot": str(Path(snapshot).resolve()),
        "dimension": DIMENSION,
        "query_instruction": QUERY_INSTRUCTION,
    }


class BGEEncoder:
    def __init__(self, model: dict):
        self.onnx = None
        if model.get("backend") == "onnx":
            from category_matcher.onnx_backend import ONNXEncoder

            self.onnx = ONNXEncoder(model)
            self.model = self.onnx.model
            return
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        self.model = SentenceTransformer(
            model["snapshot"], device="cpu", local_files_only=True, trust_remote_code=False
        )
        if self.model.get_embedding_dimension() != model["dimension"]:
            raise ValueError("Model embedding dimension does not match the manifest")
        self.instruction = model["query_instruction"]

    def encode_documents(self, texts: list[str], *, progress: bool = False) -> np.ndarray:
        if self.onnx is not None:
            return self.onnx.encode_documents(texts, progress=progress)
        return self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        ).astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if self.onnx is not None:
            return self.onnx.encode_queries(texts)
        return self.encode_documents([self.instruction + text for text in texts])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("models/hub"))
    parser.add_argument("--output", type=Path, default=Path("models/embedding_model.json"))
    args = parser.parse_args()
    metadata = download_model(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
