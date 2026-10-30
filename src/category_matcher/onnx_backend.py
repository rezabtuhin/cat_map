"""Local CPU-only ONNX sessions and tokenization shared by embedding and NLI adapters."""

import json
import re
from pathlib import Path

import numpy as np

from category_matcher.model_assets import REQUIRED_FILES, PinnedFile, verify_file


def verified_model_files(metadata: dict) -> dict[str, str]:
    """Validate exactly the release-pinned files; download cache metadata is not model state."""
    if (
        not isinstance(metadata.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", metadata["revision"]) is None
    ):
        raise ValueError("ONNX models require an immutable revision")
    snapshot = Path(metadata["snapshot"]).resolve()
    if not snapshot.is_dir():
        raise ValueError("Missing ONNX model directory; run scripts/prepare_onnx_models.py")
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_FILES:
        raise ValueError("ONNX metadata must pin the five required files")
    inventory = {}
    for name, expected in files.items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot):
            raise ValueError("ONNX model file escapes the pinned snapshot")
        if (
            not isinstance(expected, dict)
            or type(expected.get("size")) is not int
            or expected["size"] <= 0
            or not isinstance(expected.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None
        ):
            raise ValueError("Invalid ONNX file integrity metadata")
        if not verify_file(path, PinnedFile(name, expected["size"], expected["sha256"])):
            raise ValueError(f"Missing pinned ONNX model file: {name}")
        inventory[name] = expected["sha256"]
    return inventory


class ONNXTextModel:
    def __init__(self, metadata: dict):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        verified_model_files(metadata)
        self.metadata = metadata
        directory = Path(metadata["snapshot"])
        for key, maximum in (("max_length", 512), ("batch_size", 32), ("intra_op_threads", 2)):
            if type(metadata.get(key)) is not int or not 1 <= metadata[key] <= maximum:
                raise ValueError(f"Invalid ONNX {key}")
        tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=metadata["max_length"])
        pad = tokenizer_config.get("pad_token", "<pad>")
        pad = pad["content"] if isinstance(pad, dict) else pad
        pad_id = self.tokenizer.token_to_id(pad)
        if pad_id is None:
            raise ValueError("The tokenizer has no configured padding token")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad)
        options = ort.SessionOptions()
        options.intra_op_num_threads = metadata["intra_op_threads"]
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.enable_cpu_mem_arena = False
        # Loading by bytes disallows resolving external graph files from arbitrary paths.
        graph = (directory / "onnx/model_int8.onnx").read_bytes()
        self.session = ort.InferenceSession(
            graph, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {value.name for value in self.session.get_inputs()}
        self.outputs = {value.name for value in self.session.get_outputs()}
        if not {"input_ids", "attention_mask"} <= self.inputs or self.inputs - {
            "input_ids",
            "attention_mask",
            "token_type_ids",
        }:
            raise ValueError("Unexpected ONNX text model inputs")

    def run(
        self, texts: list[str] | list[tuple[str, str]], output: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if not texts or output not in self.outputs:
            raise ValueError("Expected nonempty text and a configured ONNX output")
        encoded = self.tokenizer.encode_batch(texts)
        values = {
            "input_ids": np.asarray([row.ids for row in encoded], dtype=np.int64),
            "attention_mask": np.asarray([row.attention_mask for row in encoded], dtype=np.int64),
            "token_type_ids": np.asarray([row.type_ids for row in encoded], dtype=np.int64),
        }
        result = self.session.run([output], {name: values[name] for name in self.inputs})[0]
        if not np.isfinite(result).all():
            raise ValueError("ONNX inference produced nonfinite outputs")
        return np.asarray(result, dtype=np.float32), values["attention_mask"]


class ONNXEncoder:
    def __init__(self, metadata: dict):
        self.model = ONNXTextModel(metadata)
        self.metadata = metadata
        if metadata.get("dimension") != 384 or metadata.get("pooling") != "mean":
            raise ValueError("Unsupported E5 embedding configuration")

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self.metadata["dimension"]), dtype=np.float32)
        batches = []
        size = self.metadata["batch_size"]
        for start in range(0, len(texts), size):
            hidden, mask = self.model.run(
                [prefix + text for text in texts[start : start + size]], "last_hidden_state"
            )
            if hidden.shape != (*mask.shape, self.metadata["dimension"]):
                raise ValueError("Unexpected E5 output dimensions")
            weights = mask.astype(np.float32)
            lengths = weights.sum(axis=1, keepdims=True)
            if np.any(lengths == 0):
                raise ValueError("An encoded input has no unmasked tokens")
            pooled = np.einsum("btd,bt->bd", hidden, weights) / lengths
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("E5 produced a zero-norm embedding")
            batches.append((pooled / norms).astype(np.float32))
        return np.concatenate(batches)

    def encode_documents(self, texts: list[str], *, progress: bool = False) -> np.ndarray:
        return self._encode(texts, self.metadata["document_instruction"])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.metadata["query_instruction"])


class ONNXNLIPredictor:
    def __init__(self, metadata: dict):
        self.model = ONNXTextModel(metadata)
        self.batch_size = metadata["batch_size"]
        labels = json.loads((Path(metadata["snapshot"]) / "config.json").read_text())["id2label"]
        if labels.get(str(metadata["entailment_index"])) != "entailment":
            raise ValueError("NLI entailment index disagrees with the pinned configuration")

    def predict(self, pairs: list[tuple[str, str]], **_kwargs) -> np.ndarray:
        batches = [
            self.model.run(pairs[start : start + self.batch_size], "logits")[0]
            for start in range(0, len(pairs), self.batch_size)
        ]
        result = np.concatenate(batches) if batches else np.empty((0, 3), dtype=np.float32)
        if result.shape != (len(pairs), 3):
            raise ValueError("Expected three NLI logits per pair")
        return result
