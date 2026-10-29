"""Isolated compatibility smoke test for the pinned models and local CPU backend."""

import argparse
import ctypes
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from category_matcher.artifacts import load_manifest, resolve_bundle
from category_matcher.encoder import BGEEncoder
from category_matcher.memory import process_memory
from category_matcher.quantization import PROFILES, apply_profile, tensor_bytes
from category_matcher.reranker import CrossEncoderReranker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["encoder", "reranker"], required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES - {"float32"}), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(resolve_bundle(Path("artifacts/google")))
    texts = ["machines for cutting lawn grass", "movies on DVD", "Hoodies & Sweatshirts"]
    if args.role == "encoder":
        adapter = BGEEncoder(manifest["model"])
        module = adapter.model[0].auto_model

        def predict():
            return adapter.encode_queries(texts)
    else:
        adapter = CrossEncoderReranker(manifest["ranking_model"])
        module = adapter.model.model
        pairs = [
            ("The product category is Jackets.", "The product category is Coats & Jackets."),
            (
                "The product category is Fitness Equipment.",
                "The product category is Exercise & Fitness.",
            ),
        ]

        def predict():
            return adapter.model.predict(pairs, batch_size=8, show_progress_bar=False)

    before = tensor_bytes(module)
    reference = predict()
    started = time.perf_counter()
    converted = apply_profile(module, args.profile)
    convert_seconds = time.perf_counter() - started
    after = tensor_bytes(module)
    output = predict()
    assert np.isfinite(output).all()
    memory_before_gc = process_memory()
    gc.collect()
    memory_after_gc = process_memory()
    # Diagnostic only: ask macOS's allocator to return unused pages after the
    # one-time conversion. No live tensor is explicitly freed or replaced here.
    returned = None
    if sys.platform == "darwin":
        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        release = system.malloc_zone_pressure_relief
        release.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        release.restype = ctypes.c_size_t
        returned = release(None, 0)
    report = {
        "baseline_artifact_version": manifest["artifact_version"],
        "role": args.role,
        "conversion": converted,
        "conversion_seconds": convert_seconds,
        "before": before,
        "after": after,
        "process_memory": process_memory(),
        "memory_before_gc": memory_before_gc,
        "memory_after_gc": memory_after_gc,
        "allocator_bytes_returned": returned,
        "output_shape": list(output.shape),
        "maximum_absolute_output_difference": float(np.abs(reference - output).max()),
        "note": "Compatibility only; this does not measure matching accuracy or warm latency.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
