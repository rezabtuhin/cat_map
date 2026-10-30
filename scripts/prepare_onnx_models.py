"""Prepare the proposed multilingual INT8 models; network access requires --download."""

import argparse
import fcntl
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "onnx/model_int8.onnx",
}


@dataclass(frozen=True)
class PinnedFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PinnedModel:
    role: str
    name: str
    revision: str
    directory: str
    files: tuple[PinnedFile, ...]


def load_pins(path: Path) -> list[PinnedModel]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or set(document) != {"format_version", "models"}:
        raise ValueError("Invalid ONNX pin document")
    if type(document["format_version"]) is not int or document["format_version"] != 1:
        raise ValueError("Unsupported ONNX pin format")
    models = document["models"]
    if not isinstance(models, dict) or set(models) != {"embedding", "ranking"}:
        raise ValueError("ONNX pins must define embedding and ranking models")
    result = []
    for role in ("embedding", "ranking"):
        model = models[role]
        if not isinstance(model, dict) or set(model) != {"name", "revision", "directory", "files"}:
            raise ValueError(f"Invalid {role} model pin")
        for key, pattern in (
            ("name", r"[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*"),
            ("revision", r"[0-9a-f]{40}"),
            ("directory", r"[a-z0-9]+(?:-[a-z0-9]+)*"),
        ):
            if not isinstance(model[key], str) or re.fullmatch(pattern, model[key]) is None:
                raise ValueError(f"Invalid {role} {key}")
        files = model["files"]
        if not isinstance(files, dict) or set(files) != REQUIRED_FILES:
            raise ValueError(f"{role} must pin exactly the five required INT8 model files")
        pinned_files = []
        for name, expected in files.items():
            if (
                not isinstance(expected, dict)
                or set(expected) != {"size", "sha256"}
                or type(expected["size"]) is not int
                or expected["size"] <= 0
                or not isinstance(expected["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None
            ):
                raise ValueError(f"Invalid integrity pin for {role}/{name}")
            pinned_files.append(PinnedFile(name, expected["size"], expected["sha256"]))
        result.append(
            PinnedModel(
                role, model["name"], model["revision"], model["directory"], tuple(pinned_files)
            )
        )
    if len({model.directory for model in result}) != len(result):
        raise ValueError("ONNX model directories must be distinct")
    return result


def model_directory(root: Path, model: PinnedModel) -> Path:
    directory = root / model.directory
    if directory.resolve().parent != root.resolve():
        raise ValueError("Model directory escapes model storage")
    for file in model.files:
        if not (directory / file.name).resolve().is_relative_to(directory.resolve()):
            raise ValueError("Model file escapes its model directory")
    return directory


def verify_file(path: Path, expected: PinnedFile) -> bool:
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != expected.size:
        raise ValueError(f"Pinned file size mismatch: {path}; preserve/move it before retrying")
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != expected.sha256:
        raise ValueError(f"Pinned file checksum mismatch: {path}; preserve/move it before retrying")
    return True


def prepare_models(models: list[PinnedModel], root: Path, *, download: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".onnx-preparation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another ONNX preparation command is running") from error

        # Check every existing file before fetching anything; never overwrite drifted files.
        missing = []
        for model in models:
            directory = model_directory(root, model)
            for file in model.files:
                if not verify_file(directory / file.name, file):
                    missing.append((model, directory, file))
        if missing and not download:
            raise ValueError(
                f"Missing {len(missing)} model files; rerun this command with --download"
            )
        for model, directory, file in missing:
            print(f"Downloading {model.role}: {file.name}", flush=True)
            hf_hub_download(
                model.name,
                file.name,
                revision=model.revision,
                local_dir=directory,
                token=False,
            )
            if not verify_file(directory / file.name, file):
                raise ValueError(f"Download did not create {model.role}/{file.name}")
        for model in models:
            print(f"Verified {model.role}: {model.name} @ {model.revision}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned files")
    args = parser.parse_args()
    models = load_pins(PROJECT_ROOT / "config/onnx_model_pins.json")
    prepare_models(models, PROJECT_ROOT / "models", download=args.download)
    print("ONNX files are prepared. The current API still uses its existing model configuration.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, HfHubHTTPError) as error:
        print(f"Model preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
