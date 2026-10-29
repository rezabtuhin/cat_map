"""Prepare isolated container state, or run developer commands without bootstrapping."""

import logging
import os
import subprocess
import sys
from pathlib import Path


def setup() -> None:
    """Acquire pinned models on first use and build only when no release is selected."""
    result = subprocess.run(
        ["sha256sum", "--check", "--status", "/opt/project-inputs.sha256"], check=False
    )
    if result.returncode:
        raise ValueError("Dependencies changed; rebuild with: docker compose up --build")

    mode = os.environ.get("CATEGORY_MATCHER_MODEL_DOWNLOAD", "auto")
    if mode not in {"auto", "offline"}:
        raise ValueError("CATEGORY_MATCHER_MODEL_DOWNLOAD must be auto or offline")

    from category_matcher.api.settings import APISettings
    from category_matcher.bundles import FileLease, load_active

    settings = APISettings.from_env()
    with FileLease(settings.artifacts / ".bootstrap.lock"):
        metadata = [
            Path("models") / name for name in ("embedding_model.json", "nli_reranker_model.json")
        ]
        preparation = [sys.executable, "scripts/prepare_models.py"]
        if mode == "auto" and not all(path.exists() for path in metadata):
            print("First setup: downloading the pinned model snapshots.", flush=True)
            preparation.append("--download")
        subprocess.run(preparation, check=True)

        if not (settings.artifacts / "active.json").exists():
            # Fail before a costly build if another API owns this artifact root.
            with FileLease(settings.artifacts / ".service.lock"):
                pass
            print("First setup: building and activating the Google taxonomy bundle.", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "category_matcher",
                    "--artifacts",
                    str(settings.artifacts),
                    "build",
                    "--activate",
                ],
                check=True,
            )
        else:
            try:
                load_active(settings.artifacts, require_release=True)
            except (ValueError, OSError) as error:
                raise ValueError(
                    "The saved release could not be validated. Stop the API and inspect the "
                    "error before rebuilding; see doc/DOCKER.md#rebuild-matching-artifacts. "
                    f"Reason: {error}"
                ) from error


def main() -> int:
    command = sys.argv[1:] or ["serve"]
    if command not in (["serve"], ["setup"]):
        # Operator-supplied commands are executed directly, never through a shell.
        os.execvp(command[0], command)
    setup()
    if command == ["serve"]:
        import uvicorn

        from category_matcher.api.app import create_app
        from category_matcher.api.settings import APISettings

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        # Docker publishes this container interface only on the host's loopback address.
        uvicorn.run(
            create_app(APISettings.from_env()),
            host="0.0.0.0",
            port=8000,
            workers=1,
            access_log=False,
            timeout_keep_alive=5,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Container startup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
