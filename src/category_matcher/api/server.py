"""Run one local API process; the bounded service owns model concurrency."""

import argparse
import logging
from pathlib import Path

import uvicorn

from category_matcher.api.app import create_app
from category_matcher.api.settings import APISettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["127.0.0.1", "::1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--queue-size", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-body-bytes", type=int)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    values = APISettings.from_env().model_dump()
    values.update(
        {
            key: value
            for key, value in vars(args).items()
            if key not in {"host", "port"} and value is not None
        }
    )
    settings = APISettings.model_validate(values)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
