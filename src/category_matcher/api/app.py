"""FastAPI composition root; the event loop never owns a SQLite connection."""

import asyncio
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from category_matcher.api.errors import ServiceError, error_response
from category_matcher.api.schemas import (
    CategoryResponse,
    ErrorResponse,
    MatchRequest,
    MatchResponse,
    TaxonomyInfo,
)
from category_matcher.api.service import InferenceService, log_event
from category_matcher.api.settings import APISettings


class RequestBoundary:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        event_id = str(uuid4())
        state = scope.setdefault("state", {})
        state.update(event_id=event_id, received_at=started)
        status = 500

        async def tracked_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"
                ]
                message = {**message, "headers": headers + [(b"x-request-id", event_id.encode())]}
            await send(message)

        async def reject(code, message, http_status):
            state["error_code"] = code
            await error_response(code, message, http_status, event_id)(scope, receive, tracked_send)

        try:
            if scope["method"] == "POST":
                headers = dict(scope.get("headers", []))
                if scope["path"] == "/v1/categories/match" and (
                    headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
                    != b"application/json"
                ):
                    return await reject("unsupported_media_type", "Use application/json", 415)
                try:
                    length = int(headers.get(b"content-length", b"0"))
                    if length < 0:
                        raise ValueError("Negative content length")
                except ValueError:
                    return await reject("invalid_content_length", "Invalid Content-Length", 400)
                if length > self.settings.max_body_bytes:
                    return await reject(
                        "request_too_large", "Request body exceeds the configured limit", 413
                    )
                parts, size = [], 0
                while True:
                    remaining = started + self.settings.timeout_seconds - time.perf_counter()
                    if remaining <= 0:
                        return await reject("request_timeout", "The request deadline expired", 504)
                    try:
                        message = await asyncio.wait_for(receive(), remaining)
                    except TimeoutError:
                        return await reject("request_timeout", "The request deadline expired", 504)
                    if message["type"] == "http.disconnect":
                        status = 499
                        return
                    body = message.get("body", b"")
                    size += len(body)
                    if size > self.settings.max_body_bytes:
                        return await reject(
                            "request_too_large", "Request body exceeds the configured limit", 413
                        )
                    parts.append(body)
                    if not message.get("more_body", False):
                        break
                buffered = b"".join(parts)
                replayed = False
                original_receive = receive

                async def replay():
                    nonlocal replayed
                    if not replayed:
                        replayed = True
                        return {"type": "http.request", "body": buffered, "more_body": False}
                    return await original_receive()

                receive = replay
            await self.app(scope, receive, tracked_send)
        except asyncio.CancelledError:
            status = 499
            state["error_code"] = "request_cancelled"
            raise
        finally:
            log_event(
                "request_completed",
                event_id=event_id,
                method=scope["method"],
                path=scope["path"],
                http_status=status,
                stage="http",
                total_ms=(time.perf_counter() - started) * 1000,
                error_code=state.get("error_code"),
            )


def create_app(settings: APISettings | None = None, *, matcher_factory=None) -> FastAPI:
    settings = settings or APISettings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        service = InferenceService(settings, matcher_factory)
        app.state.service = service
        service.start()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(
        title="Category Matcher Development API",
        version="0.4.0",
        lifespan=lifespan,
        description="Local development API. Only accepted matches assign a category.",
    )
    app.add_middleware(RequestBoundary, settings=settings)

    def service(request):
        instance = getattr(request.app.state, "service", None)
        if instance is None:
            raise ServiceError("service_unready", "The matching service is not ready")
        return instance

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        request.state.error_code = error.code
        return error_response(error.code, error.message, error.status, request.state.event_id)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        request.state.error_code = "invalid_request"
        details = [
            {"location": list(e["loc"]), "message": e["msg"], "type": e["type"]}
            for e in error.errors()
        ]
        return error_response(
            "invalid_request", "Invalid request data", 422, request.state.event_id, details
        )

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        code = {404: "not_found", 405: "method_not_allowed"}.get(error.status_code, "http_error")
        request.state.error_code = code
        return error_response(code, str(error.detail), error.status_code, request.state.event_id)

    @app.get("/health/live")
    async def live():
        return {"live": True, "service_mode": "development"}

    @app.get("/health/ready")
    async def ready(request: Request):
        instance = service(request)
        return JSONResponse(
            instance.health(), status_code=200 if instance.state == "ready" else 503
        )

    @app.get(
        "/v1/taxonomies",
        response_model=list[TaxonomyInfo],
        responses={503: {"model": ErrorResponse}},
    )
    async def taxonomies(request: Request):
        instance = service(request)
        instance.ensure_ready()
        return [
            TaxonomyInfo(
                taxonomy_key="google",
                taxonomy_version=instance.taxonomy_version,
                artifact_version=instance.artifact_version,
                languages=["en"],
                category_count=len(instance.categories),
            )
        ]

    @app.get(
        "/v1/taxonomies/{taxonomy_key}/categories/{category_id}",
        response_model=CategoryResponse,
        responses={code: {"model": ErrorResponse} for code in (404, 422, 503)},
    )
    async def canonical(
        request: Request,
        taxonomy_key: str,
        category_id: str,
        taxonomy_version: str | None = Query(default=None, min_length=1, max_length=128),
    ):
        instance = service(request)
        instance.validate_version(taxonomy_version)
        if taxonomy_key != "google" or category_id not in instance.categories:
            raise ServiceError("category_not_found", "Unknown taxonomy or category ID", 404)
        category = instance.categories[category_id]
        return CategoryResponse(
            id=category.category_id,
            label=category.label,
            path=category.full_path,
            taxonomy_key="google",
            taxonomy_version=instance.taxonomy_version,
            artifact_version=instance.artifact_version,
            parent_category_id=category.parent_id,
            depth=category.depth,
            is_leaf=category.is_leaf,
        )

    @app.post(
        "/v1/categories/match",
        response_model=MatchResponse,
        responses={code: {"model": ErrorResponse} for code in (413, 415, 422, 503, 504)},
    )
    async def match(payload: MatchRequest, request: Request):
        return await service(request).submit(
            payload, request.state.event_id, request.state.received_at
        )

    return app
