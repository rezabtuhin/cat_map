"""One owner thread holds SQLite and resident models; admission stays bounded."""

import asyncio
import json
import logging
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType

from category_matcher.api.errors import ServiceError
from category_matcher.api.schemas import CanonicalCategory, FeedbackResponse, MatchResponse, Timings
from category_matcher.api.settings import APISettings
from category_matcher.artifacts import digest
from category_matcher.bundles import FileLease, load_active, validate_policy
from category_matcher.feedback import FeedbackError, submit_feedback
from category_matcher.matcher import Matcher
from category_matcher.memory import process_memory
from category_matcher.runtime import RuntimeStore

logger = logging.getLogger("category_matcher.api")


def log_event(event, **fields):
    logger.info(json.dumps({"event": event, **fields}))


class ResidentMatcher:
    def __init__(self, matcher, lease):
        self.matcher, self.lease = matcher, lease

    def __getattr__(self, name):
        return getattr(self.matcher, name)

    def close(self):
        try:
            self.matcher.close()
        finally:
            self.lease.close()


def initialize_matcher(settings: APISettings):
    lease = FileLease(settings.artifacts / ".service.lock", shared=True).__enter__()
    matcher = None
    try:
        bundle, manifest, policy_path = load_active(settings.artifacts, require_release=True)
        if settings.policy is not None:
            validate_policy(settings.policy, manifest)
            if policy_path is None or digest(policy_path) != digest(settings.policy):
                raise ValueError("Activate a release with this policy before serving")
        matcher = Matcher(bundle)
        matcher.policy = validate_policy(policy_path, manifest)
        vector = matcher.encode_queries(["startup category warmup"])[0]
        matcher.match("startup category warmup", query_vector=vector)
        return ResidentMatcher(matcher, lease)
    except BaseException:
        if matcher is not None:
            matcher.close()
        lease.close()
        raise


@dataclass(eq=False)
class Job:
    request: object
    event_id: str
    received_at: float
    admitted_at: float
    future: asyncio.Future
    operation: str = "match"


class InferenceService:
    def __init__(self, settings, factory=None):
        self.settings = settings
        self.factory = factory or (lambda: initialize_matcher(settings))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="category-inference")
        self.state = "starting"
        self.matcher = None
        self.categories = MappingProxyType({})
        self.artifact_version = self.taxonomy_version = None
        self.policy_enabled = False
        self.pending = deque()
        self.active = None
        self.active_task = self.startup_task = None
        self.closing_task = None
        self.initialized = asyncio.Event()
        self.runtime = RuntimeStore(settings.runtime_db)

    def start(self):
        self.startup_task = asyncio.create_task(self._bootstrap())

    async def _bootstrap(self):
        started = time.perf_counter()
        try:
            if self.settings.runtime_db.resolve().is_relative_to(self.settings.artifacts.resolve()):
                raise ValueError("Runtime database must be outside immutable artifact storage")
            await asyncio.get_running_loop().run_in_executor(self.executor, self.runtime.initialize)
            self.matcher = await asyncio.get_running_loop().run_in_executor(
                self.executor, self.factory
            )
            self.categories = MappingProxyType(dict(self.matcher.store.categories))
            self.artifact_version = self.matcher.manifest["artifact_version"]
            self.taxonomy_version = self.matcher.version
            self.policy_enabled = self.matcher.policy is not None
            if self.state == "starting":
                self.state = "ready"
                log_event(
                    "startup_ready",
                    artifact_version=self.artifact_version,
                    semantic_auto_acceptance_enabled=self.policy_enabled,
                    initialization_ms=(time.perf_counter() - started) * 1000,
                    stage="initialization",
                )
        except Exception as error:
            if self.state == "starting":
                self.state = "failed"
            logger.error(
                json.dumps(
                    {
                        "event": "startup_failed",
                        "stage": "initialization",
                        "error_type": type(error).__name__,
                        "traceback": traceback.format_exc(),
                    }
                )
            )
        finally:
            self.initialized.set()

    async def wait_ready(self, timeout=60):
        await asyncio.wait_for(self.initialized.wait(), timeout)
        self.ensure_ready()

    def ensure_ready(self):
        if self.state != "ready":
            raise ServiceError("service_unready", "The matching service is not ready")

    def validate_version(self, version):
        self.ensure_ready()
        if version is not None and version != self.taxonomy_version:
            raise ServiceError(
                "unsupported_taxonomy_version", "The requested taxonomy version is not active", 422
            )

    def health(self):
        return {
            "ready": self.state == "ready",
            "state": self.state,
            "service_mode": "development",
            "taxonomy_key": "google",
            "taxonomy_version": self.taxonomy_version,
            "artifact_version": self.artifact_version,
            "active_requests": int(self.active is not None),
            "pending_requests": len(self.pending),
            "queue_capacity": self.settings.queue_size,
            "semantic_auto_acceptance_enabled": self.policy_enabled,
            "capabilities": {"source_ancestors": True, "provisional_selection": True},
            "process_memory": process_memory(),
        }

    async def submit(self, request, event_id, received_at, *, operation="match"):
        self.ensure_ready()
        if operation == "match":
            self.validate_version(request.taxonomy_version)
        remaining = received_at + self.settings.timeout_seconds - time.perf_counter()
        if remaining <= 0:
            raise ServiceError("request_timeout", "The request deadline expired", 504)
        if self.active is not None and len(self.pending) >= self.settings.queue_size:
            raise ServiceError("queue_full", "The inference queue is full")
        job = Job(
            request,
            event_id,
            received_at,
            time.perf_counter(),
            asyncio.get_running_loop().create_future(),
            operation,
        )
        if self.active is None:
            self._start_job(job)
        else:
            self.pending.append(job)
        try:
            return await asyncio.wait_for(asyncio.shield(job.future), remaining)
        except TimeoutError as error:
            self._abandon(job)
            raise ServiceError("request_timeout", "The request deadline expired", 504) from error
        except asyncio.CancelledError:
            self._abandon(job)
            raise

    def _abandon(self, job):
        if job in self.pending:
            self.pending.remove(job)
        job.future.cancel()

    def _start_job(self, job):
        self.active = job
        self.active_task = asyncio.create_task(self._run_job(job))

    def _perform(self, job):
        started = time.perf_counter()
        payload = job.request
        if job.operation == "feedback":
            try:
                record = submit_feedback(
                    self.runtime,
                    payload.model_dump(),
                    job.event_id,
                    self.categories,
                    self.taxonomy_version,
                )
            except FeedbackError as error:
                raise ServiceError(error.code, str(error), error.status) from error
            return FeedbackResponse.model_validate(record)
        try:
            result = self.matcher.match(
                payload.category or "",
                category_id=payload.category_id,
                parent_id=payload.parent_category_id,
                context=payload.context or "",
                source_ancestors=payload.source_ancestors,
                language=payload.language,
            )
        except ValueError as error:
            raise ServiceError("invalid_category_input", str(error), 422) from error

        def canonical(record):
            stored = self.categories.get(record["id"])
            if stored is None or (record["label"], record["path"]) != (
                stored.label,
                stored.full_path,
            ):
                raise RuntimeError("Noncanonical inference output")
            return CanonicalCategory(
                id=stored.category_id, label=stored.label, path=stored.full_path
            )

        category = canonical(result.category) if result.category is not None else None
        alternatives = (
            [canonical(record) for record in result.candidates[: payload.limit]]
            if (result.status in {"provisional", "needs_clarification"})
            else []
        )
        # Candidates already follow the final neural ranking and specificity safeguards.
        provisional_category = (
            alternatives[0] if result.status == "provisional" and alternatives else None
        )
        if (
            result.artifact_version != self.artifact_version
            or result.taxonomy_version != self.taxonomy_version
            or result.taxonomy_key != "google"
        ):
            raise RuntimeError("Inference mixed artifact or taxonomy versions")
        finished = time.perf_counter()
        response = MatchResponse(
            status=result.status,
            taxonomy_key="google",
            taxonomy_version=self.taxonomy_version,
            artifact_version=self.artifact_version,
            category=category,
            alternatives=alternatives,
            provisional_category=provisional_category,
            requires_review=provisional_category is not None,
            method=result.method,
            reason_code=result.reason_code,
            clarification=result.clarification,
            event_id=job.event_id,
            timings=Timings(
                queue_wait_ms=(started - job.admitted_at) * 1000,
                inference_ms=(finished - started) * 1000,
                total_ms=(finished - job.received_at) * 1000,
            ),
        )
        self.runtime.record_match(payload.model_dump(), response.model_dump())
        return response

    async def _run_job(self, job):
        try:
            response = await asyncio.get_running_loop().run_in_executor(
                self.executor, self._perform, job
            )
            if job.operation == "match":
                log_event(
                    "match_completed",
                    event_id=job.event_id,
                    artifact_version=self.artifact_version,
                    status=response.status,
                    method=response.method,
                    stage="matching",
                    client_waiting=not job.future.done(),
                    **response.timings.model_dump(),
                )
            else:
                log_event(
                    "feedback_recorded",
                    event_id=job.event_id,
                    artifact_version=response.artifact_version,
                    stage="feedback",
                    feedback_id=response.feedback_id,
                    review_status="pending",
                )
            if not job.future.done():
                job.future.set_result(response)
        except ServiceError as error:
            if not job.future.done():
                job.future.set_exception(error)
        except Exception as error:
            logger.error(
                json.dumps(
                    {
                        "event": "inference_failed",
                        "stage": "matching",
                        "event_id": job.event_id,
                        "artifact_version": self.artifact_version,
                        "error_type": type(error).__name__,
                        "traceback": traceback.format_exc(),
                    }
                )
            )
            if self.state == "ready":
                self.state = "failed"
            if not job.future.done():
                job.future.set_exception(
                    ServiceError("inference_failed", "The inference service failed")
                )
            self._reject_pending()
        finally:
            self.active = self.active_task = None
            if self.state == "ready":
                while self.pending:
                    next_job = self.pending.popleft()
                    if not next_job.future.done():
                        if (
                            time.perf_counter()
                            >= next_job.received_at + self.settings.timeout_seconds
                        ):
                            next_job.future.set_exception(
                                ServiceError("request_timeout", "The request deadline expired", 504)
                            )
                            continue
                        self._start_job(next_job)
                        break

    def _reject_pending(self):
        while self.pending:
            job = self.pending.popleft()
            if not job.future.done():
                job.future.set_exception(
                    ServiceError("service_unready", "The matching service is not ready")
                )

    async def close(self):
        if self.closing_task is None:
            self.closing_task = asyncio.create_task(self._close())
        await asyncio.shield(self.closing_task)

    async def _close(self):
        self.state = "stopping"
        self._reject_pending()
        if self.startup_task is not None:
            await asyncio.shield(self.startup_task)
        if self.active_task is not None:
            await asyncio.shield(self.active_task)
        if self.matcher is not None:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.matcher.close)
        self.executor.shutdown(wait=True)
        self.state = "stopped"
