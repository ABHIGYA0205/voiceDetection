"""Process-wide, lifecycle-owned single-worker inference execution service.

Replaces cancellation-sensitive asyncio.Semaphore / asyncio.to_thread with a dedicated
single-thread executor that serializes all physical model inference across HTTP and
WebSocket sessions, ensuring strict mutual exclusion even across request cancellations.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger("voiceshield.api.inference_worker")

P = ParamSpec("P")
T = TypeVar("T")


class InferenceWorkerClosedError(RuntimeError):
    """Raised when submitting an inference job to a closed or shutting-down worker."""


class InferenceWorker:
    """Dedicated single-worker execution service for synchronous model inference.

    Invariants:
    1. Exactly one physical synchronous inference callable may execute at a time.
    2. Request/session coroutine cancellation does NOT release physical execution capacity.
    3. Queued jobs submitted after a cancelled caller remain queued until the physically
       running job finishes.
    4. Worker lifecycle is owned by application lifespan, independent of client connections.
    """

    def __init__(self, thread_name_prefix: str = "voiceshield-inference") -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=thread_name_prefix,
        )
        self._lock = threading.Lock()
        self._running_lock = threading.Lock()
        self._active_futures: set[concurrent.futures.Future[Any]] = set()
        self._running_jobs: int = 0
        self._is_shutting_down: bool = False
        self._is_shutdown: bool = False
        self._worker_thread_id: int | None = None

    @property
    def accepting_jobs(self) -> bool:
        """Return True if the worker is accepting new inference jobs."""
        with self._lock:
            return not self._is_shutting_down and not self._is_shutdown

    @property
    def outstanding_job_count(self) -> int:
        """Return count of unresolved jobs (both queued and physically executing)."""
        with self._lock:
            return len([f for f in self._active_futures if not f.done()])

    @property
    def running_job_count(self) -> int:
        """Return count of jobs currently physically executing in the worker thread (0 or 1)."""
        with self._running_lock:
            return self._running_jobs

    @property
    def is_shutdown(self) -> bool:
        """Return True if the worker has completed shutdown."""
        with self._lock:
            return self._is_shutdown

    @property
    def worker_thread_id(self) -> int | None:
        """Return thread identifier of the dedicated inference worker thread, if active."""
        with self._running_lock:
            return self._worker_thread_id

    async def execute(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Execute a synchronous callable on the dedicated single-thread worker.

        If the calling coroutine is cancelled while awaiting:
        - The underlying concurrent.futures.Future is cancelled if still queued.
        - If execution has already physically begun, the job remains tracked and occupies
          the sole worker thread until completion.
        - CancelledError is re-raised to allow prompt caller unwinding without freeing the worker.
        """
        with self._lock:
            if self._is_shutting_down or self._is_shutdown:
                raise InferenceWorkerClosedError(
                    "Inference worker is closed and no longer accepting jobs."
                )

        loop = asyncio.get_running_loop()

        def _worker_wrapper() -> T:
            with self._running_lock:
                self._running_jobs += 1
                self._worker_thread_id = threading.get_ident()
            try:
                return fn(*args, **kwargs)
            finally:
                with self._running_lock:
                    self._running_jobs -= 1

        concurrent_future = self._executor.submit(_worker_wrapper)

        with self._lock:
            self._active_futures.add(concurrent_future)

        def _on_future_done(f: concurrent.futures.Future[Any]) -> None:
            with self._lock:
                self._active_futures.discard(f)
            # Retrieve exception to avoid unretrieved future exception warnings
            try:
                f.exception()
            except (concurrent.futures.CancelledError, Exception) as exc:
                logger.debug("Inference worker physical job finished with: %s", exc)

        concurrent_future.add_done_callback(_on_future_done)

        asyncio_future = asyncio.wrap_future(concurrent_future, loop=loop)

        def _on_asyncio_done(f: asyncio.Future[Any]) -> None:
            try:
                f.exception()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                pass
            except Exception as exc:
                logger.debug("Inference worker asyncio wrapper finished with: %s", exc)

        asyncio_future.add_done_callback(_on_asyncio_done)

        try:
            return await asyncio.shield(asyncio_future)
        except asyncio.CancelledError:
            # Caller was cancelled. Attempt to cancel queued future.
            cancelled = concurrent_future.cancel()
            if not cancelled:
                logger.info(
                    "Awaiting coroutine cancelled, but physical inference job has already started "
                    "and will occupy the single worker until completion."
                )
            raise

    async def submit(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Alias for execute to support alternative invocation styles."""
        return await self.execute(fn, *args, **kwargs)

    def stop_intake(self) -> None:
        """Atomically stop accepting new jobs and cancel queued jobs."""
        with self._lock:
            if not self._is_shutdown:
                self._is_shutting_down = True
                for fut in list(self._active_futures):
                    fut.cancel()

    async def shutdown(self, timeout: float = 5.0) -> bool:
        """Stop accepting new jobs, cancel queued jobs, and drain running jobs up to timeout.

        Args:
            timeout: Maximum seconds to await completion of physically running jobs.

        Returns:
            True if all jobs drained cleanly; False if timeout expired with jobs still running.
        """
        with self._lock:
            if self._is_shutdown:
                return True
        self.stop_intake()

        start_time = time.monotonic()
        drained = True

        while True:
            with self._lock:
                pending_futures = [f for f in self._active_futures if not f.done()]
            if not pending_futures:
                break
            elapsed = time.monotonic() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                drained = False
                logger.warning(
                    "InferenceWorker shutdown timeout (%.2fs) expired with %d job(s) still physically executing.",
                    timeout,
                    len(pending_futures),
                )
                break
            await asyncio.sleep(min(0.05, max(0.005, remaining)))

        if drained:
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
        else:
            # Avoid blocking the event loop on an unkillable native thread
            self._executor.shutdown(wait=False, cancel_futures=True)

        with self._lock:
            self._is_shutdown = True

        return drained

    def shutdown_sync(self, timeout: float = 5.0) -> bool:
        """Synchronously drain and shut down the worker for non-async testing."""
        with self._lock:
            if self._is_shutdown:
                return True
            self._is_shutting_down = True
            for fut in list(self._active_futures):
                fut.cancel()

        start_time = time.monotonic()
        drained = True

        while True:
            with self._lock:
                pending_futures = [f for f in self._active_futures if not f.done()]
            if not pending_futures:
                break
            elapsed = time.monotonic() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                drained = False
                logger.warning(
                    "InferenceWorker shutdown_sync timeout (%.2fs) expired with %d job(s) still executing.",
                    timeout,
                    len(pending_futures),
                )
                break
            time.sleep(min(0.05, max(0.005, remaining)))

        if drained:
            self._executor.shutdown(wait=True, cancel_futures=True)
        else:
            self._executor.shutdown(wait=False, cancel_futures=True)

        with self._lock:
            self._is_shutdown = True

        return drained
