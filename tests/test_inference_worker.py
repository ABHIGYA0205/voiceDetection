"""Deterministic unit tests for InferenceWorker concurrency, cancellation, and shutdown lifecycle."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from services.api.inference_worker import InferenceWorker, InferenceWorkerClosedError


@pytest.mark.asyncio
async def test_worker_cancellation_does_not_release_physical_capacity() -> None:
    """Test A: Cancellation of an awaiting task does not release physical worker capacity.

    A second job submitted after the first was cancelled must remain queued until the
    first physical callable actually returns. Maximum physical concurrency must strictly remain 1.
    """
    worker = InferenceWorker()
    job1_started = threading.Event()
    job1_release = threading.Event()
    job2_started = threading.Event()

    active_count = 0
    max_concurrency = 0
    count_lock = threading.Lock()

    def job1() -> str:
        nonlocal active_count, max_concurrency
        with count_lock:
            active_count += 1
            max_concurrency = max(max_concurrency, active_count)
        job1_started.set()
        job1_release.wait(timeout=5.0)
        with count_lock:
            active_count -= 1
        return "job1_done"

    def job2() -> str:
        nonlocal active_count, max_concurrency
        with count_lock:
            active_count += 1
            max_concurrency = max(max_concurrency, active_count)
        job2_started.set()
        with count_lock:
            active_count -= 1
        return "job2_done"

    # Start job1 in a background task
    task1 = asyncio.create_task(worker.execute(job1))

    # Wait for job1 to physically start running in the worker thread
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: job1_started.wait(timeout=5.0))
    assert job1_started.is_set(), "Job 1 failed to signal start"

    with count_lock:
        assert active_count == 1
        assert max_concurrency == 1

    # Cancel the task awaiting job1
    task1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task1

    # Submit job2 while job1 is still blocked on job1_release
    task2 = asyncio.create_task(worker.execute(job2))

    # Give the event loop a few turns to attempt dispatch
    await asyncio.sleep(0.05)

    # Job 2 MUST NOT have started because job 1 still physically occupies the worker
    assert not job2_started.is_set()
    with count_lock:
        assert active_count == 1
        assert max_concurrency == 1
    assert worker.running_job_count == 1

    # Now release job 1
    job1_release.set()

    # Job 2 should now run and complete
    result2 = await task2
    assert result2 == "job2_done"
    assert job2_started.is_set()

    with count_lock:
        assert active_count == 0
        assert max_concurrency == 1

    await worker.shutdown(timeout=2.0)


@pytest.mark.asyncio
async def test_worker_queued_cancellation() -> None:
    """Test B: A queued job cancelled before execution must never be executed."""
    worker = InferenceWorker()
    job1_started = threading.Event()
    job1_release = threading.Event()
    job2_executed = threading.Event()

    def job1() -> str:
        job1_started.set()
        job1_release.wait(timeout=5.0)
        return "job1_done"

    def job2() -> str:
        job2_executed.set()
        return "job2_done"

    task1 = asyncio.create_task(worker.execute(job1))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: job1_started.wait(timeout=5.0))

    # Submit job2 while job1 is running; job2 is queued
    task2 = asyncio.create_task(worker.execute(job2))
    await asyncio.sleep(0.02)
    assert not job2_executed.is_set()

    # Cancel job2 while still queued
    task2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task2

    # Release job1
    job1_release.set()
    await task1

    # Give worker time to process next item in queue (which was cancelled)
    await asyncio.sleep(0.05)
    assert not job2_executed.is_set(), "Cancelled queued job was executed unexpectedly"

    await worker.shutdown(timeout=2.0)


@pytest.mark.asyncio
async def test_worker_thread_identity() -> None:
    """Test C: All jobs execute on a single dedicated thread, distinct from the loop thread."""
    worker = InferenceWorker()
    event_loop_thread_id = threading.get_ident()
    thread_ids: list[int] = []

    def job() -> int:
        tid = threading.get_ident()
        thread_ids.append(tid)
        return tid

    for _ in range(5):
        tid = await worker.execute(job)
        assert tid != event_loop_thread_id

    # All calls executed on the same worker thread
    assert len(set(thread_ids)) == 1
    assert worker.worker_thread_id == thread_ids[0]

    await worker.shutdown(timeout=2.0)


@pytest.mark.asyncio
async def test_worker_shutdown_drain() -> None:
    """Test D: Shutdown rejects new submissions and drains active jobs."""
    worker = InferenceWorker()
    job_started = threading.Event()
    job_release = threading.Event()
    job_completed = False

    def controlled_job() -> str:
        nonlocal job_completed
        job_started.set()
        job_release.wait(timeout=5.0)
        job_completed = True
        return "completed"

    task = asyncio.create_task(worker.execute(controlled_job))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: job_started.wait(timeout=5.0))

    # Stop worker intake
    worker.stop_intake()

    # Immediate rejection of new jobs
    assert not worker.accepting_jobs
    with pytest.raises(InferenceWorkerClosedError):
        await worker.execute(lambda: "rejected")

    # Release running job
    job_release.set()
    await task
    assert job_completed

    # Shutdown should drain and complete cleanly
    drained = await worker.shutdown(timeout=2.0)
    assert drained is True
    assert worker.is_shutdown is True
    assert worker.outstanding_job_count == 0


@pytest.mark.asyncio
async def test_worker_shutdown_timeout_truthfulness(caplog: pytest.LogCaptureFixture) -> None:
    """Test E: Shutdown accurately reports False and logs warning when jobs remain blocked."""
    worker = InferenceWorker()
    job_started = threading.Event()
    job_release = threading.Event()

    def blocked_job() -> str:
        job_started.set()
        job_release.wait(timeout=5.0)
        return "done"

    task = asyncio.create_task(worker.execute(blocked_job))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: job_started.wait(timeout=5.0))

    try:
        with caplog.at_level(logging.WARNING):
            drained = await worker.shutdown(timeout=0.1)

        # Truthful reporting: did NOT drain within 0.1s
        assert drained is False
        assert any("shutdown timeout" in rec.message for rec in caplog.records)
    finally:
        # Cleanup: release job so thread exits cleanly and test never leaks
        job_release.set()
        await task
        # Final cleanup via public shutdown API (no direct _executor access)
        drained_final = await worker.shutdown(timeout=2.0)
        assert drained_final is True


# ==============================================================================
# Regression Tests: Disconnect Inference Lifecycle Synchronization (R5)
# ==============================================================================


@pytest.mark.asyncio
async def test_active_plus_queued_shutdown_no_deadlock() -> None:
    """Test A: Active plus queued shutdown deadlock regression.

    Job 1 starts and blocks on threading.Event.
    Job 2 is submitted and confirmed queued/tracked.
    Initiate stop_intake or shutdown.
    Prove the operation returns/progresses rather than deadlocking.
    Prove job 2 is cancelled and never physically executes.
    Release job 1.
    Prove clean drain and shutdown.
    """
    worker = InferenceWorker()
    job1_started = threading.Event()
    job1_release = threading.Event()
    job2_started = threading.Event()

    def job1() -> str:
        job1_started.set()
        job1_release.wait(timeout=5.0)
        return "job1_done"

    def job2() -> str:
        job2_started.set()
        return "job2_done"

    task1 = asyncio.create_task(worker.execute(job1))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: job1_started.wait(timeout=5.0))
    assert job1_started.is_set()

    task2 = asyncio.create_task(worker.execute(job2))
    # Give event loop a turn so task2 submits job2 to worker
    await asyncio.sleep(0.02)
    assert worker.outstanding_job_count == 2
    assert worker.running_job_count == 1
    assert not job2_started.is_set()

    # Initiate stop_intake with finite timeout.
    # Before the fix, stop_intake deadlocked because fut.cancel() synchronously called
    # _on_future_done which tried to reacquire self._lock.
    await asyncio.wait_for(asyncio.to_thread(worker.stop_intake), timeout=2.0)

    # Prove job 2 was cancelled and never physically executes
    with pytest.raises(asyncio.CancelledError):
        await task2
    assert not job2_started.is_set()

    # Release job 1 and verify clean drain and shutdown
    job1_release.set()
    res1 = await task1
    assert res1 == "job1_done"

    drained = await worker.shutdown(timeout=2.0)
    assert drained is True
    assert worker.is_shutdown is True
    assert worker.outstanding_job_count == 0
    assert worker.running_job_count == 0


def test_active_plus_queued_deadlock_subprocess_isolation() -> None:
    """Test A (Subprocess Isolation): Active plus queued shutdown deadlock regression."""
    code = """
import asyncio
import sys
import threading
from services.api.inference_worker import InferenceWorker

async def main():
    w = InferenceWorker()
    job1_started = threading.Event()
    job1_release = threading.Event()
    job2_started = threading.Event()

    def job1():
        job1_started.set()
        job1_release.wait(timeout=5.0)
        return "j1"

    def job2():
        job2_started.set()
        return "j2"

    t1 = asyncio.create_task(w.execute(job1))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: job1_started.wait(timeout=5.0))

    t2 = asyncio.create_task(w.execute(job2))
    await asyncio.sleep(0.05)
    assert w.outstanding_job_count == 2
    assert w.running_job_count == 1

    # stop_intake must return immediately without deadlocking
    w.stop_intake()

    job1_release.set()
    await t1
    try:
        await t2
    except asyncio.CancelledError:
        pass
    assert not job2_started.is_set()

    drained = await w.shutdown(timeout=2.0)
    assert drained is True
    assert w.outstanding_job_count == 0
    sys.exit(0)

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        timeout=10.0,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Subprocess failed or deadlocked: {result.stderr}"


@pytest.mark.asyncio
async def test_atomic_submit_versus_stop_race_submission_wins() -> None:
    """Test B: Atomic submit-versus-stop race where submission wins lock first.

    The future is registered in _active_futures before stop_intake completes,
    so stop_intake sees and cancels it.
    """
    worker = InferenceWorker()
    callable_executed = False

    def sample_callable() -> str:
        nonlocal callable_executed
        callable_executed = True
        return "result"

    # Block worker with a running job
    job1_started = threading.Event()
    job1_release = threading.Event()

    def blocking_job() -> None:
        job1_started.set()
        job1_release.wait(5.0)

    task1 = asyncio.create_task(worker.execute(blocking_job))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: job1_started.wait(timeout=5.0))

    # Hook submit to trigger stop_intake from another thread while inside submit (under lock)
    orig_submit = worker._executor.submit
    stop_called = threading.Event()

    def stop_worker_thread() -> None:
        worker.stop_intake()
        stop_called.set()

    def wrapped_submit(fn: Any, *args: Any, **kwargs: Any) -> Any:
        fut = orig_submit(fn, *args, **kwargs)
        t = threading.Thread(target=stop_worker_thread)
        t.start()
        # Small delay to ensure the stop thread is waiting on worker._lock
        time.sleep(0.02)
        return fut

    worker._executor.submit = wrapped_submit  # type: ignore[assignment]

    task2 = asyncio.create_task(worker.execute(sample_callable))
    await loop.run_in_executor(None, lambda: stop_called.wait(timeout=5.0))

    # Submission won the lock before stop_intake; the future was tracked and then cancelled by stop_intake
    with pytest.raises(asyncio.CancelledError):
        await task2

    assert not callable_executed, "Queued callable must not execute after stop_intake"

    job1_release.set()
    await task1
    drained = await worker.shutdown(timeout=2.0)
    assert drained is True


@pytest.mark.asyncio
async def test_atomic_submit_versus_stop_race_stop_wins() -> None:
    """Test B: Atomic submit-versus-stop race where stop_intake wins lock first.

    worker.execute raises InferenceWorkerClosedError and callable never executes.
    """
    worker = InferenceWorker()
    callable_executed = False

    def sample_callable() -> str:
        nonlocal callable_executed
        callable_executed = True
        return "result"

    worker.stop_intake()
    assert not worker.accepting_jobs

    with pytest.raises(InferenceWorkerClosedError):
        await worker.execute(sample_callable)

    assert not callable_executed
    drained = await worker.shutdown(timeout=2.0)
    assert drained is True


@pytest.mark.asyncio
async def test_submit_runtime_error_translated_to_worker_closed() -> None:
    """Test B: If submit races with executor shutdown, RuntimeError translates to InferenceWorkerClosedError."""
    worker = InferenceWorker()
    # Shut down underlying executor directly to force submit RuntimeError
    worker._executor.shutdown(wait=True)

    with pytest.raises(InferenceWorkerClosedError) as exc_info:
        await worker.execute(lambda: "test")
    assert "shutting down or closed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_detached_exception_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Test C: Detached exception logging with DETACHED_BOOM and caplog.

    - running callable blocks;
    - awaiting coroutine is cancelled;
    - callable is released and raises RuntimeError("DETACHED_BOOM");
    - log contains DETACHED_BOOM;
    - no unhandled-future warning occurs;
    - worker remains usable.
    """
    worker = InferenceWorker()
    job_started = threading.Event()
    job_release = threading.Event()

    def failing_job() -> None:
        job_started.set()
        job_release.wait(timeout=5.0)
        raise RuntimeError("DETACHED_BOOM")

    task = asyncio.create_task(worker.execute(failing_job))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: job_started.wait(timeout=5.0))

    # Cancel awaiting coroutine
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Physical job is still running; release it to raise DETACHED_BOOM
    with caplog.at_level(logging.ERROR):
        job_release.set()
        # Poll until the physical job has finished
        for _ in range(50):
            if worker.outstanding_job_count == 0 and worker.running_job_count == 0:
                break
            await asyncio.sleep(0.02)

    assert "DETACHED_BOOM" in caplog.text
    assert "InferenceWorker physical job failed with unhandled exception" in caplog.text

    # Worker remains usable for subsequent jobs
    healthy_result = await worker.execute(lambda: "healthy_again")
    assert healthy_result == "healthy_again"

    drained = await worker.shutdown(timeout=2.0)
    assert drained is True


@pytest.mark.asyncio
async def test_repeated_shutdown_after_timeout() -> None:
    """Test D: Repeated shutdown after timeout.

    - Running job exceeds first short timeout.
    - first shutdown returns False;
    - worker remains non-accepting and reports one outstanding/running job;
    - release job;
    - second public shutdown returns True;
    - outstanding/running counts are zero;
    - do not access _executor directly.
    """
    worker = InferenceWorker()
    job_started = threading.Event()
    job_release = threading.Event()

    def long_job() -> str:
        job_started.set()
        job_release.wait(timeout=5.0)
        return "long_job_complete"

    task = asyncio.create_task(worker.execute(long_job))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: job_started.wait(timeout=5.0))

    # First shutdown with very short timeout times out
    drained1 = await worker.shutdown(timeout=0.05)
    assert drained1 is False
    assert worker.accepting_jobs is False
    assert worker.is_shutdown is False
    assert worker.outstanding_job_count == 1
    assert worker.running_job_count == 1

    # Release running job
    job_release.set()
    res = await task
    assert res == "long_job_complete"

    # Wait for completion callback
    for _ in range(50):
        if worker.outstanding_job_count == 0 and worker.running_job_count == 0:
            break
        await asyncio.sleep(0.01)

    assert worker.outstanding_job_count == 0
    assert worker.running_job_count == 0

    # Second public shutdown returns True
    drained2 = await worker.shutdown(timeout=2.0)
    assert drained2 is True
    assert worker.is_shutdown is True
    assert worker.outstanding_job_count == 0
    assert worker.running_job_count == 0

    # Repeated shutdown call returns True immediately
    drained3 = await worker.shutdown(timeout=1.0)
    assert drained3 is True


@pytest.mark.asyncio
async def test_worker_shutdown_timeout_validation() -> None:
    """Test E: Assert negative, NaN, and infinite timeout values are rejected."""
    worker = InferenceWorker()

    # Negative timeout
    with pytest.raises(ValueError, match="Shutdown timeout must be a non-negative finite float"):
        await worker.shutdown(timeout=-1.0)

    # NaN timeout
    with pytest.raises(ValueError, match="Shutdown timeout must be a non-negative finite float"):
        await worker.shutdown(timeout=float("nan"))

    # Inf timeout
    with pytest.raises(ValueError, match="Shutdown timeout must be a non-negative finite float"):
        await worker.shutdown(timeout=float("inf"))

    # shutdown_sync validation
    with pytest.raises(ValueError, match="Shutdown timeout must be a non-negative finite float"):
        worker.shutdown_sync(timeout=-2.0)

    with pytest.raises(ValueError, match="Shutdown timeout must be a non-negative finite float"):
        worker.shutdown_sync(timeout=float("nan"))

    with pytest.raises(ValueError, match="Shutdown timeout must be a non-negative finite float"):
        worker.shutdown_sync(timeout=float("inf"))

    drained = await worker.shutdown(timeout=1.0)
    assert drained is True


@pytest.mark.asyncio
async def test_main_lifespan_shutdown_timeout_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test E: Main lifespan safely parses VOICESHIELD_INFERENCE_SHUTDOWN_TIMEOUT with fallback."""
    from fastapi import FastAPI

    from services.api.main import lifespan

    test_app = FastAPI()

    # Test invalid string value
    monkeypatch.setenv("VOICESHIELD_INFERENCE_SHUTDOWN_TIMEOUT", "invalid_value")
    with caplog.at_level(logging.WARNING):
        async with lifespan(test_app):
            pass
    assert any(
        "Invalid VOICESHIELD_INFERENCE_SHUTDOWN_TIMEOUT" in rec.message for rec in caplog.records
    )

    # Test negative float value
    caplog.clear()
    monkeypatch.setenv("VOICESHIELD_INFERENCE_SHUTDOWN_TIMEOUT", "-5.0")
    with caplog.at_level(logging.WARNING):
        async with lifespan(test_app):
            pass
    assert any("must be finite and >= 0" in rec.message for rec in caplog.records)
