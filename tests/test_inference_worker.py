"""Deterministic unit tests for InferenceWorker concurrency, cancellation, and shutdown lifecycle."""

from __future__ import annotations

import asyncio
import logging
import threading

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
        # Final cleanup shutdown
        await asyncio.to_thread(worker._executor.shutdown, wait=True)
