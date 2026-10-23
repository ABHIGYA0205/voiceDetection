"""Deterministic unit tests for AtomicAdmissionController and FileUploadAdmissionMiddleware."""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import MutableMapping
from typing import Any

import pytest

from services.api.resource_admission import (
    MAX_UPLOAD_REQUEST_BYTES,
    AtomicAdmissionController,
    FileUploadAdmissionMiddleware,
)

# ==============================================================================
# 1. AtomicAdmissionController Unit Tests
# ==============================================================================


def test_capacity_and_unique_token_behavior() -> None:
    """Test A: Capacity and unique-token reservation behavior.

    - reserve exactly limit unique tokens;
    - next reservation fails;
    - duplicate token cannot consume a second slot;
    - unknown release changes nothing;
    - valid release frees exactly one slot.
    """
    controller = AtomicAdmissionController(limit=3, name="test_resource")
    assert controller.limit == 3
    assert controller.name == "test_resource"
    assert controller.accepting is True
    assert controller.reserved_count == 0
    assert controller.available_count == 3

    # Reserve up to capacity
    assert controller.try_reserve("token_1") is True
    assert controller.try_reserve("token_2") is True
    assert controller.try_reserve("token_3") is True
    assert controller.reserved_count == 3
    assert controller.available_count == 0

    # Next unique reservation must fail
    assert controller.try_reserve("token_4") is False
    assert controller.reserved_count == 3

    # Duplicate token cannot consume a second slot
    assert controller.try_reserve("token_1") is False
    assert controller.reserved_count == 3

    # Unknown release changes nothing
    assert controller.release("unknown_token") is False
    assert controller.reserved_count == 3
    assert controller.available_count == 0

    # Valid release frees exactly one slot
    assert controller.release("token_2") is True
    assert controller.reserved_count == 2
    assert controller.available_count == 1

    # Now a new reservation succeeds
    assert controller.try_reserve("token_5") is True
    assert controller.reserved_count == 3
    assert controller.available_count == 0


def test_invalid_limit_raises_value_error() -> None:
    """Verify limit must be a positive integer."""
    with pytest.raises(ValueError, match="must be a positive integer"):
        AtomicAdmissionController(limit=0)

    with pytest.raises(ValueError, match="must be a positive integer"):
        AtomicAdmissionController(limit=-5)


def test_concurrent_reservation_stress() -> None:
    """Test B: Concurrent reservation stress.

    - many threads race for a small limit;
    - exactly limit reservations succeed;
    - reserved_count never exceeds limit;
    - all successful tokens release back to zero.
    """
    limit = 5
    controller = AtomicAdmissionController(limit=limit, name="stress_resource")
    num_threads = 50
    successful_tokens: list[str] = []
    lock = threading.Lock()

    def attempt_reservation(thread_id: int) -> None:
        token = f"thread_token_{thread_id}"
        if controller.try_reserve(token):
            with lock:
                successful_tokens.append(token)
                # Invariant: reserved_count must never exceed limit
                assert controller.reserved_count <= limit

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(attempt_reservation, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    # Exactly limit reservations must succeed
    assert len(successful_tokens) == limit
    assert controller.reserved_count == limit
    assert controller.available_count == 0

    # Release all successful tokens
    for token in successful_tokens:
        assert controller.release(token) is True

    assert controller.reserved_count == 0
    assert controller.available_count == limit


def test_close_intake() -> None:
    """Test C: Close intake semantics.

    - existing tokens remain counted;
    - new reservations fail;
    - existing releases work;
    - count reaches zero.
    """
    controller = AtomicAdmissionController(limit=2, name="intake_resource")
    assert controller.try_reserve("tok_a") is True
    assert controller.reserved_count == 1

    # Close intake
    controller.close_intake()
    assert controller.accepting is False

    # Existing tokens remain counted
    assert controller.reserved_count == 1

    # New reservations must fail even though available_count > 0
    assert controller.try_reserve("tok_b") is False
    assert controller.reserved_count == 1

    # Existing releases still work cleanly
    assert controller.release("tok_a") is True
    assert controller.reserved_count == 0
    assert controller.release("tok_a") is False


# ==============================================================================
# 2. FileUploadAdmissionMiddleware Unit Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_middleware_bypasses_non_target_requests() -> None:
    """Verify middleware does not consume slots for GET or unrelated routes."""
    controller = AtomicAdmissionController(limit=1, name="uploads")
    downstream_called = False

    async def mock_app(scope: Any, receive: Any, send: Any) -> None:
        nonlocal downstream_called
        downstream_called = True

    async def dummy_receive() -> MutableMapping[str, Any]:
        return {}

    async def dummy_send(m: MutableMapping[str, Any]) -> None:
        pass

    middleware = FileUploadAdmissionMiddleware(mock_app, controller=controller)

    # GET request
    scope: dict[str, Any] = {"type": "http", "method": "GET", "path": "/health"}
    await middleware(scope, dummy_receive, dummy_send)
    assert downstream_called is True
    assert controller.reserved_count == 0

    # Non-HTTP scope (e.g. websocket)
    downstream_called = False
    scope = {"type": "websocket", "path": "/api/v1/stream"}
    await middleware(scope, dummy_receive, dummy_send)
    assert downstream_called is True
    assert controller.reserved_count == 0


@pytest.mark.asyncio
async def test_middleware_rejects_when_admission_full() -> None:
    """Test 429 when upload admission slots are exhausted; receive is never called."""
    controller = AtomicAdmissionController(limit=1, name="uploads")
    assert controller.try_reserve("existing_upload") is True
    assert controller.reserved_count == 1

    receive_called = False
    downstream_called = False

    async def mock_receive() -> MutableMapping[str, Any]:
        nonlocal receive_called
        receive_called = True
        return {"type": "http.request", "body": b"", "more_body": False}

    sent_messages: list[MutableMapping[str, Any]] = []

    async def mock_send(msg: MutableMapping[str, Any]) -> None:
        sent_messages.append(msg)

    async def mock_app(scope: Any, receive: Any, send: Any) -> None:
        nonlocal downstream_called
        downstream_called = True

    middleware = FileUploadAdmissionMiddleware(mock_app, controller=controller)
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/analyze/file",
        "headers": [(b"content-length", b"100")],
    }

    await middleware(scope, mock_receive, mock_send)

    assert downstream_called is False
    assert receive_called is False

    # Check 429 response
    start_msg = next((m for m in sent_messages if m["type"] == "http.response.start"), None)
    assert start_msg is not None
    assert start_msg["status"] == 429
    headers = dict(start_msg["headers"])
    assert headers.get(b"retry-after") == b"5"


@pytest.mark.asyncio
async def test_middleware_rejects_content_length_overflow() -> None:
    """Test 413 when Content-Length exceeds ceiling; receive is never called."""
    controller = AtomicAdmissionController(limit=2, name="uploads")
    receive_called = False
    downstream_called = False

    async def mock_receive() -> MutableMapping[str, Any]:
        nonlocal receive_called
        receive_called = True
        return {"type": "http.request", "body": b"", "more_body": False}

    sent_messages: list[MutableMapping[str, Any]] = []

    async def mock_send(msg: MutableMapping[str, Any]) -> None:
        sent_messages.append(msg)

    async def mock_app(scope: Any, receive: Any, send: Any) -> None:
        nonlocal downstream_called
        downstream_called = True

    middleware = FileUploadAdmissionMiddleware(mock_app, controller=controller)
    overflow_len = str(MAX_UPLOAD_REQUEST_BYTES + 1024).encode("ascii")
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/analyze/file",
        "headers": [(b"content-length", overflow_len)],
    }

    await middleware(scope, mock_receive, mock_send)

    assert downstream_called is False
    assert receive_called is False
    assert controller.reserved_count == 0  # Reservation released in finally

    start_msg = next((m for m in sent_messages if m["type"] == "http.response.start"), None)
    assert start_msg is not None
    assert start_msg["status"] == 413


@pytest.mark.asyncio
async def test_middleware_rejects_streaming_chunks_overflow() -> None:
    """Test 413 when chunked/streamed body cumulative size exceeds ceiling."""
    controller = AtomicAdmissionController(limit=2, name="uploads")
    sent_messages: list[MutableMapping[str, Any]] = []

    async def mock_send(msg: MutableMapping[str, Any]) -> None:
        sent_messages.append(msg)

    async def mock_app(scope: Any, receive: Any, send: Any) -> None:
        # Downstream attempts to consume body
        while True:
            msg = await receive()
            if not msg.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    # Set a small test cap to verify chunk streaming behavior
    small_cap = 500
    middleware = FileUploadAdmissionMiddleware(
        mock_app, controller=controller, max_upload_bytes=small_cap
    )

    chunks: list[dict[str, Any]] = [
        {"type": "http.request", "body": b"x" * 300, "more_body": True},
        {"type": "http.request", "body": b"x" * 300, "more_body": False},  # Cumulative 600 > 500
    ]
    chunk_idx = 0

    async def mock_receive() -> MutableMapping[str, Any]:
        nonlocal chunk_idx
        if chunk_idx < len(chunks):
            c = chunks[chunk_idx]
            chunk_idx += 1
            return c
        return {"type": "http.request", "body": b"", "more_body": False}

    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/analyze/file",
        "headers": [],  # Missing Content-Length
    }

    await middleware(scope, mock_receive, mock_send)

    assert controller.reserved_count == 0

    start_msg = next((m for m in sent_messages if m["type"] == "http.response.start"), None)
    assert start_msg is not None
    assert start_msg["status"] == 413
