"""Process-wide atomic admission control and request-resource limiting for VoiceShield.

Provides thread-safe slot reservation for WebSocket streams and offline file uploads,
as well as ASGI-level request body size enforcement before multipart parsing.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any

from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from voiceshield.core.schema import Disclaimers

logger = logging.getLogger("voiceshield.api.admission")

# Resource constants
MAX_FILE_BYTES: int = 25 * 1024 * 1024  # 25 MB audio payload
MAX_UPLOAD_REQUEST_OVERHEAD_BYTES: int = (
    1 * 1024 * 1024
)  # 1 MB multipart boundary & headers overhead
MAX_UPLOAD_REQUEST_BYTES: int = (
    MAX_FILE_BYTES + MAX_UPLOAD_REQUEST_OVERHEAD_BYTES
)  # 26 MB total ceiling
MAX_CONCURRENT_SESSIONS_DEFAULT: int = 10
MAX_CONCURRENT_FILE_UPLOADS_DEFAULT: int = 2


class ApiErrorDetail(BaseModel):
    """Structured error payload for API rejections."""

    error_code: str = Field(description="Machine-readable error classification.")
    message: str = Field(description="Human-readable explanation of rejection.")
    details: dict[str, Any] = Field(default_factory=dict, description="Diagnostic details.")
    disclaimers: Disclaimers = Field(default_factory=Disclaimers)


class PayloadTooLargeError(Exception):
    """Internal exception raised when streaming request body exceeds configured ceiling."""

    def __init__(self, bytes_received: int) -> None:
        self.bytes_received = bytes_received
        super().__init__(f"Request body exceeded maximum limit ({bytes_received} bytes received)")


class AtomicAdmissionController:
    """Thread-safe, process-wide atomic admission controller for bounded resource slots.

    Guarantees:
    1. Simultaneous reservation requests cannot exceed the configured limit.
    2. Releasing a token is idempotent and duplicate/unknown releases do not alter counts.
    3. Closing intake prevents new reservations while allowing existing reservations to release.
    4. Short lock-protected operations; no external blocking or I/O while holding the lock.
    """

    def __init__(self, limit: int, name: str = "resource") -> None:
        if limit < 1:
            raise ValueError(
                f"Admission limit for '{name}' must be a positive integer, got {limit}"
            )
        self._limit = limit
        self._name = name
        self._lock = threading.Lock()
        self._accepting = True
        self._reserved_tokens: set[str] = set()

    @property
    def name(self) -> str:
        """Name of the resource being governed."""
        return self._name

    @property
    def limit(self) -> int:
        """Configured maximum concurrent reservations."""
        return self._limit

    @property
    def accepting(self) -> bool:
        """Return True if the controller is currently accepting new reservations."""
        with self._lock:
            return self._accepting

    @property
    def reserved_count(self) -> int:
        """Return current count of reserved tokens."""
        with self._lock:
            return len(self._reserved_tokens)

    @property
    def available_count(self) -> int:
        """Return count of remaining available reservation slots."""
        with self._lock:
            return max(0, self._limit - len(self._reserved_tokens))

    def try_reserve(self, token: str) -> bool:
        """Atomically attempt to reserve a slot using a unique token.

        Returns True if reservation was granted, False otherwise.
        """
        with self._lock:
            if not self._accepting:
                return False
            if token in self._reserved_tokens:
                # Token already holds a reservation; duplicate attempt does not consume a second slot
                return False
            if len(self._reserved_tokens) >= self._limit:
                return False
            self._reserved_tokens.add(token)
            return True

    def release(self, token: str) -> bool:
        """Release a previously reserved token.

        Returns True if the token was found and released, False if unknown or already released.
        """
        with self._lock:
            if token in self._reserved_tokens:
                self._reserved_tokens.remove(token)
                return True
            return False

    def close_intake(self) -> None:
        """Stop accepting new reservations while allowing existing ones to release."""
        with self._lock:
            self._accepting = False


async def _send_json_response(
    send: Send,
    status_code: int,
    payload: dict[str, Any],
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    """Send structured JSON response over raw ASGI send channel."""
    body = json.dumps(payload).encode("utf-8")
    resp_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if headers:
        resp_headers.extend(headers)

    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": resp_headers,
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        }
    )


class FileUploadAdmissionMiddleware:
    """ASGI middleware enforcing file upload admission and body sizing before multipart parsing.

    Guarantees:
    1. Limits concurrent POST requests to /api/v1/analyze/file before body parsing or spooling.
    2. Rejects excess concurrent requests with HTTP 429 and Retry-After header.
    3. Enforces a hard total request body cap in two layers:
       - Content-Length precheck before reading body
       - Streaming receive counter for chunked/missing Content-Length
    4. Rejects oversized request bodies with HTTP 413 without generating generic 500 errors.
    5. Cleanly releases admission tokens in all completion, error, and cancellation paths.
    """

    def __init__(
        self,
        app: ASGIApp,
        controller: AtomicAdmissionController | None = None,
        max_upload_bytes: int = MAX_UPLOAD_REQUEST_BYTES,
    ) -> None:
        self.app = app
        self._controller = controller
        self._max_upload_bytes = max_upload_bytes

    def _get_controller(self, scope: Scope) -> AtomicAdmissionController | None:
        if self._controller is not None:
            return self._controller
        app_obj = scope.get("app")
        if app_obj is not None and hasattr(app_obj, "state"):
            return getattr(app_obj.state, "file_upload_admission", None)
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        if method != "POST" or path not in ("/api/v1/analyze/file", "/api/v1/analyze/file/"):
            await self.app(scope, receive, send)
            return

        controller = self._get_controller(scope)
        token = str(uuid.uuid4())
        reserved = False

        if controller is not None:
            reserved = controller.try_reserve(token)
        else:
            # Fallback if admission controller is not configured
            reserved = True

        if not reserved:
            limit_val = (
                controller.limit if controller is not None else MAX_CONCURRENT_FILE_UPLOADS_DEFAULT
            )
            active_val = controller.reserved_count if controller is not None else limit_val
            error_payload = {
                "detail": ApiErrorDetail(
                    error_code="FILE_UPLOAD_CONCURRENCY_LIMIT_REACHED",
                    message="Maximum concurrent file upload analysis requests reached. Please retry shortly.",
                    details={"limit": limit_val, "active_uploads": active_val},
                ).model_dump()
            }
            await _send_json_response(
                send,
                status_code=429,
                payload=error_payload,
                headers=[(b"retry-after", b"5"), (b"connection", b"close")],
            )
            return

        try:
            # Layer A: Content-Length header precheck
            headers_dict = {k.lower(): v for k, v in scope.get("headers", [])}
            cl_header = headers_dict.get(b"content-length")
            if cl_header is not None:
                try:
                    cl_val = int(cl_header.decode("ascii"))
                    if cl_val > self._max_upload_bytes:
                        error_payload = {
                            "detail": ApiErrorDetail(
                                error_code="PAYLOAD_TOO_LARGE",
                                message=(
                                    f"Total upload request body exceeds maximum allowed limit of "
                                    f"{self._max_upload_bytes / (1024 * 1024):.0f} MB."
                                ),
                                details={
                                    "max_request_bytes": self._max_upload_bytes,
                                    "received_bytes": cl_val,
                                    "max_file_bytes": MAX_FILE_BYTES,
                                },
                            ).model_dump()
                        }
                        await _send_json_response(
                            send,
                            status_code=413,
                            payload=error_payload,
                            headers=[(b"connection", b"close")],
                        )
                        return
                except (ValueError, UnicodeDecodeError):
                    pass

            # Layer B: Streaming receive counter (protects chunked or falsified Content-Length)
            bytes_received = 0
            payload_too_large = False
            response_started = False

            async def counting_receive() -> Message:
                nonlocal bytes_received, payload_too_large
                if payload_too_large:
                    return {"type": "http.request", "body": b"", "more_body": False}
                message = await receive()
                if message["type"] == "http.request":
                    body_chunk = message.get("body", b"")
                    bytes_received += len(body_chunk)
                    if bytes_received > self._max_upload_bytes:
                        payload_too_large = True
                        raise PayloadTooLargeError(bytes_received=bytes_received)
                return message

            async def filtering_send(message: Message) -> None:
                nonlocal response_started
                if payload_too_large:
                    # Suppress downstream error responses (e.g. Starlette 400 parsing error)
                    return
                response_started = True
                await send(message)

            try:
                await self.app(scope, counting_receive, filtering_send)
            except Exception:
                if not payload_too_large:
                    raise

            if payload_too_large and not response_started:
                error_payload = {
                    "detail": ApiErrorDetail(
                        error_code="PAYLOAD_TOO_LARGE",
                        message=(
                            f"Total upload request body exceeds maximum allowed limit of "
                            f"{self._max_upload_bytes / (1024 * 1024):.0f} MB."
                        ),
                        details={
                            "max_request_bytes": self._max_upload_bytes,
                            "received_bytes": bytes_received,
                            "max_file_bytes": MAX_FILE_BYTES,
                        },
                    ).model_dump()
                }
                await _send_json_response(
                    send,
                    status_code=413,
                    payload=error_payload,
                    headers=[(b"connection", b"close")],
                )
        finally:
            if controller is not None:
                controller.release(token)
