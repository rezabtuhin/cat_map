"""Stable error envelopes contain no provisional assignments."""

from starlette.responses import JSONResponse


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 503):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def error_response(code, message, status, event_id, details=None):
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    headers = {"X-Request-ID": event_id}
    if status == 503:
        headers["Retry-After"] = "1"
    return JSONResponse({"error": error, "event_id": event_id}, status_code=status, headers=headers)
