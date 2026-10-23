"""
Error responses for the API.
"""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """A 422 that says which field failed and why, without repeating the value that was sent.

    FastAPI's default body repeats every offending value. A NaN or an Infinity sent as JSON cannot
    be written back as JSON, and that turned a bad request into a 500. Leaving the value out fixes
    this, and it also means the service never echoes a woman's data back in an error.
    """
    problems = [{"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": problems})
