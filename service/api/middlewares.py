import time

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response

from service.log import access_logger, app_logger
from service.models import Error
from service.response import server_error


class AccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        started_at = time.perf_counter()
        response = await call_next(request)
        request_time = time.perf_counter() - started_at

        status_code = response.status_code

        access_logger.info(
            msg="",
            extra={
                "request_time": round(request_time, 4),
                "status_code": status_code,
                "requested_url": request.url,
                "method": request.method,
            },
        )
        return response


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await call_next(request)
        except Exception as e:  # pylint: disable=W0703,W1203
            app_logger.exception(msg=f"Caught unhandled {e.__class__} exception: {e}")
            error = Error(error_key="server_error", error_message="Internal Server Error")
            return server_error([error])


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        whitelisted_paths = ["/docs", "/openapi.json", "/redoc"]  # Разрешенные маршруты без аутентификации
        if request.url.path in whitelisted_paths:
            return await call_next(request)
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Missing or invalid Authorization header"}
            )
        token = auth_header.split(" ")[1]
        if token != request.app.state.auth_token:
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Invalid token"})
        return await call_next(request)


def add_middlewares(app: FastAPI) -> None:
    # do not change order
    app.add_middleware(ExceptionHandlerMiddleware)
    app.add_middleware(BearerAuthMiddleware)
    app.add_middleware(AccessMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
