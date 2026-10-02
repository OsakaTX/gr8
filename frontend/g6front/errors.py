"""vLLM-shaped error bodies: {"error": {"message", "type", "param", "code"}}."""
import functools
import json

from aiohttp import web

# compact, UTF-8 JSON like pydantic model_dump_json()
dumps = functools.partial(json.dumps, ensure_ascii=False, separators=(',', ':'))

_TYPES = {400: 'BadRequestError', 404: 'NotFoundError', 500: 'InternalServerError', 503: 'ServiceUnavailableError'}


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400, param: str | None = None):
        super().__init__(message)
        self.message, self.status, self.param = message, status, param

    def body(self) -> dict:
        return {'error': {'message': self.message, 'type': _TYPES.get(self.status, 'BadRequestError'),
                          'param': self.param, 'code': self.status}}

    def response(self) -> web.Response:
        return web.json_response(self.body(), status=self.status, dumps=dumps)

    def sse(self) -> bytes:
        return f'data: {dumps(self.body())}\n\n'.encode()
