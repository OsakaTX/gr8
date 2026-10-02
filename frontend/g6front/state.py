"""Process-wide state shared by the routes."""
import json
import time
import uuid

from aiohttp import web

from .engine_client import EngineClient
from .errors import ApiError
from .template import ChatTemplate
from .tokenizer import GlmTokenizer


class State:
    def __init__(self, args):
        self.args = args
        self.tok = GlmTokenizer(args.model_dir)
        self.template = ChatTemplate(args.model_dir, {'eos_token': self.tok.eos_token, 'pad_token': self.tok.pad_token})
        self.engine = EngineClient(args.engine_sock)
        self.served = list(args.served_model_name)
        self.model_name = self.served[0]
        self.reasoning_fields = {'both': ('reasoning', 'reasoning_content'), 'reasoning': ('reasoning',),
                                 'reasoning_content': ('reasoning_content',)}[args.reasoning_fields]
        self.started = int(time.time())

    @property
    def max_model_len(self) -> int:
        if self.args.max_model_len:
            return self.args.max_model_len
        if self.engine.hello:
            return int(self.engine.hello.get('max_model_len', 0))
        return 0

    @property
    def max_logprobs(self) -> int:
        return int((self.engine.hello or {}).get('max_logprobs', 20))

    def check_model(self, body: dict):
        m = body.get('model')
        if m is not None and m not in self.served:
            raise ApiError(f'The model `{m}` does not exist.', status=404, param='model')

    def require_engine(self):
        if not self.engine.ready:
            raise ApiError('engine not connected', status=503)


def new_id(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


async def read_json(request: web.Request) -> dict:
    try:
        body = await request.json(loads=json.loads)
    except ValueError as e:
        raise ApiError(f'invalid JSON body: {e}')
    if not isinstance(body, dict):
        raise ApiError('request body must be a JSON object')
    return body


def sse(obj) -> bytes:
    return b'data: ' + json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode() + b'\n\n'


async def open_sse(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(headers={'Content-Type': 'text/event-stream; charset=utf-8',
                                       'Cache-Control': 'no-cache'})
    await resp.prepare(request)
    return resp


def usage(prompt_tokens: int, completion_tokens: int, cached: int | None, details: bool, full: bool) -> dict:
    u = {'prompt_tokens': prompt_tokens, 'total_tokens': prompt_tokens + completion_tokens,
         'completion_tokens': completion_tokens}
    if details:
        u['prompt_tokens_details'] = {'cached_tokens': cached or 0}
    elif full:
        u['prompt_tokens_details'] = None
    return u
