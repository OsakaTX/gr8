"""One request's generation: engine frames -> text deltas with stop tokens / stop strings applied (vLLM semantics)."""
import contextlib
from dataclasses import dataclass, field

from .engine_client import EngineClient
from .errors import ApiError


@dataclass
class Step:
    text: str = ''                  # newly released text
    ids: list = field(default_factory=list)       # ids this step (stop token included)
    logprobs: list = field(default_factory=list)  # engine logprob entries, one per id (if requested)
    finish_reason: str | None = None
    stop_reason: int | str | None = None


@dataclass
class Result:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    prompt_logprobs: list | None = None
    ids: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    text: str = ''
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class Generation:
    def __init__(self, engine: EngineClient, tokenizer, rid: str, prompt: list[int], params: dict,
                 skip_special_tokens: bool = True):
        self.engine, self.t, self.rid, self.prompt = engine, tokenizer, rid, prompt
        self.params = params
        self.stop = params['_stop']
        self.include_stop = params['_include_stop']
        self.stop_ids = set(params['stop_ids'])
        self.min_tokens = params['min_tokens']
        self.hold = max((len(s) for s in self.stop), default=0) - 1 if self.stop and not self.include_stop else 0
        self.detok = tokenizer.detokenizer(skip_special_tokens)
        self.res = Result(prompt_tokens=len(prompt))
        self.sent = 0  # chars of res.text already released

    def _frame(self):
        return {'id': self.rid, 'prompt': self.prompt,
                **{k: v for k, v in self.params.items() if not k.startswith('_')}}

    async def steps(self):
        """Async iterator of Step; the last one has finish_reason set. Result in self.res."""
        if not self.engine.ready:
            raise ApiError('engine not connected', status=503)
        r = self.res
        agen = self.engine.generate(self._frame())
        async with contextlib.aclosing(agen):
            async for f in agen:
                t = f['type']
                if t == 'prompt_logprobs':
                    r.prompt_logprobs = f['lps']
                elif t == 'tokens':
                    step = self._on_tokens(f)
                    if step.finish_reason:  # stop string matched: leaving the loop cancels the engine request
                        yield step
                        return
                    if step.text or step.ids:
                        yield step
                elif t == 'finish':
                    reason = f['reason']
                    r.cached_tokens = f.get('cached_tokens', 0)
                    if reason == 'error':
                        raise ApiError(f"engine error: {f.get('message', 'unknown')}", status=500)
                    if reason == 'abort':
                        raise ApiError('request aborted by the engine', status=500)
                    r.finish_reason = reason
                    if reason == 'stop' and f.get('stop_id') is not None:
                        sid = f['stop_id']
                        r.stop_reason = None if sid == self.t.eos_token_id else sid
                    yield Step(text=self._release(final=True), finish_reason=r.finish_reason,
                               stop_reason=r.stop_reason)
                    return

    def _on_tokens(self, f) -> Step:
        r = self.res
        ids, lps = f['ids'], f.get('logprobs') or []
        step = Step(ids=[], logprobs=[])
        for j, tid in enumerate(ids):
            r.ids.append(tid)
            r.completion_tokens += 1
            step.ids.append(tid)
            if lps:
                r.logprobs.append(lps[j])
                step.logprobs.append(lps[j])
            is_stop_tok = tid in self.stop_ids and r.completion_tokens > self.min_tokens
            if is_stop_tok:
                continue  # the engine finishes on it; its text is never shown
            before = len(r.text)
            r.text += self.detok.step(tid)
            if self.stop and r.completion_tokens > self.min_tokens:
                hit = self._find_stop(before)
                if hit is not None:
                    i, s = hit
                    r.text = r.text[:i + len(s)] if self.include_stop else r.text[:i]
                    r.finish_reason, r.stop_reason = 'stop', s
                    step.text = self._release(final=True)
                    step.finish_reason, step.stop_reason = 'stop', s
                    return step
        step.text = self._release(final=False)
        return step

    def _find_stop(self, before: int):
        start = max(0, before - max(len(s) for s in self.stop) + 1)
        best = None
        for s in self.stop:
            i = self.res.text.find(s, start)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, s)
        return best

    def _release(self, final: bool) -> str:
        end = len(self.res.text) if final else max(self.sent, len(self.res.text) - self.hold)
        out = self.res.text[self.sent:end]
        self.sent = end
        return out

