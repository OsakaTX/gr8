"""GLM 5.3 output parser: reasoning split + tool calls, matching what the old stacks' vLLM (0.30, patched GLM image,
`--reasoning-parser glm45 --tool-call-parser glm47`) returned. Our own code, written from reading vLLM's
parser/glm47_moe.py + parser/engine/parser_engine.py.

Behaviour reproduced:
- The prompt always ends with `<think>` (the GLM 5.3 template opens it even for enable_thinking=False; the patched
  image hard-wires thinking on), so output starts in REASONING. `</think>` switches to CONTENT. A stray `<think>` in
  REASONING and a stray `</think>` in CONTENT are swallowed; `<think>` in CONTENT re-enters REASONING.
- `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>` -> tool call {name, arguments JSON};
  `<tool_call>` inside reasoning ends the reasoning. Values are strings, coerced by the tool's JSON schema
  (null > integer > number > boolean > object > array > string). Unknown tool names are dropped.
- Streaming: partial tags are held back until they resolve; trailing whitespace of reasoning is held until more
  reasoning arrives (and dropped at the end of reasoning); whitespace-only content is held until non-whitespace
  content arrives (dropped if a tool call follows, emitted at the end otherwise).
- Non-streaming: reasoning is rstripped; content is stripped when tool calls were made; empty -> None.
Deviation: tool-call argument JSON is streamed one completed key/value pair at a time (vLLM streams finer-grained
prefixes). The concatenated arguments string is identical.
Tool calls are parsed only when the request has tools and tool_choice != "none"; otherwise the text stays content.
"""
import json
import math
import re
import uuid

THINK_START, THINK_END = '<think>', '</think>'
TOOL_START, TOOL_END = '<tool_call>', '</tool_call>'
ARG_KEY = '<arg_key>'

R, C, T = 'reasoning', 'content', 'tool'
_TERMS = {R: (THINK_START, THINK_END, TOOL_START), C: (THINK_START, THINK_END, TOOL_START), T: (TOOL_END,)}
_TERMS_NOTOOLS = {R: (THINK_START, THINK_END), C: (THINK_START, THINK_END), T: (TOOL_END,)}

_ARG_RE = re.compile(r'<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>', re.DOTALL)


def tool_call_id() -> str:
    return f'chatcmpl-tool-{uuid.uuid4().hex}'


# ---- schema coercion (vLLM tool_parsers/utils.py semantics) -------------------------------------------------------

def _schema_types(schema) -> set:
    if not isinstance(schema, dict):
        return {'string'}
    types = set()
    t = schema.get('type')
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types |= {x for x in t if isinstance(x, str)}
    enum = schema.get('enum')
    if isinstance(enum, list):
        for v in enum:
            types.add('null' if v is None else 'boolean' if isinstance(v, bool) else 'integer' if isinstance(v, int)
                      else 'number' if isinstance(v, float) else 'string' if isinstance(v, str)
                      else 'array' if isinstance(v, list) else 'object' if isinstance(v, dict) else 'string')
    for k in ('anyOf', 'oneOf', 'allOf'):
        if isinstance(schema.get(k), list):
            for s in schema[k]:
                types |= _schema_types(s)
    return types or {'string'}


_ALIASES = {'int': 'integer', 'float': 'number', 'bool': 'boolean', 'dict': 'object', 'list': 'array', 'str': 'string'}


def _coerce_str(value: str, types: set):
    types = {_ALIASES.get(t.strip().lower(), t.strip().lower()) for t in types}
    for cand in ('null', 'integer', 'number', 'boolean', 'object', 'array', 'string'):
        if cand not in types:
            continue
        if cand == 'null':
            if value.lower() == 'null':
                return None
        elif cand == 'string':
            return value
        elif cand == 'integer':
            try:
                return int(value)
            except ValueError:
                pass
        elif cand == 'number':
            try:
                f = float(value)
            except ValueError:
                continue
            if math.isfinite(f):
                return int(f) if f == int(f) else f
        elif cand == 'boolean':
            v = value.lower().strip()
            if v in ('true', '1'):
                return True
            if v in ('false', '0'):
                return False
        else:
            try:
                p = json.loads(value)
            except ValueError:
                continue
            if (cand == 'object' and isinstance(p, dict)) or (cand == 'array' and isinstance(p, list)):
                return p
    return value


def _coerce(value, schema):
    if isinstance(value, str):
        return _coerce_str(value, _schema_types(schema))
    if isinstance(value, dict) and isinstance(schema.get('properties'), dict):
        return {k: _coerce(v, schema['properties'][k]) if isinstance(schema['properties'].get(k), dict) else v
                for k, v in value.items()}
    if isinstance(value, list) and isinstance(schema.get('items'), dict):
        return [_coerce(v, schema['items']) for v in value]
    return value


class _Slot:
    def __init__(self):
        self.body = ''          # raw text between <tool_call> and </tool_call>
        self.name = None
        self.id = None
        self.done_pairs = 0
        self.streamed = ''      # arguments JSON emitted so far
        self.valid = True


class GlmParser:
    def __init__(self, tools: list | None = None, tool_choice='auto', include_reasoning: bool = True):
        self.tools = {t['function']['name']: t['function'] for t in (tools or [])}
        self.parse_tools = bool(tools) and tool_choice != 'none'
        self.include_reasoning = include_reasoning
        self.terms = _TERMS if self.parse_tools else _TERMS_NOTOOLS
        self.state = R
        self.buf = ''
        self.slots: list[_Slot] = []
        self.reasoning_ended = False
        self.deferred_reasoning = ''
        self.deferred_content = ''
        self.content_has_nonws = False
        # full-text accumulators for the non-streaming result
        self.all_reasoning = []
        self.all_content = []

    # ---- lexer ----------------------------------------------------------------------------------------------------
    def _lex(self, text: str, final: bool):
        """Yield ('text', s) and ('term', tag) in order; keep a possible partial tag at the end of buf."""
        self.buf += text
        while self.buf:
            terms = self.terms[self.state]
            best = None
            for tag in terms:
                i = self.buf.find(tag)
                if i >= 0 and (best is None or i < best[0]):
                    best = (i, tag)
            if best is not None:
                i, tag = best
                if i:
                    yield 'text', self.buf[:i]
                self.buf = self.buf[i + len(tag):]
                yield 'term', tag
                continue
            keep = 0
            if not final:
                for tag in terms:
                    for n in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                        if self.buf.endswith(tag[:n]):
                            keep = max(keep, n)
                            break
            out, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
            if out:
                yield 'text', out
            return

    # ---- core -----------------------------------------------------------------------------------------------------
    def _feed(self, text: str, final: bool):
        reasoning, content, tool_deltas = [], [], []
        for kind, v in self._lex(text, final):
            if kind == 'text':
                if self.state == R:
                    reasoning.append(v)
                elif self.state == C:
                    content.append(v)
                else:
                    self._tool_text(v, tool_deltas)
                continue
            if self.state == R:
                if v == THINK_END:
                    self.state, self.reasoning_ended = C, True
                elif v == TOOL_START:
                    self.state, self.reasoning_ended = T, True
                    self.slots.append(_Slot())
            elif self.state == C:
                if v == THINK_START:
                    self.state = R
                elif v == TOOL_START:
                    self.state = T
                    self.slots.append(_Slot())
            else:  # TOOL_END
                self._tool_end(tool_deltas)
                self.state = C
        if final and self.state == T and self.slots:
            self._tool_end(tool_deltas)  # unterminated call at end of output
        return ''.join(reasoning), ''.join(content), tool_deltas

    def _tool_text(self, v, deltas):
        s = self.slots[-1]
        s.body += v
        idx = len(self.slots) - 1
        if s.name is None and ARG_KEY in s.body:
            s.name = s.body.split(ARG_KEY, 1)[0].strip()
            s.valid = bool(s.name) and (not self.tools or s.name in self.tools)
            if s.valid:
                s.id = tool_call_id()
                deltas.append({'index': idx, 'id': s.id, 'type': 'function', 'function': {'name': s.name}})
        if s.name is not None and s.valid:
            pairs = self._pairs(s)
            if len(pairs) > s.done_pairs:
                s.done_pairs = len(pairs)
                prefix = self._args_json(s.name, pairs)[:-1]  # without the closing brace
                if prefix.startswith(s.streamed) and prefix != s.streamed:
                    deltas.append({'index': idx, 'function': {'arguments': prefix[len(s.streamed):]}})
                    s.streamed = prefix

    def _tool_end(self, deltas):
        s = self.slots[-1]
        idx = len(self.slots) - 1
        if s.name is None:
            s.name = s.body.strip()
            s.valid = bool(s.name) and (not self.tools or s.name in self.tools)
            if s.valid:
                s.id = tool_call_id()
                deltas.append({'index': idx, 'id': s.id, 'type': 'function',
                               'function': {'name': s.name, 'arguments': self._args_json(s.name, [])}})
                s.streamed = self._args_json(s.name, [])
            return
        if s.valid:
            final = self._args_json(s.name, self._pairs(s))
            if final.startswith(s.streamed) and final != s.streamed:
                deltas.append({'index': idx, 'function': {'arguments': final[len(s.streamed):]}})
            s.streamed = final

    @staticmethod
    def _pairs(s):
        return [(m.group('key').strip(), m.group('value')) for m in _ARG_RE.finditer(s.body)]

    def _args_json(self, name, pairs) -> str:
        params = {}
        for k, v in pairs:
            params[k] = v
        props = (self.tools.get(name) or {}).get('parameters') or {}
        props = props.get('properties') if isinstance(props, dict) else None
        if isinstance(props, dict):
            params = {k: _coerce(v, props[k]) if isinstance(props.get(k), dict) else v for k, v in params.items()}
        return json.dumps(params, ensure_ascii=False)

    # ---- streaming API --------------------------------------------------------------------------------------------
    def delta(self, text: str, finished: bool = False) -> dict | None:
        """Feed newly decoded text; return a DeltaMessage-shaped dict or None."""
        reasoning, content, tools = self._feed(text, finished)
        self.all_reasoning.append(reasoning)
        self.all_content.append(content)
        valid_tools = any(s.valid for s in self.slots)
        # reasoning: hold trailing whitespace until we know more reasoning follows
        if reasoning:
            comb = self.deferred_reasoning + reasoning
            trimmed = comb.rstrip()
            self.deferred_reasoning = comb[len(trimmed):]
            reasoning = trimmed
        elif self.reasoning_ended:
            self.deferred_reasoning = ''
        # content: hold whitespace-only content until non-whitespace arrives
        content = self.deferred_content + content
        self.deferred_content = ''
        if content and not self.content_has_nonws:
            if content.strip():
                self.content_has_nonws = True
            elif valid_tools:
                content = ''
            elif not finished:
                self.deferred_content, content = content, ''
        out = {}
        if content:
            out['content'] = content
        if reasoning and self.include_reasoning:
            out['reasoning'] = reasoning
        if tools:
            out['tool_calls'] = _coalesce(tools)
        return out or None

    # ---- non-streaming API ----------------------------------------------------------------------------------------
    def parse_full(self, text: str):
        """-> (reasoning | None, content | None, tool_calls list). Use on a fresh parser."""
        r, c, _ = self._feed(text, True)
        calls = [{'id': s.id or tool_call_id(), 'type': 'function',
                  'function': {'name': s.name, 'arguments': self._args_json(s.name, self._pairs(s))}}
                 for s in self.slots if s.valid and s.name]
        reasoning = r.rstrip() or None
        if calls:
            c = c.strip()
        return reasoning, (c or None), calls


def _coalesce(deltas):
    merged = {}
    for d in deltas:
        m = merged.get(d['index'])
        if m is None:
            merged[d['index']] = {**d, 'function': dict(d['function'])}
            continue
        for k in ('id', 'type'):
            if k in d and k not in m:
                m[k] = d[k]
        f = d['function']
        if 'name' in f and 'name' not in m['function']:
            m['function']['name'] = f['name']
        if 'arguments' in f:
            m['function']['arguments'] = m['function'].get('arguments', '') + f['arguments']
    return list(merged.values())
