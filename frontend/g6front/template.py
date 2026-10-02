"""Chat template rendering, token-identical to what the old vLLM stacks sent to the model.

Two layers, both written from reading the code they mirror:
- normalize_messages(): vLLM's message parsing for this template ("openai" content format: string content becomes
  a list of text parts; tool results become one string; assistant `reasoning` -> `reasoning_content`; tool-call
  `arguments` JSON strings -> dicts; `developer` -> merged `system`).
- ChatTemplate.render(): transformers' jinja environment (sandboxed, trim_blocks, lstrip_blocks, loopcontrols,
  tojson(ensure_ascii=False), raise_exception, strftime_now) + continue_final_message trimming.
The rendered text is then encoded with add_special_tokens=False (the template itself emits [gMASK]<sop>).
"""
import copy
import json
import os
from datetime import datetime

import jinja2
import jinja2.ext
import jinja2.meta
from jinja2.sandbox import ImmutableSandboxedEnvironment


class TemplateError(ValueError):
    pass


class _GenerationTag(jinja2.ext.Extension):
    """{% generation %}...{% endgeneration %} (transformers' AssistantTracker); renders the body unchanged."""
    tags = {'generation'}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(('name:endgeneration',), drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method('_body'), [], [], body).set_lineno(lineno)

    def _body(self, caller):
        return caller()


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise(msg):
    raise TemplateError(msg)


def _strftime_now(fmt):
    return datetime.now().strftime(fmt)


# keyword args the renderer owns; a request cannot override them through chat_template_kwargs
_RESERVED = {'messages', 'tools', 'documents', 'chat_template', 'tokenize'}


class ChatTemplate:
    def __init__(self, model_dir: str, special_tokens: dict):
        path = os.path.join(model_dir, 'chat_template.jinja')
        with open(path) as f:
            self.source = f.read()
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[_GenerationTag, jinja2.ext.loopcontrols])
        env.filters['tojson'] = _tojson
        env.globals['raise_exception'] = _raise
        env.globals['strftime_now'] = _strftime_now
        self.template = env.from_string(self.source)
        self.variables = jinja2.meta.find_undeclared_variables(env.parse(self.source))
        self.special_tokens = special_tokens
        self.supports_developer = '"developer"' in self.source or "'developer'" in self.source

    def render(self, messages: list, tools: list | None = None, add_generation_prompt: bool = True,
               continue_final_message: bool = False, documents=None, **kwargs) -> str:
        if continue_final_message and add_generation_prompt:
            raise TemplateError('Cannot set both `continue_final_message` and `add_generation_prompt` to True.')
        conv = normalize_messages(messages, self.supports_developer)
        kw = {k: v for k, v in kwargs.items() if k not in _RESERVED}
        try:
            text = self.template.render(messages=conv, tools=tools, documents=documents,
                                        add_generation_prompt=add_generation_prompt,
                                        continue_final_message=continue_final_message,
                                        **self.special_tokens, **kw)
        except TemplateError:
            raise
        except jinja2.TemplateError as e:
            raise TemplateError(f'chat template error: {e}') from e
        if continue_final_message:
            text = _trim_to_final_message(text, conv)
        return text


def _trim_to_final_message(text: str, conv: list) -> str:
    final = conv[-1].get('content') if conv else None
    if isinstance(final, (list, tuple)):
        for block in reversed(final):
            if isinstance(block, dict) and 'text' in block:
                final = block['text']
                break
        else:
            raise TemplateError('continue_final_message needs a final message with text content')
    if not isinstance(final, str) or final.strip() not in text:
        raise TemplateError('continue_final_message: final message not found in the rendered prompt')
    loc = text.rindex(final.strip())
    if text[loc:loc + len(final.lstrip())] == final:
        return text[:loc + len(final.lstrip())]
    return text[:loc + len(final.strip())]


# ---- message normalization (vLLM chat_utils behaviour for an "openai"-content-format template) -----------------

def _parts(content) -> list:
    if content is None:
        return []
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]
    if not isinstance(content, list):
        raise TemplateError('message content must be a string, a list of parts, or null')
    out = []
    for p in content:
        if isinstance(p, str):
            out.append({'type': 'text', 'text': p})
        elif isinstance(p, dict):
            t = p.get('type')
            if t in ('text', 'input_text', 'output_text'):
                out.append({'type': 'text', 'text': p.get('text', '')})
            elif t == 'refusal':
                out.append({'type': 'text', 'text': p.get('refusal', '')})
            else:
                out.append(dict(p))
        else:
            raise TemplateError('content parts must be objects or strings')
    return out


def _parse_args(fn: dict):
    a = fn.get('arguments')
    if not a:
        return {}
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return {}
    return a if isinstance(a, dict) else {}


def normalize_messages(messages: list, supports_developer: bool = False) -> list:
    if not isinstance(messages, list):
        raise TemplateError('`messages` must be a list')
    conv = []
    for m in messages:
        if not isinstance(m, dict) or 'role' not in m:
            raise TemplateError('each message needs a role')
        role = m['role']
        c = {'role': role, 'content': _parts(m.get('content'))}
        if role == 'assistant':
            tcs = m.get('tool_calls')
            if tcs is not None:
                if not isinstance(tcs, list):
                    tcs = list(tcs)
                tcs = copy.deepcopy(tcs)
                for tc in tcs:
                    if not isinstance(tc, dict):
                        raise TemplateError('assistant tool_calls entries must be objects.')
                    fn = tc.get('function')
                    if tc.get('type', 'function') != 'function' or not isinstance(fn, dict):
                        raise TemplateError("chat completions only support assistant tool_calls of type 'function'.")
                    fn['arguments'] = _parse_args(fn)
                if tcs:
                    c['tool_calls'] = tcs
            if m.get('reasoning') is not None:
                c['reasoning'] = m['reasoning']
                c['reasoning_content'] = m['reasoning']
        elif role == 'tool':
            if 'tool_call_id' in m:
                c['tool_call_id'] = m['tool_call_id']
            parts = c['content']
            if not any(isinstance(p, dict) and p.get('type') != 'text' for p in parts):
                texts = [p.get('text', '') for p in parts]
                c['content'] = '\n'.join(texts) if texts else ''
        if isinstance(m.get('name'), str):
            c['name'] = m['name']
        if role == 'developer':
            c['tools'] = m.get('tools')
        conv.append(c)
    if not supports_developer and any(c['role'] == 'developer' for c in conv):
        conv = _developer_to_system(conv)
    return conv


def _developer_to_system(conv):
    conv = [dict(c, role='system') if c['role'] == 'developer' else c for c in conv]
    for c in conv:
        if c['role'] == 'system':
            c.pop('tools', None)
    sys_texts, rest, needs = [], [], False
    for i, c in enumerate(conv):
        if c['role'] == 'system':
            if i > 0 or sys_texts:
                needs = True
            content = c.get('content', '')
            if isinstance(content, list):
                content = '\n'.join(p['text'] if isinstance(p, dict) else p for p in content
                                    if isinstance(p, str) or (isinstance(p, dict) and 'text' in p))
            if content:
                sys_texts.append(content)
        else:
            rest.append(c)
    if not needs:
        return conv
    return [{'role': 'system', 'content': '\n\n'.join(sys_texts)}, *rest]


def normalize_tools(tools) -> list | None:
    """vLLM's ChatCompletionToolsParam.model_dump(): {type, function:{name, description, parameters[, strict]
    [, defer_loading]}[, defer_loading]} with None description/parameters kept (the template prints them)."""
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise TemplateError('`tools` must be a list')
    out = []
    for t in tools:
        if not isinstance(t, dict) or not isinstance(t.get('function'), dict) or 'name' not in t['function']:
            raise TemplateError('each tool needs {"type": "function", "function": {"name": ...}}')
        if t.get('type', 'function') != 'function':
            raise TemplateError("only tools of type 'function' are supported")
        f = t['function']
        fd = {'name': f['name'], 'description': f.get('description'), 'parameters': f.get('parameters')}
        dl = f.get('defer_loading', t.get('defer_loading'))
        if f.get('strict') is not None:
            fd['strict'] = f['strict']
        if dl is not None:
            fd['defer_loading'] = dl
        td = {'type': 'function', 'function': fd}
        if t.get('defer_loading') is not None:
            td['defer_loading'] = t['defer_loading']
        out.append(td)
    return out


def build_template_kwargs(chat_template_kwargs: dict | None, add_generation_prompt: bool = True,
                          continue_final_message: bool = False, reasoning_effort=None,
                          documents=None) -> dict:
    """vLLM ChatCompletionRequest.build_chat_params(): request chat_template_kwargs are the defaults; the top-level
    fields override them unless None/'auto'."""
    user = dict(chat_template_kwargs or {})
    extra = {'add_generation_prompt': add_generation_prompt, 'continue_final_message': continue_final_message,
             'documents': documents, 'reasoning_effort': reasoning_effort}
    if reasoning_effort is not None and 'enable_thinking' not in user:
        extra['enable_thinking'] = reasoning_effort != 'none'
    bad = {'chat_template', 'tokenize'} & user.keys()
    if bad:
        raise TemplateError(f'Found unexpected chat template kwargs from request: {bad}')
    return user | {k: v for k, v in extra.items() if v is not None and v != 'auto'}
