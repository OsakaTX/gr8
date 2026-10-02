"""GLM parser unit tests. Tool-call cases are ported from vLLM tests/tool_parsers/test_glm47_moe_tool_parser.py
(inputs + expected results; the parser code is ours). Run: python -m unittest tests.test_parser -v"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from g6front.glm_parser import GlmParser  # noqa: E402

TOOLS = [{'type': 'function', 'function': {'name': 'get_current_date', 'description': None, 'parameters': {}}},
         {'type': 'function', 'function': {'name': 'get_weather', 'description': None, 'parameters': {
             'type': 'object', 'properties': {'city': {'type': 'string'}, 'date': {'type': 'string'},
                                              'days': {'type': 'integer'}, 'metric': {'type': 'boolean'}}}}}]


def full(text, tools=TOOLS):
    """Non-streaming result for output text that starts after the prompt's <think>."""
    return GlmParser(tools, 'auto').parse_full(text)


def content_only(text):
    """vLLM's tool-parser tests feed content directly (no reasoning): prefix a closed think block."""
    return full('</think>' + text)


def stream(pieces, tools=TOOLS):
    p = GlmParser(tools, 'auto')
    out = []
    for i, piece in enumerate(pieces):
        d = p.delta(piece, finished=(i == len(pieces) - 1))
        if d:
            out.append(d)
    return out


def join(deltas, key):
    return ''.join(d.get(key, '') for d in deltas)


def tool_args(deltas):
    calls = {}
    for d in deltas:
        for tc in d.get('tool_calls', []):
            c = calls.setdefault(tc['index'], {'name': None, 'arguments': '', 'id': None})
            c['name'] = tc['function'].get('name', c['name'])
            c['id'] = tc.get('id', c['id'])
            c['arguments'] += tc['function'].get('arguments', '')
    return [calls[i] for i in sorted(calls)]


class ToolsNonStreaming(unittest.TestCase):
    def test_no_tool_call(self):
        r, c, calls = content_only('This is a plain response.')
        self.assertEqual((c, calls), ('This is a plain response.', []))

    def test_zero_arg_inline(self):
        _, c, calls = content_only('<tool_call>get_current_date</tool_call>')
        self.assertEqual(calls[0]['function']['name'], 'get_current_date')
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {})
        self.assertIsNone(c)

    def test_zero_arg_newline(self):
        _, _, calls = content_only('<tool_call>get_current_date\n</tool_call>')
        self.assertEqual(calls[0]['function']['name'], 'get_current_date')

    def test_args_same_line(self):
        _, _, calls = content_only('<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value>'
                                   '</tool_call>')
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {'city': 'Beijing'})

    def test_args_with_newlines(self):
        _, _, calls = content_only('<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>'
                                   '\n</tool_call>')
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {'city': 'Beijing'})

    def test_whitespace_preserved_in_arg_values(self):
        _, _, calls = content_only('<tool_call>get_weather<arg_key>city</arg_key><arg_value>  Beijing  '
                                   '</arg_value></tool_call>')
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {'city': '  Beijing  '})

    def test_content_before(self):
        _, c, calls = content_only('Checking.<tool_call>get_current_date</tool_call>')
        self.assertTrue(calls)
        self.assertEqual(c, 'Checking.')

    def test_multiple(self):
        _, _, calls = content_only(
            '<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>'
            '<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>')
        self.assertEqual(len(calls), 2)

    def test_whitespace_content_none(self):
        _, c, _ = content_only('  \n  <tool_call>get_current_date</tool_call>')
        self.assertIsNone(c)

    def test_schema_coercion(self):
        _, _, calls = content_only('<tool_call>get_weather<arg_key>days</arg_key><arg_value>3</arg_value>'
                                   '<arg_key>metric</arg_key><arg_value>true</arg_value></tool_call>')
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {'days': 3, 'metric': True})

    def test_unknown_tool_dropped(self):
        _, _, calls = content_only('<tool_call>nope<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>')
        self.assertEqual(calls, [])

    def test_no_tools_in_request_keeps_text(self):
        r, c, calls = full('x</think><tool_call>get_current_date</tool_call>', tools=None)
        self.assertEqual((r, c, calls), ('x', '<tool_call>get_current_date</tool_call>', []))

    def test_tool_call_inside_reasoning_ends_reasoning(self):
        r, c, calls = full('thinking <tool_call>get_current_date</tool_call>')
        self.assertEqual(r, 'thinking')
        self.assertEqual(calls[0]['function']['name'], 'get_current_date')


class ToolsStreaming(unittest.TestCase):
    def test_no_args(self):
        d = stream(['</think>', '<tool_call>', 'get_current_date', '</tool_call>'])
        calls = tool_args(d)
        self.assertEqual(calls[0]['name'], 'get_current_date')
        self.assertEqual(calls[0]['arguments'], '{}')
        self.assertTrue(calls[0]['id'].startswith('chatcmpl-tool-'))

    def test_with_args(self):
        pieces = ['</think>', '<tool_call>', 'get_weather', '<arg_key>', 'city', '</arg_key>', '<arg_value>',
                  'Beij', 'ing', '</arg_value>', '<arg_key>', 'days', '</arg_key>', '<arg_value>', '5',
                  '</arg_value>', '</tool_call>']
        d = stream(pieces)
        calls = tool_args(d)
        self.assertEqual(calls[0]['name'], 'get_weather')
        self.assertEqual(json.loads(calls[0]['arguments']), {'city': 'Beijing', 'days': 5})
        # streamed arguments concatenate to exactly the non-streaming string
        _, _, full_calls = content_only(''.join(pieces[1:]))
        self.assertEqual(calls[0]['arguments'], full_calls[0]['function']['arguments'])

    def test_split_tags(self):
        text = '</think>ok <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>'
        d = stream([text[i:i + 3] for i in range(0, len(text), 3)])
        self.assertEqual(json.loads(tool_args(d)[0]['arguments']), {'city': 'Paris'})
        self.assertEqual(join(d, 'content'), 'ok ')


class Reasoning(unittest.TestCase):
    def test_split(self):
        r, c, _ = full('I think.\n\n</think>\n\nThe answer.')
        self.assertEqual((r, c), ('I think.', '\n\nThe answer.'))

    def test_no_end_all_reasoning(self):
        self.assertEqual(full('still thinking')[:2], ('still thinking', None))

    def test_stray_think_start_swallowed(self):
        self.assertEqual(full('<think>abc</think>def')[:2], ('abc', 'def'))

    def test_stream_matches_full(self):
        text = 'Let me see.  \n</think>\n\nHello <b>world</b> </think>tail'
        for size in (1, 2, 5, 7, 100):
            d = stream([text[i:i + size] for i in range(0, len(text), size)])
            r, c, _ = full(text)
            self.assertEqual(join(d, 'reasoning'), r, size)
            self.assertEqual(join(d, 'content'), c, size)

    def test_trailing_reasoning_ws_held(self):
        p = GlmParser(None, 'none')
        self.assertEqual(p.delta('a \n'), {'reasoning': 'a'})
        self.assertEqual(p.delta('b'), {'reasoning': ' \nb'})
        self.assertIsNone(p.delta('  </think>'))

    def test_ws_only_content_held_then_released(self):
        p = GlmParser(None, 'none')
        p.delta('r</think>')
        self.assertIsNone(p.delta('\n\n'))
        self.assertEqual(p.delta('Hi'), {'content': '\n\nHi'})

    def test_include_reasoning_false(self):
        p = GlmParser(None, 'none', include_reasoning=False)
        self.assertIsNone(p.delta('secret'))
        self.assertEqual(p.delta('</think>shown', finished=True), {'content': 'shown'})


if __name__ == '__main__':
    unittest.main()
