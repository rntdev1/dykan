"""Parser for Gemma 4's native tool-call syntax.

The model emits calls as

    <|tool_call>call:name{key:<|"|>text<|"|>,n:5,flag:true,list:[...],obj:{...}}<tool_call|>

which is not JSON: keys are bare, strings are wrapped in the <|"|> token.
`parse_tool_calls` returns [{"name", "arguments"}], and raises nothing: a call
that cannot be parsed is returned with "arguments": None so it can be counted
as invalid.
"""

from __future__ import annotations

import re

QUOTE = '<|"|>'
_CALL_RE = re.compile(r"<\|tool_call>call:([^\s{]+)\{", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")
_KEY_RE = re.compile(r"[^:,{}\[\]]+")


class _Parser:
    def __init__(self, text: str, pos: int):
        self.text, self.pos = text, pos

    def _skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \n\t":
            self.pos += 1

    def _eat(self, token: str) -> bool:
        self._skip()
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def mapping_body(self) -> dict:
        """Parses `key:value,...}` (the opening brace is already consumed)."""
        out: dict = {}
        if self._eat("}"):
            return out
        while True:
            self._skip()
            if self._eat(QUOTE):  # quoted key
                end = self.text.index(QUOTE, self.pos)
                key, self.pos = self.text[self.pos : end], end + len(QUOTE)
            else:
                m = _KEY_RE.match(self.text, self.pos)
                if not m:
                    raise ValueError("expected a key")
                key, self.pos = m.group().strip(), m.end()
            if not self._eat(":"):
                raise ValueError("expected ':'")
            out[key] = self.value()
            if self._eat("}"):
                return out
            if not self._eat(","):
                raise ValueError("expected ',' or '}'")

    def value(self):
        self._skip()
        if self._eat(QUOTE):
            end = self.text.index(QUOTE, self.pos)
            value, self.pos = self.text[self.pos : end], end + len(QUOTE)
            return value
        if self._eat("{"):
            return self.mapping_body()
        if self._eat("["):
            items = []
            if self._eat("]"):
                return items
            while True:
                items.append(self.value())
                if self._eat("]"):
                    return items
                if not self._eat(","):
                    raise ValueError("expected ',' or ']'")
        for word, const in (("true", True), ("false", False), ("null", None)):
            if self._eat(word):
                return const
        m = _NUMBER_RE.match(self.text, self.pos)
        if m:
            self.pos = m.end()
            return float(m.group()) if any(c in m.group() for c in ".eE") else int(m.group())
        raise ValueError("unexpected value")


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in _CALL_RE.finditer(text):
        try:
            arguments = _Parser(text, m.end()).mapping_body()
        except (ValueError, IndexError):
            arguments = None
        calls.append({"name": m.group(1), "arguments": arguments})
    return calls


def strip_markup(text: str) -> str:
    """The user-visible part of a model turn: no tool blocks, thoughts or turn tokens."""
    text = re.sub(r"<\|tool_call>.*?(<tool_call\|>|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|tool_response>.*?(<tool_response\|>|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel>.*?(<channel\|>|$)", "", text, flags=re.DOTALL)
    return re.sub(r"<\|?[a-z_]+\|?>", "", text).strip()
