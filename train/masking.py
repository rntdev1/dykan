"""Which parts of a rendered Gemma 4 conversation the model is trained to produce.

Loss is computed only on what the model itself writes:
  * everything in a model turn, including tool calls and the closing <turn|>;
  * the opening <|tool_response> token, which the model emits to hand over to
    the tool;
but not the tool's response, the system prompt, tool declarations or user turns.
"""

from __future__ import annotations

MODEL_OPEN = "<|turn>model\n"
TURN_CLOSE = "<turn|>"
RESPONSE_OPEN = "<|tool_response>"
RESPONSE_CLOSE = "<tool_response|>"


def supervised_spans(rendered: str) -> list[tuple[int, int]]:
    """Character spans [start, end) of `rendered` that carry loss."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        start = rendered.find(MODEL_OPEN, pos)
        if start < 0:
            return spans
        cursor = start + len(MODEL_OPEN)
        close = rendered.find(TURN_CLOSE, cursor)
        end = close + len(TURN_CLOSE) if close >= 0 else len(rendered)
        while True:
            response = rendered.find(RESPONSE_OPEN, cursor, end)
            if response < 0:
                break
            hand_over = response + len(RESPONSE_OPEN)
            if hand_over > cursor:
                spans.append((cursor, hand_over))
            response_end = rendered.find(RESPONSE_CLOSE, hand_over, end)
            if response_end < 0:  # conversation ends inside a tool response
                cursor = end
                break
            cursor = response_end + len(RESPONSE_CLOSE)
        if end > cursor:
            spans.append((cursor, end))
        pos = end


def token_mask(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[bool]:
    """True for tokens that start inside a supervised span. `offsets` are the
    tokenizer's character offsets for each token."""
    mask, i = [], 0
    for start, end in offsets:
        while i < len(spans) and spans[i][1] <= start:
            i += 1
        mask.append(i < len(spans) and spans[i][0] <= start < spans[i][1] and end > start)
    return mask
