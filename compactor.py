import tiktoken

from config import HISTORY_LIMIT, FAST_MODEL


def _count_tokens(messages: list[dict]) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        total += len(enc.encode(msg.get("content", "")))
    return total


def maybe_compact(
    messages: list[dict],
    client,
    turn_count: int,
) -> tuple[list[dict], bool, int]:
    if turn_count < HISTORY_LIMIT or turn_count % HISTORY_LIMIT != 0:
        return messages, False, 0

    system_messages = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    if len(non_system) < 2:
        return messages, False, 0

    last_user = non_system[-1]
    to_summarize = non_system[:-1]

    tokens_before = _count_tokens(to_summarize)

    summary_response = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Summarize the following conversation in one concise paragraph, preserving key facts, preferences, and decisions.",
            },
            {
                "role": "user",
                "content": "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in to_summarize
                ),
            },
        ],
        temperature=0.0,
    )
    summary = summary_response.choices[0].message.content.strip()

    summary_message = {
        "role": "assistant",
        "content": f"[Summary of previous conversation]: {summary}",
    }

    tokens_after = _count_tokens([summary_message])
    tokens_saved = max(0, tokens_before - tokens_after)

    new_messages = system_messages + [summary_message, last_user]
    return new_messages, True, tokens_saved
