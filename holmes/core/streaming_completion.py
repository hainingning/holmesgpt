"""Stream text deltas, then return the complete response to the agent's tool loop."""
import contextvars
import functools
from typing import Any, Protocol

MAX_STREAM_BYTES = 32 * 1024 * 1024
MAX_CHUNKS = 100_000


class StreamObserver(Protocol):
    """Optional, process-configured instrumentation with per-request state."""
    def begin(self, kwargs: dict) -> Any: ...
    def observe(self, method: str, event: Any) -> None: ...
    def end(self, token: Any, response: Any, outcome: str) -> None: ...


stream_observer: StreamObserver | None = None


def completion(llm, *, cancel_event, metadata, **kwargs):
    from litellm import stream_chunk_builder
    from holmes.utils.stream import StreamEvents, StreamMessage
    from holmes.core.tool_calling_llm import LLMInterruptedError

    from holmes.core.response_stream import activate, reset, StreamIntegrityError
    chunks, size, finish_reason = [], 0, None
    observer = stream_observer
    assembly, token = activate(observer)
    diagnostic = observer.begin(kwargs) if observer is not None else None
    response, outcome = None, 'closed'
    stream = None
    try:
        stream = llm.completion(stream=True, **kwargs)
        for chunk in stream:
            if cancel_event is not None and cancel_event.is_set():
                raise LLMInterruptedError()
            size += len(chunk.model_dump_json().encode('utf-8'))
            if size > MAX_STREAM_BYTES or len(chunks) >= MAX_CHUNKS:
                raise RuntimeError('Model stream exceeded response budget')
            chunks.append(chunk)
            if observer is not None:
                observer.observe("normalized_chunk", chunk)
            for choice in chunk.choices:
                if choice.index != 0:
                    raise RuntimeError('Multiple model choices are not supported by this streaming contract')
                # LiteLLM normalizes Responses and Chat Completions deltas.
                # Reasoning and partial tool arguments stay out of AI_MESSAGE.
                content = getattr(choice.delta, 'content', None)
                if content:
                    if not isinstance(content, str):
                        raise RuntimeError('Unsupported model content delta')
                    yield StreamMessage(event=StreamEvents.AI_MESSAGE,
                                        data={'content':content, 'metadata':metadata})
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
        if finish_reason is None:
            raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
        response = stream_chunk_builder(chunks, messages=kwargs.get('messages'))
        response = assembly.finish(response, finish_reason)
        outcome = 'returned'
        return response
    finally:
        reset(token)
        if observer is not None:
            observer.end(diagnostic, response, outcome)
        close = getattr(stream, 'close', None)
        if callable(close):
            close()


def preserve_context(function):
    # Starlette advances synchronous SSE generators in separate worker contexts.
    # The upstream retry / tracing scopes now span text yields, so all next/close
    # calls must use the same Context; never leak request scope into other chats.
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        context = contextvars.copy_context()
        iterator = function(*args, **kwargs)
        try:
            while True:
                try:
                    value = context.run(next, iterator)
                except StopIteration:
                    return
                yield value
        finally:
            context.run(iterator.close)
    return wrapped
