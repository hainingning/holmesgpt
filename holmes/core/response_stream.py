"""Per-request stream integrity for the pinned Responses -> Chat bridge."""
import contextvars
import functools
import json
from threading import Lock

_current = contextvars.ContextVar('mimi_stream_integrity', default=None)


_adapter_lock = Lock()


class StreamIntegrityError(RuntimeError):
    pass


class Assembly:
    def __init__(self, observer=None):
        self.observer = observer
        self.saw_responses = False
        self.terminal = None
        self.final_calls = None

    def observe(self, event):
        if hasattr(event, 'model_dump'):
            event = event.model_dump()
        if not isinstance(event, dict):
            return
        kind = event.get('type', '')
        if not isinstance(kind, str) or not kind.startswith('response.'):
            return
        self.saw_responses = True
        if kind in ('response.completed', 'response.incomplete', 'response.failed'):
            response = event.get('response') or {}
            if not isinstance(response, dict):
                raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
            self.terminal = kind
            if kind == 'response.completed' and response.get('status') in (None, 'completed'):
                output = response.get('output')
                if isinstance(output, list):
                    if any(not isinstance(item, dict) for item in output):
                        raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
                    self.final_calls = [item for item in output if item.get('type') == 'function_call']
            else:
                self.terminal = 'response.incomplete'

    def finish(self, response, finish_reason):
        if finish_reason == 'length':
            raise StreamIntegrityError('MIMI_STREAM_TRUNCATED')
        if finish_reason == 'content_filter':
            raise StreamIntegrityError('MIMI_STREAM_FILTERED')
        if finish_reason not in ('stop', 'tool_calls'):
            raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
        if self.saw_responses and self.terminal != 'response.completed':
            raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
        if response is None or not response.choices:
            raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
        payload = response.model_dump()
        message = payload['choices'][0]['message']
        calls = message.get('tool_calls') or []
        if self.final_calls is not None:
            authoritative = []
            by_id = {call['id']: call for call in calls}
            if len(by_id) != len(calls):
                raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
            seen = set()
            for item in self.final_calls:
                ident, name, arguments = item.get('call_id'), item.get('name'), item.get('arguments')
                if not ident or ident in seen or not name or item.get('status') not in (None, 'completed'):
                    raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
                seen.add(ident)
                try:
                    complete = json.loads(arguments)
                except (ValueError, TypeError):
                    raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE') from None
                if not isinstance(complete, dict):
                    raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
                existing = by_id.get(ident)
                if existing:
                    function = existing.get('function') or {}
                    if function.get('name') != name:
                        raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
                    partial = function.get('arguments')
                    # Empty arguments may result from missing deltas. Only the
                    # completed provider event can repair them; never infer args.
                    if partial not in (None, '', '{}'):
                        try:
                            if json.loads(partial) != complete:
                                raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
                        except (ValueError, TypeError):
                            raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE') from None
                authoritative.append({'id': ident, 'type': 'function', 'function': {'name': name, 'arguments': arguments}})
            if set(by_id) - seen:
                raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
            message['tool_calls'] = authoritative or None
            if authoritative:
                payload['choices'][0]['finish_reason'] = 'tool_calls'
        for call in message.get('tool_calls') or []:
            try:
                value = json.loads(call['function']['arguments'])
            except (ValueError, TypeError, KeyError):
                raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE') from None
            if not isinstance(value, dict):
                raise StreamIntegrityError('MIMI_STREAM_INCOMPLETE')
        return type(response)(**payload)


def activate(observer=None):
    install()
    assembly = Assembly(observer)
    return assembly, _current.set(assembly)


def reset(token):
    _current.reset(token)


def install():
    from litellm.completion_extras.litellm_responses_transformation.transformation import OpenAiResponsesToChatCompletionStreamIterator as iterator
    with _adapter_lock:
        original = iterator.translate_responses_chunk_to_openai_stream
        if getattr(original, "_holmes_observes_responses", False):
            return

        @functools.wraps(original)
        def translate(parsed_chunk, tool_call_index_map=None):
            assembly = _current.get()
            if assembly is not None:
                if assembly.observer is not None:
                    assembly.observer.observe("raw", parsed_chunk)
                assembly.observe(parsed_chunk)
            return original(parsed_chunk, tool_call_index_map=tool_call_index_map)

        translate._holmes_observes_responses = True
        iterator.translate_responses_chunk_to_openai_stream = staticmethod(translate)
