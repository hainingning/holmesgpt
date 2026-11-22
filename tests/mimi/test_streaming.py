"""Run with the exact image dependencies; no provider or host requests."""
import json
import threading
import contextvars
import unittest
from unittest.mock import Mock

from litellm import ModelResponseStream
from holmes.core import streaming_completion as streaming


def chunk(delta=None, finish=None, usage=None):
    value = {'id':'fixture', 'model':'gpt-4o', 'created':1, 'object':'chat.completion.chunk',
             'choices':[{'index':0, 'delta':delta or {}, 'finish_reason':finish}]}
    if usage is not None: value['usage'] = usage
    return ModelResponseStream(**value)


class Stream:
    def __init__(self, chunks): self.chunks, self.closed = chunks, False
    def __iter__(self):
        for value in self.chunks:
            if isinstance(value, Exception): raise value
            yield value
    def close(self): self.closed = True


def consume(values, cancel=None):
    stream = Stream(values)
    model = Mock()
    model.completion.return_value = stream
    iterator = streaming.completion(model, cancel_event=cancel, metadata={}, messages=[{'role':'user','content':'fixture'}])
    events = []
    try:
        while True: events.append(next(iterator))
    except StopIteration as end:
        model.completion.assert_called_once()
        assert model.completion.call_args.kwargs['stream'] is True
        return events, end.value, stream


class StreamingTests(unittest.TestCase):

    def test_generator_scope_survives_starlette_context_switches(self):
        variable=contextvars.ContextVar('fixture',default='outside')
        @streaming.preserve_context
        def scoped():
            token=variable.set('inside')
            try:
                yield variable.get()
                yield variable.get()
            finally:variable.reset(token)
        iterator=scoped()
        self.assertEqual(contextvars.Context().run(next,iterator),'inside')
        self.assertEqual(contextvars.Context().run(next,iterator),'inside')
        contextvars.Context().run(iterator.close)
        self.assertEqual(variable.get(),'outside')

    def test_text_is_incremental_and_usage_is_preserved(self):
        events, response, stream = consume([
            chunk({'role':'assistant','content':'第一段'}), chunk({'content':'，第二段'}),
            chunk(finish='stop',usage={'prompt_tokens':10,'completion_tokens':7,'total_tokens':17})])
        self.assertEqual([e.data['content'] for e in events], ['第一段','，第二段'])
        self.assertEqual(response.choices[0].message.content, '第一段，第二段')
        self.assertEqual(response.usage.total_tokens,17)
        self.assertTrue(stream.closed)

    def test_fragmented_parallel_tool_calls_and_reasoning_stay_private(self):
        events, response, stream = consume([
            chunk({'reasoning_content':'private reasoning'}),
            chunk({'tool_calls':[{'index':0,'id':'a','type':'function','function':{'name':'inspect_disks','arguments':'{"host'}}]}),
            chunk({'tool_calls':[{'index':1,'id':'b','type':'function','function':{'name':'inspect_logs','arguments':'{"host_id":2}'}}]}),
            chunk({'tool_calls':[{'index':0,'function':{'arguments':'_id":1}'}}]}), chunk(finish='tool_calls')])
        self.assertEqual(events, [])
        calls = response.choices[0].message.tool_calls
        self.assertEqual([(c.id,c.function.name,json.loads(c.function.arguments)) for c in calls],
                         [('a','inspect_disks',{'host_id':1}),('b','inspect_logs',{'host_id':2})])
        self.assertTrue(stream.closed)

    def test_partial_stream_failure_closes_without_retry(self):
        stream = Stream([chunk({'content':'partial'}), RuntimeError('fixture lost connection')])
        model = Mock(); model.completion.return_value = stream
        iterator = streaming.completion(model,cancel_event=None,metadata={},messages=[])
        self.assertEqual(next(iterator).data['content'],'partial')
        with self.assertRaises(RuntimeError): next(iterator)
        self.assertTrue(stream.closed); model.completion.assert_called_once()

    def test_incomplete_and_canceled_stream_never_return_tool_calls(self):
        with self.assertRaises(RuntimeError): consume([chunk({'content':'partial'})])
        event=threading.Event(); event.set()
        from holmes.core.tool_calling_llm import LLMInterruptedError
        with self.assertRaises(LLMInterruptedError): consume([chunk(finish='stop')],event)

    def test_additional_choice_is_not_silently_discarded(self):
        extra=chunk(finish='stop');extra.choices[0].index=1
        with self.assertRaises(RuntimeError):consume([extra])

    def test_truncated_and_filtered_streams_never_return_tool_calls(self):
        for reason in ('length', 'content_filter'):
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                consume([chunk({'tool_calls':[{'index':0,'id':'a','type':'function','function':{'name':'inspect_files','arguments':'{}'}}]}), chunk(finish=reason)])

    def test_completed_response_repairs_missing_arguments(self):
        from holmes.core.response_stream import Assembly
        from litellm import ModelResponse
        assembly = Assembly()
        args = '{"host_id":1,"script":"echo 中文","reason":"fixture"}'
        assembly.observe({'type':'response.completed','response':{'status':'completed','output':[{'type':'function_call','call_id':'a','name':'inspect_files','arguments':args,'status':'completed'}]}})
        response = ModelResponse(choices=[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','tool_calls':[{'id':'a','type':'function','function':{'name':'inspect_files','arguments':'{}'}}]}}])
        result = assembly.finish(response, 'tool_calls')
        self.assertEqual(result.choices[0].message.tool_calls[0].function.arguments, args)

    def test_incomplete_provider_terminal_cannot_be_normalized_to_success(self):
        from holmes.core.response_stream import Assembly
        from litellm import ModelResponse
        assembly = Assembly()
        assembly.observe({'type':'response.incomplete','response':{'status':'incomplete','output':[]}})
        with self.assertRaisesRegex(RuntimeError,'MIMI_STREAM_INCOMPLETE'):
            assembly.finish(ModelResponse(choices=[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'partial'}}]),'stop')



if __name__ == '__main__': unittest.main()
