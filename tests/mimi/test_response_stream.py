import json
import unittest
from litellm import ModelResponse
from holmes.core.response_stream import Assembly, StreamIntegrityError


class IntegrityTests(unittest.TestCase):
    def test_chat_complete_parallel_calls_keep_unicode(self):
        calls = [{'id': str(i), 'type': 'function', 'function': {'name': 'inspect_files', 'arguments': json.dumps({'host_id': i + 1, 'script': 'echo 中文', 'reason': '只读核对'})}} for i in range(2)]
        response = ModelResponse(choices=[{'message': {'role': 'assistant', 'tool_calls': calls}, 'finish_reason': 'tool_calls'}])
        result = Assembly().finish(response, 'tool_calls')
        self.assertEqual([c.function.arguments for c in result.choices[0].message.tool_calls], [c['function']['arguments'] for c in calls])

    def test_missing_and_unknown_terminal_are_rejected(self):
        response = ModelResponse(choices=[{'message': {'role': 'assistant', 'content': 'text'}, 'finish_reason': 'stop'}])
        for reason in (None, 'unknown'):
            with self.assertRaises(StreamIntegrityError): Assembly().finish(response, reason)
        state = Assembly()
        state.observe({'type': 'response.created'})
        with self.assertRaises(StreamIntegrityError): state.finish(response, 'stop')

    def test_malformed_completed_response_is_rejected(self):
        for response in (['invalid'], {'output': ['invalid']}):
            with self.assertRaises(StreamIntegrityError):
                Assembly().observe({'type': 'response.completed', 'response': response})



if __name__ == '__main__': unittest.main()
