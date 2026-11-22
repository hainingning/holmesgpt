import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from holmes.common import env_vars
from holmes.core.truncation import input_context_window_limiter as limiter
from holmes.utils.stream import build_stream_event_token_count


class InputBudgetTests(unittest.TestCase):
    def test_capacity_reserves_output_and_matches_the_reported_input_budget(self):
        llm = Mock()
        for capacity, output, expected in [(1_050_000, 128_000, 258_000), (32_000, 8_000, 24_000)]:
            with self.subTest(capacity=capacity), patch.object(env_vars, "MAX_INPUT_TOKENS", 258_000), \
                    patch.object(limiter, "ENABLE_CONVERSATION_HISTORY_COMPACTION", True), \
                    patch.object(limiter, "get_context_window_compaction_threshold_pct", return_value=95):
                llm.get_context_window_size.return_value = capacity
                llm.get_maximum_output_token.return_value = output
                llm.count_tokens.return_value = SimpleNamespace(total_tokens=expected * 95 // 100)
                self.assertIsNone(limiter.check_compaction_needed(llm, [], []))
                llm.count_tokens.return_value.total_tokens += 1
                start = limiter.check_compaction_needed(llm, [], [])
                effective_capacity = start.data["metadata"]["max_context_size"]
                event = build_stream_event_token_count({"max_tokens": effective_capacity, "max_output_tokens": output})
                self.assertEqual(event.data["metadata"]["input_budget"], expected)

    def test_unconfigured_threshold_keeps_the_upstream_total_window_rule(self):
        llm = Mock()
        llm.get_context_window_size.return_value = 32_000
        llm.get_maximum_output_token.return_value = 8_000
        with patch.object(env_vars, "MAX_INPUT_TOKENS", 0), \
                patch.object(limiter, "get_context_window_compaction_threshold_pct", return_value=95):
            self.assertEqual(limiter._window_limits(llm), (32_000, 8_000, 22_400))
            self.assertNotIn("input_budget", build_stream_event_token_count({}).data["metadata"])
