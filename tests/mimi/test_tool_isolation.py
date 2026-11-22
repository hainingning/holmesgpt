import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from holmes.core.models import FrontendToolDefinition
from holmes.core.tools_utils import frontend_tools
from holmes.plugins.toolsets.mcp.toolset_mcp import RemoteMCPTool


class ToolIsolationTests(unittest.TestCase):
    def test_isolation_is_opt_in_and_does_not_mutate_the_shared_executor(self):
        ai = Mock()
        ai.tool_executor.tools_by_name = {"public_search": object()}
        with patch.object(frontend_tools, "ISOLATE_EMPTY_FRONTEND_TOOLS", False):
            self.assertEqual(frontend_tools.inject_frontend_tools(ai, []), (ai, False))
        with patch.object(frontend_tools, "ISOLATE_EMPTY_FRONTEND_TOOLS", True):
            reviewer, paused = frontend_tools.inject_frontend_tools(ai, [])
            self.assertFalse(paused)
            self.assertEqual(ai.with_executor.call_args.args[0].tools_by_name, {})
            self.assertIs(reviewer, ai.with_executor.return_value)
            self.assertIn("public_search", ai.tool_executor.tools_by_name)
            tool = FrontendToolDefinition(name="platform_query", description="test fixture",
                                          parameters={"type": "object", "properties": {}}, mode="pause")
            self.assertTrue(frontend_tools.inject_frontend_tools(ai, [tool])[1])
            ai.tool_executor.clone_with_extra_tools.assert_called_once()

    def test_toolset_deadline_cancels_its_request(self):
        closed = []

        async def pending(*args):
            try:
                await asyncio.sleep(5)
            finally:
                closed.append(True)

        toolset = SimpleNamespace(name="fixture", call_timeout_seconds=0.01, _mcp_config=Mock())
        tool = SimpleNamespace(name="search", toolset=toolset, _invoke_async=pending,
                               _is_placeholder_connect_tool=lambda: False)
        result = RemoteMCPTool._invoke(tool, {"query": "fixture"}, Mock())
        self.assertEqual(result.status.value, "error")
        self.assertIn("timed out after", result.error)
        self.assertEqual(closed, [True])
