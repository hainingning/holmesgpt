import unittest
from types import SimpleNamespace

from holmes.core.json_schema_coerce import coerce_params
from holmes.core.tools import ToolParameter
from holmes.plugins.toolsets.mcp.toolset_mcp import RemoteMCPTool


class OptionalDomainTests(unittest.TestCase):
    def test_domain_filters_at_the_mcp_transport(self):
        schema = {
            name: ToolParameter(type="array", required=False)
            for name in ("include_domains", "exclude_domains")
        }
        tool = SimpleNamespace(parameters=schema)
        for strict in (False, True):
            for value in (None, [], ["example.org"], [None]):
                with self.subTest(strict=strict, value=value):
                    params = {"include_domains": ["github.com"], "exclude_domains": value}
                    expected = {"include_domains": ["github.com"]}
                    if value is not None:
                        expected["exclude_domains"] = value
                    coerced = coerce_params(params, schema, strict=strict)
                    self.assertEqual(RemoteMCPTool._strip_omitted_optional_params(tool, coerced), expected)
                    self.assertEqual(params["exclude_domains"], value)

    def test_required_null_is_still_forwarded_for_validation(self):
        schema = {"urls": ToolParameter(type="array", required=True)}
        tool = SimpleNamespace(parameters=schema)
        params = coerce_params({"urls": None}, schema)
        self.assertEqual(RemoteMCPTool._strip_omitted_optional_params(tool, params), {"urls": None})
