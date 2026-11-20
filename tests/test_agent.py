"""
Unit tests for Agent Core (Phase 2 & Phase 3 Deliverable).
Verifies:
- Autonomous tool execution and dispatch
- Step limit enforcement (capped at MAX_AGENT_STEPS)
- Repeated tool call detection and loop breaking
- Observability trace schema (step latency, event types, statuses, stop reasons)
- Direct greeting answering without tool execution
- Out-of-scope refusal handling
- Safe error handling for unknown tools or invalid inputs
"""

import unittest
import json
from unittest.mock import MagicMock, patch
from app.agent import AgentAssistant, MAX_AGENT_STEPS
from tools.registry import execute_tool, TOOL_REGISTRY


class AgentCoreTests(unittest.TestCase):

    def setUp(self):
        self.agent = AgentAssistant(max_steps=5)

    def test_agent_initialization(self):
        """Verify agent initialization and step cap configuration."""
        self.assertEqual(self.agent.max_steps, 5)
        self.assertIsNotNone(self.agent.client)

    def test_execute_tool_registry_valid_and_invalid(self):
        """Verify central tool registry dispatcher and error handling."""
        # Valid calculator execution
        res_calc = execute_tool("calculator", {"expression": "20 + 30"})
        self.assertEqual(res_calc.get("status"), "success")
        self.assertEqual(res_calc.get("value"), 50)

        # Invalid tool name
        res_unknown = execute_tool("non_existent_tool", {"foo": "bar"})
        self.assertEqual(res_unknown.get("status"), "error")
        self.assertIn("Unknown tool", res_unknown.get("error", ""))
        self.assertIn("available_tools", res_unknown)

        # Invalid arguments type
        res_bad_arg = execute_tool("calculator", "not-a-dict")
        self.assertEqual(res_bad_arg.get("status"), "error")

    def test_repeated_call_detection_and_halting(self):
        """Verify that identical repeated tool calls are detected and halted."""
        agent = AgentAssistant(max_steps=4)

        # Mock OpenAI chat completions to simulate repeated identical tool calls
        mock_choice = MagicMock()
        mock_msg = MagicMock()
        mock_tc = MagicMock()
        mock_tc.id = "call_abc123"
        mock_tc.function.name = "search_corpus"
        mock_tc.function.arguments = json.dumps({"query": "identical query", "top_k": 3})
        mock_msg.tool_calls = [mock_tc]
        mock_msg.content = None
        mock_choice.message = mock_msg

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 50
        mock_resp.usage.completion_tokens = 20

        # Simulate model repeatedly returning identical tool call
        with patch.object(agent.client.chat.completions, "create", return_value=mock_resp):
            out = agent.run("Test repeated query")

            # Check that repeated_call_detected event was logged
            events = [e for s in out["trajectory"] for e in s.get("events", [])]
            self.assertIn("repeated_call_detected", events)
            self.assertEqual(out.get("stop_reason"), "repeated_calls_halted")
            self.assertLessEqual(out.get("steps_count"), agent.max_steps)

    def test_step_limit_enforcement(self):
        """Verify that agent hard-caps execution at max_steps."""
        max_steps = 3
        agent = AgentAssistant(max_steps=max_steps)

        # Mock model to return different queries indefinitely
        def mock_completion(*args, **kwargs):
            mock_choice = MagicMock()
            mock_msg = MagicMock()
            mock_tc = MagicMock()
            mock_tc.id = f"call_{len(kwargs.get('messages', []))}"
            mock_tc.function.name = "calculator"
            mock_tc.function.arguments = json.dumps({"expression": f"1 + {len(kwargs.get('messages', []))}"})
            mock_msg.tool_calls = [mock_tc]
            mock_msg.content = None
            mock_choice.message = mock_msg
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_resp.usage.prompt_tokens = 40
            mock_resp.usage.completion_tokens = 15
            return mock_resp

        with patch.object(agent.client.chat.completions, "create", side_effect=mock_completion):
            out = agent.run("Infinite calculation test")
            self.assertLessEqual(out.get("steps_count"), max_steps)
            self.assertEqual(out.get("stop_reason"), "max_step_limit_reached")

    def test_trace_structure_and_latencies(self):
        """Verify trajectory schema contains step_latency_ms, events, and status."""
        agent = AgentAssistant(max_steps=2)

        # Mock direct answer
        mock_choice = MagicMock()
        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_msg.content = "Direct answer test."
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 20
        mock_resp.usage.completion_tokens = 10

        with patch.object(agent.client.chat.completions, "create", return_value=mock_resp):
            out = agent.run("What is a variable?")
            self.assertEqual(out["stop_reason"], "direct_answer")
            self.assertGreater(len(out["trajectory"]), 0)
            first_step = out["trajectory"][0]
            self.assertIn("step", first_step)
            self.assertIn("step_latency_ms", first_step)
            self.assertIn("events", first_step)
            self.assertIn("status", first_step)
            self.assertIn("direct_answer", first_step["events"])
            self.assertGreaterEqual(out["duration_ms"], 0.0)

    def test_direct_answering_for_greeting(self):
        """Live test: Verify agent answers a basic greeting directly without calling tools."""
        out = self.agent.run("Hello! Good morning.")
        self.assertIsInstance(out.get("answer"), str)
        self.assertGreater(len(out.get("answer")), 0)
        self.assertEqual(out.get("tool_calls_count"), 0)
        self.assertEqual(out.get("stop_reason"), "direct_answer")

    def test_out_of_scope_clean_refusal(self):
        """Live test: Verify agent refuses out-of-scope question cleanly without infinite tools."""
        out = self.agent.run("What is the traditional recipe for French onion soup?")
        answer_lower = out.get("answer", "").lower()
        self.assertTrue(
            "don't have information" in answer_lower
            or "not in the corpus" in answer_lower
            or "crystal ball" in answer_lower
            or "data science" in answer_lower
        )
        # Should stop cleanly with minimal or zero tool calls
        self.assertLessEqual(out.get("tool_calls_count"), 1)

    def test_repeated_quote_call_detection_and_halting(self):
        """Verify that repeated quote_exact calls on the same page are detected and halted."""
        agent = AgentAssistant(max_steps=6)

        mock_choice = MagicMock()
        mock_msg = MagicMock()
        mock_tc = MagicMock()
        mock_tc.id = "call_quote"
        mock_tc.function.name = "quote_exact"
        mock_tc.function.arguments = json.dumps({"page": 43, "passage": "first attempt"})
        mock_msg.tool_calls = [mock_tc]
        mock_msg.content = None
        mock_choice.message = mock_msg

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 30
        mock_resp.usage.completion_tokens = 15

        with patch.object(agent.client.chat.completions, "create", return_value=mock_resp):
            out = agent.run("Tell me about page 43")
            events = [e for s in out["trajectory"] for e in s.get("events", [])]
            self.assertIn("repeated_quote_detected", events)
            self.assertEqual(out.get("stop_reason"), "repeated_calls_halted")

    def test_citation_validation_filters_unsupported_index_pages(self):
        """Verify that unsupported index pages such as 303 are filtered from cited_pages."""
        agent = AgentAssistant(max_steps=3)

        # Step 1: Model calls search_corpus
        mock_tc = MagicMock()
        mock_tc.id = "call_search"
        mock_tc.function.name = "search_corpus"
        mock_tc.function.arguments = json.dumps({"query": "data-to-wisdom hierarchy"})

        mock_msg_step1 = MagicMock()
        mock_msg_step1.tool_calls = [mock_tc]
        mock_msg_step1.content = None
        mock_choice_step1 = MagicMock(message=mock_msg_step1)
        mock_resp_step1 = MagicMock(choices=[mock_choice_step1], usage=MagicMock(prompt_tokens=40, completion_tokens=15))

        # Step 2: Model formulates answer citing valid Page 8 and unsupported index Page 303
        mock_msg_step2 = MagicMock()
        mock_msg_step2.tool_calls = None
        mock_msg_step2.content = "The hierarchy consists of Data, Information, Knowledge, Wisdom [Page 8] and listed on [Page 303]."
        mock_choice_step2 = MagicMock(message=mock_msg_step2)
        mock_resp_step2 = MagicMock(choices=[mock_choice_step2], usage=MagicMock(prompt_tokens=60, completion_tokens=30))

        responses = [mock_resp_step1, mock_resp_step2]

        with patch.object(agent.client.chat.completions, "create", side_effect=responses):
            out = agent.run("Explain the data-to-wisdom hierarchy")

            # Page 8 must be verified
            self.assertIn(8, out.get("cited_pages", []))
            # Page 303 must NOT be in verified cited_pages
            self.assertNotIn(303, out.get("cited_pages", []))
            # Page 303 must be in hallucinated_citations
            self.assertIn(303, out.get("hallucinated_citations", []))
            # Unsupported citation [Page 303] must be stripped from final answer text
            self.assertNotIn("[Page 303]", out.get("answer", ""))
            self.assertIn("[Page 8]", out.get("answer", ""))

    def test_document_question_fast_normal_completion(self):
        """Live test: Verify document questions complete normally within 2-3 steps without max step limit."""
        out = self.agent.run("What is the data-to-wisdom hierarchy?")
        self.assertEqual(out.get("stop_reason"), "completed_task")
        # Must complete efficiently in 2-3 steps (search + synthesis)
        self.assertLessEqual(out.get("steps_count"), 3)
        self.assertIn(8, out.get("cited_pages", []))
        self.assertNotIn(303, out.get("cited_pages", []))


if __name__ == "__main__":
    unittest.main()

