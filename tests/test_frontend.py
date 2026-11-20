"""
Unit tests for the clean, minimal, expandable frontend of The Predictors application.
Verifies:
- Gradio app instantiation (build_app)
- Expandable navigation switching and state machine
- Hide-by-default behavior and single-section display logic
- Always-visible chatbot and conversation history preservation
- Visual formatting helpers (trajectory, evidence, stats, chapters)
"""

import unittest
import gradio as gr
from app.main import (
    build_app,
    switch_section,
    format_trajectory_html,
    format_agent_chat_response,
    format_pipeline_trace_html,
    format_evidence_markdown,
    format_agent_evidence_html,
    format_stats_markdown,
    format_approval_card,
    build_chapters_table_html,
    format_chat_answer,
    process_query,
)


class FrontendOptimizationTests(unittest.TestCase):

    def test_build_app(self):
        """Verify that build_app successfully constructs a Gradio Blocks instance."""
        demo = build_app()
        self.assertIsInstance(demo, gr.Blocks)

    def test_switch_section_to_doc(self):
        """Verify clicking Document Info expands panel and shows only Document section."""
        outputs = switch_section("doc", "chat")
        new_sec, detail_panel, panel_title, g_doc, g_retr, g_src, g_eval, g_stat, b_chat, b_doc, b_retr, b_src, b_eval, b_stat = outputs

        self.assertEqual(new_sec, "doc")
        self.assertTrue(detail_panel.get("visible"))
        self.assertIn("Document", panel_title)
        self.assertTrue(g_doc.get("visible"))
        self.assertFalse(g_retr.get("visible"))
        self.assertFalse(g_src.get("visible"))
        self.assertFalse(g_eval.get("visible"))
        self.assertFalse(g_stat.get("visible"))
        self.assertEqual(b_doc.get("variant"), "primary")
        self.assertEqual(b_chat.get("variant"), "secondary")

    def test_switch_section_replacement(self):
        """Verify selecting another section replaces the previous section's content."""
        # Switch from doc to retrieval
        outputs = switch_section("retrieval", "doc")
        new_sec, detail_panel, panel_title, g_doc, g_retr, g_src, g_eval, g_stat, b_chat, b_doc, b_retr, b_src, b_eval, b_stat = outputs

        self.assertEqual(new_sec, "retrieval")
        self.assertTrue(detail_panel.get("visible"))
        self.assertIn("Retrieval", panel_title)
        self.assertFalse(g_doc.get("visible"))
        self.assertTrue(g_retr.get("visible"))
        self.assertFalse(g_src.get("visible"))
        self.assertFalse(g_eval.get("visible"))
        self.assertFalse(g_stat.get("visible"))
        self.assertEqual(b_retr.get("variant"), "primary")
        self.assertEqual(b_doc.get("variant"), "secondary")

    def test_switch_section_return_to_chat(self):
        """Verify returning to chat collapses the panel and highlights chat view."""
        outputs = switch_section("chat", "retrieval")
        new_sec, detail_panel, panel_title, g_doc, g_retr, g_src, g_eval, g_stat, b_chat, b_doc, b_retr, b_src, b_eval, b_stat = outputs

        self.assertEqual(new_sec, "chat")
        self.assertFalse(detail_panel.get("visible"))
        self.assertFalse(g_doc.get("visible"))
        self.assertFalse(g_retr.get("visible"))
        self.assertFalse(g_src.get("visible"))
        self.assertFalse(g_eval.get("visible"))
        self.assertFalse(g_stat.get("visible"))
        self.assertEqual(b_chat.get("variant"), "primary")

    def test_switch_section_toggle_behavior(self):
        """Verify clicking the active section toggles it off and returns to chat view."""
        outputs = switch_section("sources", "sources")
        new_sec, detail_panel = outputs[0], outputs[1]

        self.assertEqual(new_sec, "chat")
        self.assertFalse(detail_panel.get("visible"))

    def test_format_trajectory_html(self):
        """Verify formatting of tool call trajectory cards."""
        empty_html = format_trajectory_html([])
        self.assertIn("No agent tool trajectory", empty_html)

        sample_traj = [
            {
                "tool_calls": [
                    {
                        "tool": "search_corpus",
                        "arguments": {"query": "Simpson paradox", "top_k": 5},
                        "full_result": {"matches": [{"page": 196, "source": "crystal1.pdf"}]}
                    }
                ]
            }
        ]
        res_html = format_trajectory_html(sample_traj)
        self.assertIn("search_corpus", res_html)
        self.assertIn("196", res_html)

    def test_format_trajectory_html_multi_step_order(self):
        """Verify multi-step execution preserves exact 1..N step sequence and timings."""
        multi_step_run = {
            "duration_ms": 3140.5,
            "stop_reason": "completed_task",
            "tool_calls_count": 2,
            "trajectory": [
                {
                    "step": 1,
                    "step_latency_ms": 1250.0,
                    "events": ["tool_call", "observation"],
                    "tool_calls": [
                        {
                            "tool": "search_corpus",
                            "arguments": {"query": "data science levels", "top_k": 3},
                            "status": "success",
                            "result_summary": "Found 3 passages",
                            "full_result": {"matches": [{"page": 8, "score": 0.89}]}
                        }
                    ]
                },
                {
                    "step": 2,
                    "step_latency_ms": 890.5,
                    "events": ["tool_call", "observation"],
                    "tool_calls": [
                        {
                            "tool": "calculator",
                            "arguments": {"expression": "145 - 31"},
                            "status": "success",
                            "result_summary": "114",
                            "full_result": {"expression": "145 - 31", "value": 114}
                        }
                    ]
                }
            ]
        }
        html = format_trajectory_html(multi_step_run)

        # Verify summary header
        self.assertIn("2 Total Steps", html)
        self.assertIn("3,140.5 ms", html)
        self.assertIn("Task Completed Successfully", html)

        # Verify Step 1 precedes Step 2 in string order
        idx_step1 = html.find("Step 1")
        idx_step2 = html.find("Step 2")
        self.assertNotEqual(idx_step1, -1)
        self.assertNotEqual(idx_step2, -1)
        self.assertLess(idx_step1, idx_step2)

        # Verify tool names, inputs, and observation results appear
        self.assertIn("search_corpus", html)
        self.assertIn("calculator", html)
        self.assertIn("data science levels", html)
        self.assertIn("145 - 31", html)
        self.assertIn("114", html)
        self.assertIn("1250.0 ms", html)
        self.assertIn("890.5 ms", html)

    def test_format_trajectory_html_tool_error_state(self):
        """Verify tool error displays distinct Error / Failed status badge."""
        error_traj = [
            {
                "step": 1,
                "step_latency_ms": 400.0,
                "events": ["tool_call", "observation"],
                "tool_calls": [
                    {
                        "tool": "calculator",
                        "arguments": {"expression": "10 / 0"},
                        "status": "error",
                        "result_summary": "Division by zero",
                        "full_result": {"status": "error", "error": "Division by zero"}
                    }
                ]
            }
        ]
        html = format_trajectory_html(error_traj)
        self.assertIn("Error / Failed", html)
        self.assertIn("Division by zero", html)

    def test_format_trajectory_html_direct_answer_step(self):
        """Verify direct answer step displays Final Synthesis / Direct Response badge."""
        direct_run = {
            "duration_ms": 650.0,
            "stop_reason": "direct_answer",
            "tool_calls_count": 0,
            "trajectory": [
                {
                    "step": 1,
                    "step_latency_ms": 650.0,
                    "events": ["direct_answer"],
                    "tool_calls": [],
                    "final_content": "Hello! I am ready to help you explore The Crystal Ball manual.",
                    "status": "completed"
                }
            ]
        }
        html = format_trajectory_html(direct_run)
        self.assertIn("Final Synthesis / Direct Response", html)
        self.assertIn("Direct Response (No Tools Needed)", html)
        self.assertIn("Hello! I am ready to help", html)

    def test_format_trajectory_html_pending_approval_state(self):
        """Verify pending human approval displays distinct pending status badge."""
        pending_traj = [
            {
                "step": 1,
                "step_latency_ms": 500.0,
                "events": ["tool_call", "pending_approval"],
                "tool_calls": [
                    {
                        "tool": "send_message",
                        "arguments": {"channel": "general", "message": "Broadcast summary"},
                        "status": "pending_approval",
                        "result_summary": "Action proposal created",
                        "full_result": {"status": "pending_approval", "result": "Action proposal pending confirmation"}
                    }
                ]
            }
        ]
        html = format_trajectory_html(pending_traj)
        self.assertIn("Pending Human Approval", html)
        self.assertIn("send_message", html)
        self.assertIn("Broadcast summary", html)

    def test_consistent_step_counting_and_numbering(self):
        """Verify summary and heading consistency, sequential step numbering, and synthesis distinction."""
        parallel_run = {
            "query": "Calculate compound growth across two phases",
            "answer": "The combined value is 110.25 after two 5% growth phases.",
            "duration_ms": 1420.0,
            "stop_reason": "completed_task",
            "tool_sequence": ["calculator", "calculator"],
            "tool_calls_count": 2,
            "steps_count": 3,
            "trajectory": [
                {
                    "step": 1,
                    "step_latency_ms": 620.0,
                    "events": ["tool_call", "observation"],
                    "tool_calls": [
                        {
                            "tool": "calculator",
                            "arguments": {"expression": "100 * 1.05"},
                            "status": "success",
                            "result_summary": "105.0",
                            "full_result": {"result": 105.0},
                            "duration_ms": 12.4
                        },
                        {
                            "tool": "calculator",
                            "arguments": {"expression": "105 * 1.05"},
                            "status": "success",
                            "result_summary": "110.25",
                            "full_result": {"result": 110.25},
                            "duration_ms": 8.1
                        }
                    ]
                },
                {
                    "step": 2,
                    "step_latency_ms": 800.0,
                    "events": ["completed_task"],
                    "tool_calls": [],
                    "final_content": "The combined value is 110.25 after two 5% growth phases.",
                    "status": "completed"
                }
            ]
        }

        # 1. Test format_agent_chat_response consistency & numbering
        chat_md = format_agent_chat_response(parallel_run)
        self.assertIn("3 steps | 1,420 ms | Stop reason: `completed_task`", chat_md)
        self.assertIn("View Full Execution Trace (3 steps)", chat_md)
        self.assertIn("- **Step 1:** `[Tool Call]`", chat_md)
        self.assertIn("- **Step 2:** `[Tool Call]`", chat_md)
        self.assertIn("- **Step 3:** `[Synthesized Final Answer]`", chat_md)

        # Verify sequential ordering
        pos1 = chat_md.find("- **Step 1:**")
        pos2 = chat_md.find("- **Step 2:**")
        pos3 = chat_md.find("- **Step 3:**")
        self.assertNotEqual(pos1, -1)
        self.assertNotEqual(pos2, -1)
        self.assertNotEqual(pos3, -1)
        self.assertLess(pos1, pos2)
        self.assertLess(pos2, pos3)

        # 2. Test format_trajectory_html consistency & numbering
        html = format_trajectory_html(parallel_run)
        self.assertIn("3 Total Steps", html)
        self.assertIn("Step 1", html)
        self.assertIn("Step 2", html)
        self.assertIn("Step 3", html)
        self.assertIn("🔧 Tool Call", html)
        self.assertIn("Synthesized Final Answer", html)

        h_pos1 = html.find("Step 1")
        h_pos2 = html.find("Step 2")
        h_pos3 = html.find("Step 3")
        self.assertNotEqual(h_pos1, -1)
        self.assertNotEqual(h_pos2, -1)
        self.assertNotEqual(h_pos3, -1)
        self.assertLess(h_pos1, h_pos2)
        self.assertLess(h_pos2, h_pos3)

    def test_direct_answer_step_counting_consistency(self):
        """Verify direct answer produces consistent 1-step summary and trace heading."""
        direct_run = {
            "query": "Hello",
            "answer": "Hello! How can I assist you with The Crystal Ball manual?",
            "duration_ms": 320.0,
            "stop_reason": "direct_answer",
            "tool_sequence": [],
            "tool_calls_count": 0,
            "steps_count": 1,
            "trajectory": [
                {
                    "step": 1,
                    "step_latency_ms": 320.0,
                    "events": ["direct_answer"],
                    "tool_calls": [],
                    "final_content": "Hello! How can I assist you with The Crystal Ball manual?",
                    "status": "completed"
                }
            ]
        }

        chat_md = format_agent_chat_response(direct_run)
        self.assertIn("1 steps | 320 ms | Stop reason: `direct_answer`", chat_md)
        self.assertIn("View Full Execution Trace (1 steps)", chat_md)
        self.assertIn("- **Step 1:** `[Direct Response (No Tools Needed)]`", chat_md)

        html = format_trajectory_html(direct_run)
        self.assertIn("1 Total Steps", html)
        self.assertIn("Final Synthesis / Direct Response", html)



    def test_format_evidence_markdown(self):
        """Verify source evidence cards with scores and page badges."""
        sources = [
            {
                "page": 43,
                "source": "crystal1 (1).pdf",
                "combined_score": 0.82,
                "dense_score": 0.85,
                "bm25_score": 0.76,
                "text_snippet": "Demonstration of string methods."
            }
        ]
        html = format_evidence_markdown(sources)
        self.assertIn("Page 43", html)
        self.assertIn("0.82", html)
        self.assertIn("Demonstration of string methods", html)

    def test_evidence_snippets_display_actual_text_and_handle_unavailable(self):
        """Verify evidence cards display actual text and show fallback message instead of empty quotes."""
        sources = [
            # 1. Source with 'snippet' key from search_corpus
            {
                "page": 8,
                "source": "crystal1 (1).pdf",
                "score": 0.88,
                "snippet": "The four levels of the data-to-wisdom hierarchy are Data, Information, Knowledge, and Wisdom."
            },
            # 2. Source with empty snippet
            {
                "page": 12,
                "source": "crystal1 (1).pdf",
                "score": 0.75,
                "snippet": ""
            }
        ]

        html = format_evidence_markdown(sources)

        # First card has actual retrieved text
        self.assertIn("data-to-wisdom hierarchy", html)
        self.assertIn("Page 8", html)
        self.assertIn("crystal1 (1).pdf", html)
        self.assertIn("0.88", html)

        # Second card handles missing snippet with clear message rather than empty quotes
        self.assertIn("Evidence snippet unavailable for this source chunk", html)
        # Verify no empty quotes '""' in output
        self.assertNotIn('""', html)

    def test_format_agent_evidence_html_from_trajectory(self):
        """Verify format_agent_evidence_html extracts and renders chunks from search and quote calls."""
        agent_res = {
            "trajectory": [
                {
                    "step": 1,
                    "tool_calls": [
                        {
                            "tool": "search_corpus",
                            "full_result": {
                                "status": "success",
                                "matches": [
                                    {
                                        "source": "crystal1 (1).pdf",
                                        "page": 8,
                                        "score": 0.91,
                                        "snippet": "Data consists of recorded measurements that have not yet been interpreted."
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "step": 2,
                    "tool_calls": [
                        {
                            "tool": "quote_exact",
                            "full_result": {
                                "status": "success",
                                "document": "crystal1 (1).pdf",
                                "page": 12,
                                "text": "Wisdom is the application of knowledge to make informed decisions."
                            }
                        }
                    ]
                }
            ],
            "cited_pages": [8, 12]
        }

        html = format_agent_evidence_html(agent_res)
        self.assertIn("Page 8", html)
        self.assertIn("Page 12", html)
        self.assertIn("Data consists of recorded measurements", html)
        self.assertIn("Wisdom is the application of knowledge", html)
        self.assertNotIn('""', html)


    def test_build_chapters_table(self):
        """Verify that all 29 chapters are populated in the TOC directory."""
        tbl = build_chapters_table_html()
        self.assertIn("Ch 1", tbl)
        self.assertIn("Ch 29", tbl)
        self.assertIn("Evaluating a classifier", tbl)

    def test_format_chat_answer_formatting(self):
        """Verify markdown headings, lists, inline citations, and Sources cited section."""
        raw_text = (
            "### Hierarchy Levels\n\n"
            "1. **Data**: Raw unorganized values.\n"
            "2. **Information**: Contextual data [Page 8].\n\n"
            "Mathematical formula: $$\\Delta = 145 - 43 = 102$$\n"
        )
        formatted = format_chat_answer(raw_text, [8, 12])
        self.assertIn("### Hierarchy Levels", formatted)
        self.assertIn("**Data**", formatted)
        self.assertIn("[Page 8]", formatted)
        self.assertIn("$$\\Delta", formatted)
        self.assertIn("**Sources cited:**", formatted)
        self.assertIn("- [Page 8]", formatted)
        self.assertIn("- [Page 12]", formatted)

        # When Sources cited already exists, ensure no duplication
        double_formatted = format_chat_answer(formatted, [8, 12])
        self.assertEqual(double_formatted.count("Sources cited"), 1)

    def test_layout_structure(self):
        """Verify the 2-column spacious layout, narrow sidebar, and expandable detail section."""
        demo = build_app()
        elem_ids = {block.elem_id: block for block in demo.blocks.values() if hasattr(block, "elem_id")}

        # Verify key layout elements exist
        self.assertIn("sidebar-nav-col", elem_ids)
        self.assertIn("chatbot-main-col", elem_ids)
        self.assertIn("expandable-detail-section", elem_ids)
        self.assertIn("main-chatbot", elem_ids)

        # Verify there is NO right-side panel column
        self.assertNotIn("detail-panel-col", elem_ids)

        # Verify expandable detail section is hidden by default
        detail_section = elem_ids["expandable-detail-section"]
        self.assertFalse(detail_section.visible)

        # Verify main chatbot is visible
        main_chatbot = elem_ids["main-chatbot"]
        self.assertTrue(main_chatbot.visible)


if __name__ == "__main__":
    unittest.main()
