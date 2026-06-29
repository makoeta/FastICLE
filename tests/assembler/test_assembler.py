from agno.agent import Agent

from icle.assembler.core import ASSEMBLER_INPUT_PROMPT, Assembler, AssemblerAgent
from icle.assembler.prompts import ASSEMBLER_AGENT_PROMPT


class TestAssemblerInstantiation:
    def test_creates_assembler_agent(self, mock_model):
        assembler = Assembler(model=mock_model)
        assert assembler.assembler_agent is not None

    def test_assembler_agent_is_agent_instance(self, mock_model):
        assembler = Assembler(model=mock_model)
        assert isinstance(assembler.assembler_agent, Agent)

    def test_assembler_agent_has_correct_system_message(self, mock_model):
        assembler = Assembler(model=mock_model)
        assert assembler.assembler_agent.system_message == ASSEMBLER_AGENT_PROMPT


class TestAssemblerAgentPrompt:
    def test_prompt_not_empty(self):
        assert len(ASSEMBLER_AGENT_PROMPT.strip()) > 0

    def test_prompt_mentions_synthesis(self):
        assert "Synthesis" in ASSEMBLER_AGENT_PROMPT

    def test_prompt_references_user_input_tag(self):
        assert "user_input" in ASSEMBLER_AGENT_PROMPT

    def test_prompt_references_sub_agent_outputs_tag(self):
        assert "sub_agent_outputs" in ASSEMBLER_AGENT_PROMPT

    def test_prompt_instructs_to_hide_machinery(self):
        assert "Hide the Machinery" in ASSEMBLER_AGENT_PROMPT


class TestAssemblerInputPrompt:
    def test_format_inserts_user_input(self):
        formatted = ASSEMBLER_INPUT_PROMPT.format(
            user_input="Write a poem",
            sub_agent_outputs="<tasks></tasks>",
        )
        assert "Write a poem" in formatted

    def test_format_inserts_sub_agent_outputs(self):
        formatted = ASSEMBLER_INPUT_PROMPT.format(
            user_input="test",
            sub_agent_outputs="<tasks><task>output</task></tasks>",
        )
        assert "<tasks><task>output</task></tasks>" in formatted

    def test_format_uses_xml_tags(self):
        formatted = ASSEMBLER_INPUT_PROMPT.format(
            user_input="test",
            sub_agent_outputs="output",
        )
        assert "<user_input>" in formatted
        assert "<sub_agent_outputs>" in formatted
