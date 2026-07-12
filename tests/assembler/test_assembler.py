from unittest.mock import MagicMock

from agno.agent import Agent
from agno.workflow import StepInput

from icle.assembler.core import ASSEMBLER_INPUT_PROMPT, Assembler, AssemblerAgent
from icle.assembler.prompts import ASSEMBLER_AGENT_PROMPT
from icle.models.tasks import RuntimeTask, RuntimeTaskList


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


class TestAssemblerUserInput:
    """The assembler must receive the workflow's original input via
    StepInput.input — session_state never carried it."""

    def _multi_task_step_input(self, original_input) -> StepInput:
        tasks = [
            RuntimeTask(
                task_id=f"T{i}",
                description=f"Task {i}",
                agent_ids=["general_poem_writer"],
                task_output=f"Output {i}",
            )
            for i in (1, 2)
        ]
        step_input = MagicMock(spec=StepInput)
        step_input.input = original_input
        step_input.get_last_step_content.return_value = RuntimeTaskList(
            task_list=tasks
        )
        return step_input

    def _assemble_and_capture_prompt(self, mock_model, original_input) -> str:
        assembler = Assembler(model=mock_model)
        assembler.assembler_agent = MagicMock(spec=AssemblerAgent)
        run_output = MagicMock()
        run_output.content = "final answer"
        run_output.metrics = None
        assembler.assembler_agent.run.return_value = run_output

        run_context = MagicMock()
        run_context.session_state = {}

        assembler.assemble(self._multi_task_step_input(original_input), run_context)

        return assembler.assembler_agent.run.call_args.args[0]

    def test_assemble_injects_workflow_input(self, mock_model):
        prompt = self._assemble_and_capture_prompt(
            mock_model, "Cook a vegan dinner."
        )
        assert "Cook a vegan dinner." in prompt

    def test_assemble_handles_missing_workflow_input(self, mock_model):
        prompt = self._assemble_and_capture_prompt(mock_model, None)
        assert "<user_input>\n\n</user_input>" in prompt


class TestAssemblerSingleTaskShortCircuit:
    def _single_task_step_input(self, output: str) -> StepInput:
        task = RuntimeTask(
            task_id="T1",
            description="Write a haiku about autumn",
            agent_ids=["general_poem_writer"],
            task_output=output,
        )
        step_input = MagicMock(spec=StepInput)
        step_input.get_last_step_content.return_value = RuntimeTaskList(task_list=[task])
        return step_input

    def test_single_task_returns_output_verbatim(self, mock_model):
        assembler = Assembler(model=mock_model)
        assembler.assembler_agent = MagicMock(spec=AssemblerAgent)

        output = "Crisp leaves drift and fall\n..."
        step_input = self._single_task_step_input(output)

        result = assembler.assemble(step_input, run_context=MagicMock())

        assert result.content == output

    def test_single_task_skips_synthesis_llm_call(self, mock_model):
        assembler = Assembler(model=mock_model)
        assembler.assembler_agent = MagicMock(spec=AssemblerAgent)

        step_input = self._single_task_step_input("some output")
        assembler.assemble(step_input, run_context=MagicMock())

        assembler.assembler_agent.run.assert_not_called()
