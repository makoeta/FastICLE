from pathlib import Path
from unittest.mock import MagicMock, patch

from agno.workflow import StepInput, StepOutput

from icle.campus.models.expert_config import ExpertConfig
from icle.models.tasks import CasterTask, CasterTaskList, RuntimeTask, RuntimeTaskList
from icle.runtime.core import Runtime
from icle.runtime.prompts import DEPENDENCY_CONTEXT_PREAMBLE

DUMMY_EXPERTS_DIR = str(Path(__file__).parent.parent / "data" / "dummy_experts")


class TestRuntimeInstantiation:
    def test_model_stored(self, mock_model, tmp_path):
        runtime = Runtime(model=mock_model, expert_save_dir=str(tmp_path))
        assert runtime.model == mock_model

    def test_expert_save_dir_stored(self, mock_model, tmp_path):
        runtime = Runtime(model=mock_model, expert_save_dir=str(tmp_path))
        assert runtime.expert_save_dir == str(tmp_path)


class TestRuntimeRunTask:
    def test_run_task_returns_runtime_task(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)
        caster_task = CasterTask(
            task_id="T1",
            description="Write a nature poem.",
            agent_ids=["nature_poem_writer"],
        )

        with patch("icle.runtime.core.Team") as MockTeam:
            mock_run_output = MagicMock()
            mock_run_output.content = "A beautiful poem about nature."
            MockTeam.return_value.run.return_value = mock_run_output

            result = runtime._run_task(caster_task)

        assert result.task_id == "T1"
        assert result.description == "Write a nature poem."
        assert result.agent_ids == ["nature_poem_writer"]
        assert result.task_output == "A beautiful poem about nature."

    def test_run_task_loads_expert_from_yaml(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)
        caster_task = CasterTask(
            task_id="T1",
            description="Write a nature poem.",
            agent_ids=["nature_poem_writer"],
        )

        with patch("icle.runtime.core.Team") as MockTeam:
            mock_run_output = MagicMock()
            mock_run_output.content = "output"
            MockTeam.return_value.run.return_value = mock_run_output

            runtime._run_task(caster_task)

            assert MockTeam.called
            call_kwargs = MockTeam.call_args
            assert "members" in call_kwargs.kwargs
            members = call_kwargs.kwargs["members"]
            assert len(members) == 1
            assert members[0].name == "nature_poem_writer"


class TestRuntimeStepExecution:
    def test_runtime_step_returns_step_output(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)

        caster_task_list = CasterTaskList(
            task_list=[
                CasterTask(task_id="T1", description="Write poem.", agent_ids=["nature_poem_writer"]),
            ]
        )

        mock_step_input = MagicMock(spec=StepInput)
        mock_step_input.get_last_step_content.return_value = caster_task_list

        with patch("icle.runtime.core.Team") as MockTeam:
            mock_run_output = MagicMock()
            mock_run_output.content = "A poem."
            MockTeam.return_value.run.return_value = mock_run_output

            result = runtime.runtime(mock_step_input)

        assert isinstance(result, StepOutput)

    def test_runtime_step_processes_all_tasks(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)

        caster_task_list = CasterTaskList(
            task_list=[
                CasterTask(task_id="T1", description="Task 1", agent_ids=["nature_poem_writer"]),
                CasterTask(task_id="T2", description="Task 2", agent_ids=["general_poem_writer"]),
            ]
        )

        mock_step_input = MagicMock(spec=StepInput)
        mock_step_input.get_last_step_content.return_value = caster_task_list

        with patch("icle.runtime.core.Team") as MockTeam:
            mock_run_output = MagicMock()
            mock_run_output.content = "output"
            MockTeam.return_value.run.return_value = mock_run_output

            result = runtime.runtime(mock_step_input)

        runtime_task_list: RuntimeTaskList = result.content
        assert len(runtime_task_list.task_list) == 2


class TestRuntimeDependencyContext:
    """Dependent tasks must receive their dependencies' outputs, framed with
    an explicit instruction to treat them as fixed decisions — a bare XML
    blob gets ignored (T1 designs a menu, T2 cooks something else)."""

    def _make_runtime_task(self, task_id="T1", output="dep-output") -> RuntimeTask:
        return RuntimeTask(
            task_id=task_id,
            description=f"Description of {task_id}",
            agent_ids=["nature_poem_writer"],
            task_output=output,
        )

    def test_independent_task_gets_no_context(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)
        task = CasterTask(task_id="T2", description="x", agent_ids=["a"])

        assert (
            runtime._build_prior_context(task, {"T1": self._make_runtime_task()})
            == ""
        )

    def test_context_contains_preamble_and_dependency_output(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)
        dep = self._make_runtime_task(output="The designed menu.")
        task = CasterTask(
            task_id="T2", description="x", depends_on=["T1"], agent_ids=["a"]
        )

        context = runtime._build_prior_context(task, {"T1": dep})

        assert DEPENDENCY_CONTEXT_PREAMBLE in context
        assert dep.task_output in context
        assert dep.description in context

    def test_context_excludes_unrelated_tasks(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)
        prior = {
            "T1": self._make_runtime_task("T1", "menu"),
            "TX": self._make_runtime_task("TX", "unrelated-output"),
        }
        task = CasterTask(
            task_id="T2", description="x", depends_on=["T1"], agent_ids=["a"]
        )

        context = runtime._build_prior_context(task, prior)

        assert "menu" in context
        assert "unrelated-output" not in context

    def test_context_omits_bookkeeping_fields(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)
        task = CasterTask(
            task_id="T2", description="x", depends_on=["T1"], agent_ids=["a"]
        )

        context = runtime._build_prior_context(
            task, {"T1": self._make_runtime_task()}
        )

        assert "<agent_ids>" not in context
        assert "<depends_on>" not in context

    def test_dependent_task_prompt_carries_dependency_output(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)
        caster_task_list = CasterTaskList(
            task_list=[
                CasterTask(
                    task_id="T1",
                    description="Design the menu.",
                    agent_ids=["nature_poem_writer"],
                ),
                CasterTask(
                    task_id="T2",
                    description="Write the appetizer recipe.",
                    depends_on=["T1"],
                    agent_ids=["general_poem_writer"],
                ),
            ]
        )
        mock_step_input = MagicMock(spec=StepInput)
        mock_step_input.get_last_step_content.return_value = caster_task_list

        with patch("icle.runtime.core.Team") as MockTeam:
            mock_run_output = MagicMock()
            mock_run_output.content = "The finished menu design."
            MockTeam.return_value.run.return_value = mock_run_output

            runtime.runtime(mock_step_input)

        prompts = [c.args[0] for c in MockTeam.return_value.run.call_args_list]
        assert prompts[0] == "Design the menu."
        assert DEPENDENCY_CONTEXT_PREAMBLE in prompts[1]
        assert "The finished menu design." in prompts[1]
        assert prompts[1].endswith("Write the appetizer recipe.")


class TestRuntimeExpertPrompt:
    """The expert the Runtime actually executes must carry its trained
    strategy and experience buffer — not just the one-line description the
    Caster uses for assignment."""

    def _loaded_expert_system_message(self, runtime, caster_task) -> str:
        with patch("icle.runtime.core.Team") as MockTeam:
            mock_run_output = MagicMock()
            mock_run_output.content = "output"
            MockTeam.return_value.run.return_value = mock_run_output

            runtime._run_task(caster_task)

        members = MockTeam.call_args.kwargs["members"]
        assert len(members) == 1
        return members[0].system_message

    def test_loaded_expert_prompt_injects_strategy(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)
        caster_task = CasterTask(
            task_id="T1", description="Write a poem.", agent_ids=["general_poem_writer"]
        )
        expected = ExpertConfig.from_yaml(
            str(Path(DUMMY_EXPERTS_DIR) / "general_poem_writer.yaml")
        )

        system_message = self._loaded_expert_system_message(runtime, caster_task)

        assert expected.strategy
        assert expected.strategy in system_message

    def test_loaded_expert_prompt_injects_buffer(self, mock_model):
        runtime = Runtime(model=mock_model, expert_save_dir=DUMMY_EXPERTS_DIR)
        caster_task = CasterTask(
            task_id="T1", description="Write a poem.", agent_ids=["general_poem_writer"]
        )
        expected = ExpertConfig.from_yaml(
            str(Path(DUMMY_EXPERTS_DIR) / "general_poem_writer.yaml")
        )

        system_message = self._loaded_expert_system_message(runtime, caster_task)

        assert expected.buffer
        for attempt in expected.buffer:
            assert attempt.task in system_message
            assert attempt.output in system_message
            assert str(attempt.reward) in system_message
