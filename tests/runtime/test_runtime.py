from pathlib import Path
from unittest.mock import MagicMock, patch

from agno.workflow import StepInput, StepOutput

from icle.models.tasks import CasterTask, CasterTaskList, RuntimeTaskList
from icle.runtime.core import Runtime

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
