from agno.run.workflow import WorkflowRunOutput
from agno.workflow import StepOutput

from icle import ICLE
from icle.models.tasks import RuntimeTask, RuntimeTaskList


def _runtime_task(task_id: str, agent_ids: list[str]) -> RuntimeTask:
    return RuntimeTask(
        task_id=task_id,
        description=f"desc {task_id}",
        agent_ids=agent_ids,
        task_output=f"output {task_id}",
    )


def _output_with_runtime_step(task_list: list[RuntimeTask]) -> WorkflowRunOutput:
    runtime_step = StepOutput(content=RuntimeTaskList(task_list=task_list))
    assemble_step = StepOutput(content="final answer")
    return WorkflowRunOutput(step_results=[runtime_step, assemble_step])


class TestCollectUsedExperts:
    def test_collects_experts_per_task_and_unique_list(self):
        output = _output_with_runtime_step(
            [
                _runtime_task("T1", ["nature_poem_writer"]),
                _runtime_task("T2", ["general_poem_writer", "nature_poem_writer"]),
            ]
        )

        result = ICLE._collect_used_experts(output)

        assert result["experts_by_task"] == {
            "T1": ["nature_poem_writer"],
            "T2": ["general_poem_writer", "nature_poem_writer"],
        }
        # unique + sorted
        assert result["used_experts"] == ["general_poem_writer", "nature_poem_writer"]

    def test_single_task(self):
        output = _output_with_runtime_step([_runtime_task("T1", ["general_poem_writer"])])

        result = ICLE._collect_used_experts(output)

        assert result["used_experts"] == ["general_poem_writer"]
        assert result["experts_by_task"] == {"T1": ["general_poem_writer"]}

    def test_no_runtime_step_yields_empty(self):
        output = WorkflowRunOutput(step_results=[StepOutput(content="just text")])

        result = ICLE._collect_used_experts(output)

        assert result["used_experts"] == []
        assert result["experts_by_task"] == {}

    def test_handles_nested_step_result_lists(self):
        runtime_step = StepOutput(content=RuntimeTaskList(task_list=[_runtime_task("T1", ["e1"])]))
        output = WorkflowRunOutput(step_results=[[runtime_step]])

        result = ICLE._collect_used_experts(output)

        assert result["used_experts"] == ["e1"]
