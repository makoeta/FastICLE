import logging
from collections.abc import Iterator

# Attached before the submodule imports below so that no record emitted
# during import can reach Python's last-resort stderr handler.
logging.getLogger(__name__).addHandler(logging.NullHandler())

from icle.assembler.core import Assembler
from icle.log_setup import enable_verbose_logging
from icle.runtime.core import Runtime
from icle.campus.core import Campus
from agno.models.base import Model
from agno.run.workflow import WorkflowRunOutput
from icle.models.step_identifier import STEP_IDENTIFIER
from icle.models.tasks import RuntimeTaskList

from agno.workflow import Step, StepOutput, Workflow

from icle.caster.core import CasterAgent
from icle.dispatcher.core import DispatcherAgent


class ICLE(Workflow):

    def __init__(
        self,
        model: Model,
        global_task: str,
        expert_save_dir: str,
        multi_expert_mode=False,
        verbose: bool = False,
        log_file: str | None = "icle.log",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if verbose:
            enable_verbose_logging(log_file)

        self.campus = Campus(
            global_task=global_task,
            expert_save_dir=expert_save_dir,
            learner_model=model,
            reward_model=model,
            strategy_model=model,
            task_generator_model=model,
        )

        self.caster_agent: CasterAgent = CasterAgent(
            model=model,
            global_task=global_task,
            campus=self.campus,
            multi_expert_mode=multi_expert_mode,
        )

        self.dispatcher_agent: DispatcherAgent = DispatcherAgent(model=model)

        self.runtime: Runtime = Runtime(model=model, expert_save_dir=expert_save_dir)

        self.assembler = Assembler(model=model)

        self.name = "ICLE Pipeline"
        self.steps = [
            Step(name=STEP_IDENTIFIER.DISPATCH, agent=self.dispatcher_agent),
            Step(name=STEP_IDENTIFIER.CAST, agent=self.caster_agent),
            Step(name=STEP_IDENTIFIER.RUNTIME, executor=self.runtime.runtime),
            Step(name=STEP_IDENTIFIER.ASSEMBLE, executor=self.assembler.assemble),
        ]

    def run(self, *args, **kwargs):
        output = super().run(*args, **kwargs)

        # Attach the experts that actually executed to the run's metadata, so
        # callers can inspect casting decisions without digging through
        # step_results. content stays the clean assembled answer.
        if isinstance(output, WorkflowRunOutput):
            output.metadata = {
                **(output.metadata or {}),
                **ICLE._collect_used_experts(output),
            }

        return output

    @staticmethod
    def _iter_step_outputs(output: WorkflowRunOutput) -> Iterator[StepOutput]:
        for item in output.step_results or []:
            if isinstance(item, list):
                yield from item
            else:
                yield item

    @staticmethod
    def _collect_used_experts(output: WorkflowRunOutput) -> dict:
        experts_by_task: dict[str, list[str]] = {}

        for step in ICLE._iter_step_outputs(output):
            # Identify the runtime step by its content type rather than its
            # name, so this keeps working regardless of step ordering.
            if isinstance(step.content, RuntimeTaskList):
                for task in step.content.task_list:
                    experts_by_task[task.task_id] = list(task.agent_ids)
                break

        used_experts = sorted(
            {agent_id for ids in experts_by_task.values() for agent_id in ids}
        )
        return {"used_experts": used_experts, "experts_by_task": experts_by_task}
