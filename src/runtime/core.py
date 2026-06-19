import logging
from concurrent.futures import ThreadPoolExecutor

from agno.agent import Agent
from agno.models.base import Model
from agno.run.agent import RunOutput
from agno.team import Team
from agno.workflow import StepInput, StepOutput
from dotenv import load_dotenv
from pydantic import BaseModel

from campus.models.expert_config import ExpertConfig
from models.tasks import CasterTask, CasterTaskList, RuntimeTask, RuntimeTaskList

LOGGER = logging.getLogger(__name__)


class Runtime(BaseModel):

    expert_save_dir: str
    model: Model

    def _build_batches(self, tasks: list[CasterTask]) -> list[list[CasterTask]]:
        completed: set[str] = set()
        remaining = list(tasks)
        batches = []

        while remaining:
            ready = [t for t in remaining if all(dep in completed for dep in t.depends_on)]
            if not ready:
                raise ValueError(f"Dependency cycle detected among: {[t.task_id for t in remaining]}")
            batches.append(ready)
            completed.update(t.task_id for t in ready)
            remaining = [t for t in remaining if t.task_id not in completed]

        return batches

    def runtime(self, step_input: StepInput) -> StepOutput:
        runtime_task_list: RuntimeTaskList = RuntimeTaskList()

        caster_task_list: CasterTaskList = step_input.get_last_step_content()

        for batch in self._build_batches(caster_task_list.task_list):
            prior_outputs = list(runtime_task_list.task_list)
            LOGGER.info(f"Running {len(batch)} task(s) in parallel: {[t.task_id for t in batch]}")

            with ThreadPoolExecutor() as pool:
                futures = [(task, pool.submit(self._run_task, task, prior_outputs)) for task in batch]

            for task, future in futures:
                runtime_task_list.task_list.append(future.result())

        return StepOutput(content=runtime_task_list)

    def _run_task(self, caster_task: CasterTask, prior_outputs: list[RuntimeTask] | None = None):
        load_dotenv()
        path: str = self.expert_save_dir
        experts: list[Agent] = []

        for agent_id in caster_task.agent_ids:
            expert: Agent = ExpertConfig.from_yaml(
                path + "/" + agent_id + ".yaml"
            ).to_agent()
            experts.append(expert)

        team: Team = Team(members=experts, model=self.model)

        prior_context = RuntimeTaskList(task_list=prior_outputs).to_xml() if prior_outputs else ""
        prompt = f"{prior_context}\n\n{caster_task.description}" if prior_context else caster_task.description

        expert_out: RunOutput = team.run(prompt)

        return RuntimeTask(**caster_task.model_dump(), task_output=expert_out.content)
