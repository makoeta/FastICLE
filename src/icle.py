from runtime.core import Runtime
from campus.core import Campus
from agno.models.base import Model
from step_identifier import STEP_IDENTIFIER

from agno.workflow import Step, Workflow

from caster.core import CasterAgent
from dispatcher.core import DispatcherAgent


class ICLE(Workflow):

    def __init__(
        self, model: Model, global_task: str, expert_save_dir: str, **kwargs
    ):
        super().__init__(**kwargs)

        self.campus = Campus(
            global_task=global_task,
            expert_save_dir=expert_save_dir,
            model=model
        )
        
        self.caster_agent: CasterAgent = CasterAgent(
            model=model,
            global_task=global_task,
            campus=self.campus
        )
        
        self.dispatcher_agent: DispatcherAgent = DispatcherAgent(
            model=model
        )
        
        self.runtime: Runtime = Runtime(
            model=model,
            expert_save_dir=expert_save_dir
        )

        self.name = "ICLE Pipeline"
        self.steps = [
            Step(name=STEP_IDENTIFIER.DISPATCH, agent=self.dispatcher_agent),
            Step(name=STEP_IDENTIFIER.CAST, agent=self.caster_agent),
            Step(name=STEP_IDENTIFIER.RUNTIME, executor=self.runtime.runtime)
        ]
