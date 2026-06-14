from assembler.core import Assembler
from runtime.core import Runtime
from campus.core import Campus
from agno.models.base import Model
from models.step_identifier import STEP_IDENTIFIER

from agno.workflow import Step, Workflow

from caster.core import CasterAgent
from dispatcher.core import DispatcherAgent


class ICLE(Workflow):

    def __init__(
        self,
        model: Model,
        global_task: str,
        expert_save_dir: str,
        multi_expert_mode=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

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
