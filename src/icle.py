from agno.workflow import Workflow

from caster.core import CasterAgent
from dispatcher.core import DispatcherAgent


class ICLE(Workflow):

    def __init__(
        self, dispatcher_agent: DispatcherAgent, caster_agent: CasterAgent, **kwargs
    ):
        super().__init__(**kwargs)

        self.dispatcher_agent = dispatcher_agent
        self.caster_agent = caster_agent
        self.name = "ICLE Pipeline"
        self.steps = [self.dispatcher_agent, self.caster_agent]
