from icle.models.tasks import DispatcherTaskList
from functools import wraps

from agno.agent import Agent
from agno.models.base import Model

from icle.dispatcher.prompts import DISPATCHER_SYSTEM_PROMPT


class DispatcherAgent(Agent):

    @wraps(Agent.__init__)
    def __init__(self, model: Model, **kwargs):
        super().__init__(
            name="Dispatcher",
            description="The Dispatcher Agent analyzes complex user requests and decomposes them into a logically ordered sequence of actionable sub-tasks for downstream delegation.",
            model=model,
            system_message=DISPATCHER_SYSTEM_PROMPT,
            output_schema=DispatcherTaskList,
            **kwargs,
        )
