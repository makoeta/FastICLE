from functools import wraps

from agno.agent import Agent
from agno.models.base import Model

from dispatcher.models.dispatcher_task_list import DispatcherTaskList


class DispatcherAgent(Agent):

    @wraps(Agent.__init__)
    def __init__(self, model: Model, **kwargs):
        super().__init__(
            name="Dispatcher",
            description="The Dispatcher Agent analyzes complex user requests and decomposes them into a logically ordered sequence of actionable sub-tasks for downstream delegation.",
            model=model,
            output_schema=DispatcherTaskList,
            **kwargs,
        )
