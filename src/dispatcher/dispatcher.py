from agno.agent import Agent
from agno.models.base import Model

from dispatcher.models.dispatcher_task_list import DispatcherTaskList


class DispatcherFactory:

    @classmethod
    def create_dispatcher(cls, model: Model) -> Agent:

        dispatcher = Agent(
            name="Dispatcher",
            description="The Dispatcher Agent analyzes complex user requests and decomposes them into a logically ordered sequence of actionable sub-tasks for downstream delegation.",
            model=model,
            output_schema=DispatcherTaskList,
        )

        return dispatcher
