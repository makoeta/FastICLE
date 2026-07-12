import logging

from icle.models.tasks import DispatcherTaskList
from functools import wraps

from agno.agent import Agent
from agno.models.base import Model

from icle.dispatcher.prompts import DISPATCHER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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

    @wraps(Agent.run)
    def run(self, *args, **kwargs):
        input_prompt = kwargs.get("input")
        if input_prompt is None and args:
            input_prompt = args[0]
        logger.debug("Dispatcher input:\n%s", input_prompt)

        run_output = super().run(*args, **kwargs)

        content = run_output.content
        if isinstance(content, DispatcherTaskList):
            logger.info("Dispatched %d task(s)", len(content.task_list))
            for task in content.task_list:
                logger.debug(
                    "[%s] depends_on=%s | %s",
                    task.task_id,
                    task.depends_on,
                    task.description,
                )

        return run_output
