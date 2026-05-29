from agno.agent import Agent
from agno.run.agent import RunOutput

from dispatcher.dispatcher import DispatcherFactory
from dispatcher.models.dispatcher_task_list import DispatcherTaskList
import logging

LOGGER = logging.getLogger(__name__)

prompt = """
Set up an automated backup system for a local Docker environment. It needs to safely stop specific containers, compress and backup their appdata directories to a secondary NAS drive, restart the containers, and send a status notification when finished.
"""


def test_dispatcher_run(g_data):
    dispatcher: Agent = DispatcherFactory.create_dispatcher(g_data["model"])

    output: RunOutput = dispatcher.run(prompt)
    
    assert output.content_type == "DispatcherTaskList"
    assert isinstance(output.content, DispatcherTaskList)

    task_list: DispatcherTaskList = output.content
    
    assert len(task_list.tasks) > 0
    
    LOGGER.info("Hallo")
    LOGGER.info(task_list)
    LOGGER.info(task_list.tasks[0].description)

