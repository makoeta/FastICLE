from agno.run.workflow import WorkflowRunOutput
from icle import ICLE
from dispatcher.core import DispatcherAgent
import logging
import pytest
from fasticrl.icrl_learner import ICRLLearner

from campus.core import Campus
from caster.core import CasterAgent

LOGGER = logging.getLogger(__name__)
import pytest

prompt = """
Write multiple poems for my book: Write one about the nature. The other one should be about a relation ship. The third should be about refrigerators. The last one should be about a romantic relation ship in the nature.
"""

global_task = "Poem writing."

@pytest.mark.api
def test_caster_run(g_data):
    campus: Campus = Campus(
        global_task=global_task,
        save_path="./tests/data/dummy_experts",
        model=g_data["model"],
    )
    
    caster = CasterAgent(
        model=g_data["model"], global_task=global_task, campus=campus
    )

    dispatcher: DispatcherAgent = DispatcherAgent(g_data["model"])
    
    
    icle: ICLE = ICLE(dispatcher_agent=dispatcher, caster_agent=caster)
    
    workflow_out: WorkflowRunOutput = icle.run(prompt, debug=True)
    
    
    LOGGER.info(workflow_out.content)
    
    LOGGER.info("Steps:")
    for index, step in enumerate(workflow_out.step_results):
       LOGGER.info(f"Step {index + 1}:")
       LOGGER.info(step.content)