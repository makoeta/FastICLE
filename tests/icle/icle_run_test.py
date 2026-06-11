from agno.run.workflow import WorkflowRunOutput
from icle import ICLE
import logging
import pytest


LOGGER = logging.getLogger(__name__)
import pytest

prompt = """
Write multiple poems for my book: Write one about the nature. The other one should be about a relation ship. The third should be about refrigerators. The last one should be about a romantic relation ship in the nature.
"""

global_task = "Poem writing."

@pytest.mark.api
def test_caster_run(g_data):
    icle: ICLE = ICLE(global_task=global_task, model=g_data["model"], expert_save_dir="./tests/data/dummy_experts")
    
    LOGGER.info(len(icle.caster_agent.campus.get_experts()))
    
    workflow_out: WorkflowRunOutput = icle.run(prompt, debug=True)
    
    
    LOGGER.info(workflow_out.content)
    
    LOGGER.info("Steps:")
    for index, step in enumerate(workflow_out.step_results):
       LOGGER.info(f"Step {index + 1}:")
       LOGGER.info(step.content)