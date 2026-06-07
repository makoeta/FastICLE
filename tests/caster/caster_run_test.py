from caster.models.caster_output import CasterOutput
from agno.run.agent import RunOutput
import logging
import random

import pytest
from fasticrl.icrl_learner import ICRLLearner

from campus.core import Campus
from caster.core import CasterAgent

LOGGER = logging.getLogger(__name__)
import pytest

prompt = """
Write a poem about flowers.
"""


@pytest.mark.api
def test_caster_run(g_data):
    campus: Campus = Campus(
        global_task="Write poems.",
        save_path="./tests/data/dummy_experts",
        model=g_data["model"],
    )
    LOGGER.info(type(g_data["model"]))
    caster = CasterAgent(
        model=g_data["model"], global_task="Write poetic poems.", campus=campus
    )

    caster.update_system_message()

    output: RunOutput = caster.run(prompt)

    caster_out: CasterOutput = output.content
    
    assert caster_out.assigned_tasks[0].agent_ids[0].lower() == "nature poem writer"