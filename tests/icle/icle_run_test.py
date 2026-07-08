from pathlib import Path

from agno.run.workflow import WorkflowRunOutput
from icle import ICLE
import logging
import pytest


LOGGER = logging.getLogger(__name__)
import pytest

DUMMY_EXPERTS_DIR = "./tests/data/dummy_experts"

# prompt = """
# Write multiple poems for my book: Write one about the nature. The other one should be about a relation ship. The third should be about refrigerators. The last one should be about a romantic relation ship in the nature.
# """

# global_task = "Poem writing."

prompt = "Write a detailed blog post about Tokyo and some of the attractions there and their cultural meaning. Also write from the perspective out of a german."

global_task = "Writing blog posts for a fake blog."

@pytest.mark.api
def test_caster_run(g_data):
    icle: ICLE = ICLE(global_task=global_task, model=g_data["model"], multi_expert_mode=True, expert_save_dir=DUMMY_EXPERTS_DIR)

    LOGGER.info(len(icle.caster_agent.campus.get_experts()))

    workflow_out: WorkflowRunOutput = icle.run(prompt, debug=True)


    LOGGER.info(workflow_out.content)

    LOGGER.info("Steps:")
    for index, step in enumerate(workflow_out.step_results):
       LOGGER.info(f"Step {index + 1}:")
       LOGGER.info(step.content)


@pytest.mark.api
def test_run_reports_used_experts(g_data):
    # A poem prompt over poem experts, so the Caster reuses the pre-seeded
    # dummy experts rather than training new ones.
    icle: ICLE = ICLE(
        global_task="Poem writing.",
        model=g_data["model"],
        multi_expert_mode=False,
        expert_save_dir=DUMMY_EXPERTS_DIR,
    )

    workflow_out: WorkflowRunOutput = icle.run(
        "Write a short poem about autumn leaves.", debug=True
    )

    LOGGER.info(f"Used experts metadata: {workflow_out.metadata}")

    # The run surfaces used experts on metadata (content stays the answer).
    assert workflow_out.metadata is not None
    used_experts = workflow_out.metadata["used_experts"]
    experts_by_task = workflow_out.metadata["experts_by_task"]

    # At least one expert executed, reported as a unique, sorted list of ids.
    assert used_experts
    assert used_experts == sorted(set(used_experts))
    assert all(isinstance(eid, str) for eid in used_experts)

    # Per-task and flat views are consistent with each other.
    flat = {eid for ids in experts_by_task.values() for eid in ids}
    assert flat == set(used_experts)

    # Every reported expert is a real, loadable expert on disk (the Runtime
    # loads each agent from "<save_dir>/<id>.yaml", so this must hold).
    for expert_id in used_experts:
        assert (Path(DUMMY_EXPERTS_DIR) / f"{expert_id}.yaml").is_file()