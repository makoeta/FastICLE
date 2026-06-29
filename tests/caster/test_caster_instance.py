from pathlib import Path
from unittest.mock import patch

import pytest
from agno.agent import Agent

from icle.campus.core import Campus
from icle.caster.core import CasterAgent
from icle.models.tasks import CasterTaskList

DUMMY_EXPERTS_DIR = str(Path(__file__).parent.parent / "data" / "dummy_experts")


@pytest.fixture
def campus(mock_model):
    return Campus(
        global_task="Write poems.",
        expert_save_dir=DUMMY_EXPERTS_DIR,
        learner_model=mock_model,
        reward_model=mock_model,
        strategy_model=mock_model,
        task_generator_model=mock_model,
    )


def test_is_agent_instance(mock_model, campus):
    caster = CasterAgent(model=mock_model, global_task="Write poems.", campus=campus)
    assert isinstance(caster, Agent)


def test_output_schema_is_caster_task_list(mock_model, campus):
    caster = CasterAgent(model=mock_model, global_task="Write poems.", campus=campus)
    assert caster.output_schema == CasterTaskList


def test_has_train_new_expert_tool(mock_model, campus):
    caster = CasterAgent(model=mock_model, global_task="Write poems.", campus=campus)
    tool_names = [t.__name__ for t in caster.tools]
    assert "train_new_expert" in tool_names


def test_default_multi_expert_mode_false(mock_model, campus):
    caster = CasterAgent(model=mock_model, global_task="Write poems.", campus=campus)
    assert caster.multi_expert_mode is False


def test_multi_expert_mode_true(mock_model, campus):
    caster = CasterAgent(
        model=mock_model,
        global_task="Write poems.",
        campus=campus,
        multi_expert_mode=True,
    )
    assert caster.multi_expert_mode is True


def test_update_system_message_sets_message(mock_model, campus):
    caster = CasterAgent(model=mock_model, global_task="Write poems.", campus=campus)
    caster.update_system_message()
    assert caster.system_message
    assert len(caster.system_message) > 0


def test_update_system_message_contains_global_task(mock_model, campus):
    caster = CasterAgent(model=mock_model, global_task="Write poems.", campus=campus)
    caster.update_system_message()
    assert "Write poems" in caster.system_message


def test_update_system_message_contains_expert_names(mock_model, campus):
    caster = CasterAgent(model=mock_model, global_task="Write poems.", campus=campus)
    caster.update_system_message()
    experts = campus.get_experts()
    for expert in experts:
        assert expert.name in caster.system_message


def test_update_system_message_single_mode_prompt(mock_model, campus):
    caster = CasterAgent(
        model=mock_model,
        global_task="Write poems.",
        campus=campus,
        multi_expert_mode=False,
    )
    caster.update_system_message()
    assert "SINGLE-EXPERT MODE" in caster.system_message


def test_update_system_message_multi_mode_prompt(mock_model, campus):
    caster = CasterAgent(
        model=mock_model,
        global_task="Write poems.",
        campus=campus,
        multi_expert_mode=True,
    )
    caster.update_system_message()
    assert "MULTI-EXPERT MODE" in caster.system_message


def test_global_task_stored(mock_model, campus):
    caster = CasterAgent(model=mock_model, global_task="Write poems.", campus=campus)
    assert caster.global_task == "Write poems."


def test_campus_stored(mock_model, campus):
    caster = CasterAgent(model=mock_model, global_task="Write poems.", campus=campus)
    assert caster.campus is campus


def test_run_calls_update_system_message_before_delegating(mock_model, campus):
    caster = CasterAgent(model=mock_model, global_task="Write poems.", campus=campus)
    call_order = []

    with patch.object(caster, "update_system_message", side_effect=lambda: call_order.append("update")), \
         patch.object(Agent, "run", side_effect=lambda *a, **kw: call_order.append("agent_run")):
        caster.run("write a poem")

    assert "update" in call_order
    assert "agent_run" in call_order
    assert call_order.index("update") < call_order.index("agent_run")
