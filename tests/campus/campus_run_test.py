from campus.core import Campus
import pytest
import logging

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


@pytest.mark.api
def test_train_new_expert(g_data):
    campus = Campus(
        global_task="Write poems.",
        learner_model=g_data["model"],
        reward_model=g_data["model"],
        strategy_model=g_data["model"],
        task_generator_model=g_data["model"]
    )

    campus.train_new_expert("Nature poems", "Poems about the nature.", "")
    assert True


def test_list_all_experts(g_data):
    load_dotenv()
    campus = Campus(
        global_task="Write poems.",
        learner_model=g_data["model"],
        reward_model=g_data["model"],
        strategy_model=g_data["model"],
        task_generator_model=g_data["model"]
    )

    LOGGER.info(campus.expert_save_dir)

    expert_configs = campus.get_experts()

    assert len(expert_configs) > 0
