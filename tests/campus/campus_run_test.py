import os
import logging

import pytest

from icle.campus.core import Campus
from icle.campus.models.expert_config import ExpertConfig

LOGGER = logging.getLogger(__name__)

DUMMY_EXPERTS_DIR = "./tests/data/dummy_experts"


@pytest.mark.api
def test_train_new_expert(g_data):
    campus = Campus(
        global_task="Write poems.",
        expert_save_dir=DUMMY_EXPERTS_DIR,
        learner_model=g_data["model"],
        reward_model=g_data["model"],
        strategy_model=g_data["model"],
        task_generator_model=g_data["model"],
    )

    expert_name = "test_api_expert"
    expected_file = f"{DUMMY_EXPERTS_DIR}/{expert_name}.yaml"

    try:
        campus.train_new_expert(expert_name, "A temporary expert for API testing.", "Test expert")
        assert os.path.exists(expected_file), f"Expected expert file not created: {expected_file}"
        config = ExpertConfig.from_yaml(expected_file)
        assert config.name == expert_name
    finally:
        if os.path.exists(expected_file):
            os.remove(expected_file)


def test_list_all_experts(mock_model):
    campus = Campus(
        global_task="Write poems.",
        expert_save_dir=DUMMY_EXPERTS_DIR,
        learner_model=mock_model,
        reward_model=mock_model,
        strategy_model=mock_model,
        task_generator_model=mock_model,
    )

    expert_configs = campus.get_experts()

    assert len(expert_configs) > 0
