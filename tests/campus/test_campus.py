from pathlib import Path

import pytest

from icle.campus.core import Campus
from icle.campus.models.expert_config import ExpertConfig

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


class TestCampusInstantiation:
    def test_global_task_stored(self, campus):
        assert campus.global_task == "Write poems."

    def test_expert_save_dir_stored(self, campus):
        assert campus.expert_save_dir == DUMMY_EXPERTS_DIR

    def test_auto_save_default_true(self, campus):
        assert campus.auto_save is True

    def test_agent_configs_default_empty(self, campus):
        assert campus.agent_configs == []

    def test_auto_save_can_be_disabled(self, mock_model):
        campus = Campus(
            global_task="task",
            expert_save_dir=DUMMY_EXPERTS_DIR,
            auto_save=False,
            learner_model=mock_model,
            reward_model=mock_model,
            strategy_model=mock_model,
            task_generator_model=mock_model,
        )
        assert campus.auto_save is False


class TestGetExperts:
    def test_returns_list(self, campus):
        experts = campus.get_experts()
        assert isinstance(experts, list)

    def test_returns_expert_config_instances(self, campus):
        experts = campus.get_experts()
        assert all(isinstance(e, ExpertConfig) for e in experts)

    def test_finds_multiple_experts(self, campus):
        experts = campus.get_experts()
        assert len(experts) > 1

    def test_expert_names_not_empty(self, campus):
        experts = campus.get_experts()
        assert all(e.name for e in experts)

    def test_expert_descriptions_are_strings(self, campus):
        experts = campus.get_experts()
        assert all(isinstance(e.description, str) for e in experts)

    def test_empty_directory_returns_empty_list(self, mock_model, tmp_path):
        campus = Campus(
            global_task="task",
            expert_save_dir=str(tmp_path),
            learner_model=mock_model,
            reward_model=mock_model,
            strategy_model=mock_model,
            task_generator_model=mock_model,
        )
        assert campus.get_experts() == []

    def test_invalid_yaml_is_skipped_gracefully(self, mock_model, tmp_path):
        # Valid YAML but missing required ExpertConfig fields — should be skipped with a warning
        (tmp_path / "incomplete.yaml").write_text("some_key: some_value\n")
        campus = Campus(
            global_task="task",
            expert_save_dir=str(tmp_path),
            learner_model=mock_model,
            reward_model=mock_model,
            strategy_model=mock_model,
            task_generator_model=mock_model,
        )
        assert campus.get_experts() == []

    def test_only_yaml_files_are_loaded(self, mock_model, tmp_path):
        # Non-yaml files should be ignored
        (tmp_path / "not_an_expert.txt").write_text("not yaml")
        campus = Campus(
            global_task="task",
            expert_save_dir=str(tmp_path),
            learner_model=mock_model,
            reward_model=mock_model,
            strategy_model=mock_model,
            task_generator_model=mock_model,
        )
        assert campus.get_experts() == []


