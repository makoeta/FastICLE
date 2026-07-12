from pathlib import Path
from unittest.mock import patch

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

    def test_in_memory_experts_default_empty(self, campus):
        assert campus.in_memory_experts == []

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


class TestCampusSkipsExistingExperts:
    """Retraining an existing expert wastes tokens and overwrites its learned
    state — the Campus must skip training when the (normalized) name exists.
    LLM callers demonstrably re-request identical names, so this is enforced
    in code, not in prompts."""

    def test_has_expert_finds_saved_expert(self, campus):
        assert campus.has_expert("general_poem_writer") is True

    def test_has_expert_misses_unknown_name(self, campus):
        assert campus.has_expert("nonexistent_expert") is False

    def test_has_expert_finds_in_memory_expert(self, campus):
        campus.in_memory_experts.append(
            ExpertConfig(
                name="memory_only_expert",
                description="d",
                task_description="t",
            )
        )
        assert campus.has_expert("memory_only_expert") is True

    def test_existing_expert_is_not_retrained(self, campus):
        with (
            patch("icle.campus.core.Agent") as MockAgent,
            patch("icle.campus.core.ICRLLearner") as MockLearner,
        ):
            # "General Poem Writer" normalizes to the existing id.
            result = campus.train_new_expert(
                expert_name="General Poem Writer",
                expert_task="Write poems.",
                description="d",
            )

        assert result == "general_poem_writer"
        MockAgent.assert_not_called()
        MockLearner.assert_not_called()


