from pathlib import Path

import pytest
from agno.agent import Agent

from icle.campus.models.expert_config import ExpertConfig, YAML_SUFFIX

DUMMY_EXPERTS_DIR = Path(__file__).parent.parent / "data" / "dummy_experts"
GENERAL_POEM_WRITER = DUMMY_EXPERTS_DIR / "general_poem_writer.yaml"
NATURE_POEM_WRITER = DUMMY_EXPERTS_DIR / "nature_poem_writer.yaml"


class TestFromYaml:
    def test_loads_name(self):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        assert config.name == "general_poem_writer"

    def test_loads_description(self):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        assert config.description
        assert len(config.description) > 0

    def test_loads_strategy(self):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        assert config.strategy

    def test_loads_buffer(self):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        assert config.buffer
        assert len(config.buffer) > 0

    def test_loads_task_description(self):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        assert config.task_description

    def test_accepts_path_object(self):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        assert config is not None

    def test_accepts_string_path(self):
        config = ExpertConfig.from_yaml(str(GENERAL_POEM_WRITER))
        assert config is not None

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ExpertConfig.from_yaml("/nonexistent/path/expert.yaml")

    def test_different_experts_have_different_names(self):
        general = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        nature = ExpertConfig.from_yaml(NATURE_POEM_WRITER)
        assert general.name != nature.name


class TestToYaml:
    def test_adds_yaml_suffix_when_missing(self, tmp_path):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        out_path = str(tmp_path / "expert")
        config.to_yaml(out_path)
        assert (tmp_path / "expert.yaml").exists()

    def test_no_double_suffix_when_already_present(self, tmp_path):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        out_path = str(tmp_path / "expert.yaml")
        config.to_yaml(out_path)
        assert (tmp_path / "expert.yaml").exists()
        assert not (tmp_path / "expert.yaml.yaml").exists()

    def test_round_trip_name(self, tmp_path):
        original = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        out_path = str(tmp_path / "round_trip")
        original.to_yaml(out_path)
        reloaded = ExpertConfig.from_yaml(out_path + YAML_SUFFIX)
        assert reloaded.name == original.name

    def test_round_trip_description(self, tmp_path):
        original = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        out_path = str(tmp_path / "round_trip")
        original.to_yaml(out_path)
        reloaded = ExpertConfig.from_yaml(out_path + YAML_SUFFIX)
        assert reloaded.description == original.description

    def test_round_trip_strategy(self, tmp_path):
        original = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        out_path = str(tmp_path / "round_trip")
        original.to_yaml(out_path)
        reloaded = ExpertConfig.from_yaml(out_path + YAML_SUFFIX)
        assert reloaded.strategy == original.strategy

    def test_round_trip_buffer_length(self, tmp_path):
        original = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        out_path = str(tmp_path / "round_trip")
        original.to_yaml(out_path)
        reloaded = ExpertConfig.from_yaml(out_path + YAML_SUFFIX)
        assert len(reloaded.buffer) == len(original.buffer)


class TestToAgent:
    def test_returns_agent_instance(self):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        agent = config.to_agent()
        assert isinstance(agent, Agent)

    def test_agent_name_matches_config_name(self):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        agent = config.to_agent()
        assert agent.name == config.name

    def test_agent_system_message_is_description(self):
        config = ExpertConfig.from_yaml(GENERAL_POEM_WRITER)
        agent = config.to_agent()
        assert agent.system_message == config.description

    def test_each_expert_produces_distinct_agent_name(self):
        general = ExpertConfig.from_yaml(GENERAL_POEM_WRITER).to_agent()
        nature = ExpertConfig.from_yaml(NATURE_POEM_WRITER).to_agent()
        assert general.name != nature.name
