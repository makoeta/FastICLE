from agno.agent import Agent
from pathlib import Path
from typing import Annotated

import yaml
from fasticrl.models.agent_save_state import AgentSaveState
from pydantic import Field

YAML_SUFFIX = ".yaml"


class ExpertConfig(AgentSaveState):
    name: Annotated[str, Field()]
    description: Annotated[str, Field(description="A short description about this expert.", default="No description available.")]

    def to_yaml(self, path: str):

        if not path.endswith(YAML_SUFFIX):
            path += YAML_SUFFIX

        with open(path, "w") as outfile:
            yaml.dump(self.model_dump(mode="json"), outfile, default_flow_style=False)

    def to_agent(self) -> Agent:
        return Agent(name=self.name, system_message=self.description)
        

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, "r") as file:
            data = yaml.safe_load(file)
            return ExpertConfig.model_validate(data)
