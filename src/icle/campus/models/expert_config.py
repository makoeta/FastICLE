from agno.agent import Agent
from pathlib import Path
from typing import Annotated

import yaml
from fasticrl.models.agent_save_state import AgentSaveState
from pydantic import Field

YAML_SUFFIX = ".yaml"


class ExpertConfig(AgentSaveState):
    name: Annotated[str, Field()]
    description: Annotated[
        str,
        Field(
            description="A short description about this expert.",
            default="No description available.",
        ),
    ]

    def to_yaml(self, path: str) -> str:
        """Write this config to `path` (appending the .yaml suffix if
        missing) and return the path actually written."""

        if not path.endswith(YAML_SUFFIX):
            path += YAML_SUFFIX

        with open(path, "w") as outfile:
            yaml.dump(self.model_dump(mode="json"), outfile, default_flow_style=False)

        return path

    def to_agent(self) -> Agent:
        """The runtime agent runs on FastICRL's eval rendering (inherited via
        AgentSaveState): identity framing, frozen-policy instructions, learned
        strategy, and the experience buffer in the same XML format used during
        training. The one-line ``description`` the Caster picks experts by is
        prepended."""
        return Agent(
            name=self.name,
            system_message=f"{self.description}\n\n{self.eval_system_message()}",
        )

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, "r") as file:
            data = yaml.safe_load(file)
            return ExpertConfig.model_validate(data)
