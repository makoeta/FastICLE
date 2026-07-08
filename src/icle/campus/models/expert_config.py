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

    def to_yaml(self, path: str):

        if not path.endswith(YAML_SUFFIX):
            path += YAML_SUFFIX

        with open(path, "w") as outfile:
            yaml.dump(self.model_dump(mode="json"), outfile, default_flow_style=False)

    def to_agent(self) -> Agent:
        return Agent(name=self.name, system_message=self._build_system_message())

    def _build_system_message(self) -> str:
        """Compose the runtime agent's system message from the persisted
        training artifacts. The one-line ``description`` is what the Caster uses
        to *pick* an expert; here — where the expert actually *executes* — we
        additionally inject the trained ``strategy`` and the experience
        ``buffer`` so the runtime agent runs with what it learned, not just its
        summary."""
        sections: list[str] = [self.description]

        if self.task_description:
            sections.append(f"# Task\n{self.task_description}")

        if self.strategy:
            sections.append(f"# Learned Strategy\n{self.strategy}")

        if self.buffer:
            examples = "\n\n".join(
                f"## Attempt {i} (reward={attempt.reward})\n"
                f"Task: {attempt.task}\n"
                f"Output: {attempt.output}"
                for i, attempt in enumerate(self.buffer, start=1)
            )
            sections.append(f"# Past Attempts (Experience Buffer)\n{examples}")

        return "\n\n".join(sections)

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, "r") as file:
            data = yaml.safe_load(file)
            return ExpertConfig.model_validate(data)
