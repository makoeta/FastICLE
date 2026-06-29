import logging
import os
from pathlib import Path
from typing import Annotated

from agno.agent import Agent
from agno.models.base import Model
from agno.run.agent import RunOutput
from fasticrl.icrl_learner import ICRLLearner
from pydantic import BaseModel, ConfigDict, Field

from icle.campus.models.expert_config import ExpertConfig
from icle.campus.models.training_task_list import TrainingTaskList
from icle.campus.prompts import (
    CAMPUS_AGENT_SYSTEM_PROMPT,
    ICRL_AGENT_SYSTEM_PROMPT,
)

LOGGER = logging.getLogger(__name__)


class Campus(BaseModel):

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    global_task: Annotated[str, Field()]

    expert_save_dir: Annotated[
        str, Field(default_factory=lambda: os.environ.get("EXPERT_SAVE_DIR", "./"))
    ]
    auto_save: Annotated[bool, Field(default=True)]
    agent_configs: Annotated[list[ExpertConfig], Field(default_factory=list)]

    learner_model: Model
    reward_model: Model
    strategy_model: Model

    task_generator_model: Model

    def get_experts(self) -> list[ExpertConfig]:
        expert_configs: list[ExpertConfig] = list()

        for yaml_file in Path(self.expert_save_dir).glob("*.yaml"):
            try:
                expert_config: ExpertConfig = ExpertConfig.from_yaml(yaml_file)
                expert_configs.append(expert_config)
            except Exception as e:
                LOGGER.warning(
                    f"Could not load expert config ({str(yaml_file.absolute())}): {e}"
                )

        return expert_configs

    def __generate_synth_learning_tasks(self, expert_task):
        task_agent = Agent(
            model=self.task_generator_model,
            output_schema=TrainingTaskList,
            system_message=CAMPUS_AGENT_SYSTEM_PROMPT,
        )

        run_output: RunOutput = task_agent.run(f"""Global task: {self.global_task}

                    Expert task: {expert_task}""")

        task_list: TrainingTaskList = TrainingTaskList.model_validate(
            run_output.content
        )

        return task_list

    def train_new_expert(
        self, expert_name: str, expert_task: str, description: str
    ) -> None:
        """Use this method to train a new expert for the campus.

        Args:
            expert_name (str): Name of the expert. Used later for task allocation
            expert_task (str): Task the expert is the specialist for.
        """

        expert_name = expert_name.replace(" ", "_").lower()
        LOGGER.info(f"New expert is being created: {expert_name}")
        task_list: TrainingTaskList = self.__generate_synth_learning_tasks(expert_task)

        learner = ICRLLearner(
            learner_model=self.learner_model,
            reward_model=self.reward_model,
            strategy_model=self.strategy_model,
            task_description=ICRL_AGENT_SYSTEM_PROMPT.format(
                global_task=self.global_task, expert_task=expert_task
            ),
            tasks=task_list.tasks,
        )

        learner.auto_learn()
        learner.update_strategy()

        agent_config: ExpertConfig = ExpertConfig(
            name=expert_name,
            description=description,
            **learner.agent_save_state.model_dump(),
        )
        agent_config.to_yaml(self.expert_save_dir + f"/{expert_name}")

        LOGGER.info(f"{expert_name.capitalize()} was created.")
