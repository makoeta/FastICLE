import os
from pathlib import Path
from typing import Annotated

from agno.agent import Agent
from agno.models.base import Model
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunOutput
from fasticrl.icrl_learner import ICRLLearner
from pydantic import BaseModel, ConfigDict, Field
import logging
from campus.models.expert_config import ExpertConfig
from campus.models.reward_output import RewardOutput
from campus.models.training_task_list import TrainingTaskList
from campus.prompts import (
    campus_agent_system_prompt,
    icrl_agent_system_prompt,
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
            system_message=campus_agent_system_prompt,
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

        LOGGER.debug(f"Creating synthetic tasks for '{expert_task}'...")
        task_list: TrainingTaskList = self.__generate_synth_learning_tasks(expert_task)
        LOGGER.debug(f"...created {len(task_list.tasks)} tasks.")

        learner = ICRLLearner(
            learner_model=self.learner_model,
            reward_model=self.reward_model,
            strategy_model=self.strategy_model,
            task_description=icrl_agent_system_prompt.format(
                global_task=self.global_task, expert_task=expert_task
            ),
            tasks=task_list.tasks,
        )

        LOGGER.debug(f"Start training")
        learner.auto_learn()
        learner.update_strategy()

        LOGGER.debug("Saving expert...")

        agent_config: ExpertConfig = ExpertConfig(
            name=expert_name,
            description=description,
            **learner.agent_save_state.model_dump(),
        )
        agent_config.to_yaml(self.expert_save_dir + f"/{expert_name}")

        LOGGER.debug(f"Saved expert: {expert_name}")
