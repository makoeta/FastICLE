from campus.models.reward_output import RewardOutput
from campus.prompts import reward_agent_system_prompt
from campus.prompts import icrl_agent_system_prompt
from dotenv import load_dotenv
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunOutput
from campus.prompts import campus_agent_system_prompt
from campus.models.training_task_list import TrainingTaskList
from agno.agent import Agent
from pathlib import Path
from pydantic import SkipValidation
from pydantic import ConfigDict
from agno.models.base import Model
from typing import Annotated

from pydantic import BaseModel, Field

from campus.models.agent_config import AgentConfig
from fasticrl.icrl_learner import ICRLLearner

from loguru import logger


class Campus(BaseModel):

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    global_task: Annotated[str, Field()]

    save_path: Annotated[str, Field(default="~")]
    auto_save: Annotated[bool, Field(default=True)]
    agent_configs: Annotated[list[AgentConfig], Field(default_factory=list)]

    model: Model

    def get_expert_descriptions(self) -> list[tuple[str, str]]:
        for yaml_file in Path(self.save_path).glob("*.yaml"):
            pass

    def __generate_synth_learning_tasks(self, expert_task):
        task_agent = Agent(
            model=self.model,
            output_schema=TrainingTaskList,
            system_message=campus_agent_system_prompt,
        )

        run_output: RunOutput = task_agent.run(f"""Global task: {self.global_task}

                    Expert task: {expert_task}""")

        task_list: TrainingTaskList = TrainingTaskList.model_validate(
            run_output.content
        )

        return task_list

    def train_new_expert(self, expert_name: str, expert_task: str) -> None:

        expert_name = expert_name.replace(" ", "_").lower()

        logger.debug(f"Creating synthetic tasks for '{expert_task}'...")
        task_list: TrainingTaskList = self.__generate_synth_learning_tasks(expert_task)
        logger.debug(f"...created {len(task_list.tasks)} tasks.")

        def reward_func(model_output: str, task: str) -> int:
            reward_agent = Agent(
                model=self.model,
                system_message=reward_agent_system_prompt.format(
                    expert_task=expert_task, global_task=self.global_task
                ),
                output_schema=RewardOutput,
            )

            output: RunOutput = reward_agent.run(
                f"Task given: {task}\nOutput: {model_output}"
            )

            reward_output: RewardOutput = RewardOutput.model_validate(output.content)
            return reward_output.reward

        learner = ICRLLearner(
            agent=Agent(model=self.model),
            reward_func=reward_func,
            task_description=icrl_agent_system_prompt.format(
                global_task=self.global_task, expert_task=expert_task
            ),
            tasks=task_list.tasks,
        )

        logger.debug(f"Start training")
        learner.auto_learn()
        learner.update_strategy()

        logger.debug("Saving expert...")

        agent_config: AgentConfig = AgentConfig(
            name=expert_name,
            **learner.agent_save_state.model_dump(),
        )
        agent_config.to_yaml(self.save_path + f"/{expert_name}")

        logger.debug(f"Saved expert: {expert_name}")
