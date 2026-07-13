import logging
import os
import time
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
    EXPERT_TASK_DESCRIPTION_PROMPT,
)

logger = logging.getLogger(__name__)


class Campus(BaseModel):

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    global_task: Annotated[str, Field()]

    expert_save_dir: Annotated[
        str, Field(default_factory=lambda: os.environ.get("EXPERT_SAVE_DIR", "./"))
    ]
    auto_save: Annotated[bool, Field(default=True)]
    # Experts trained with auto_save disabled; they exist only for the
    # lifetime of this Campus instance and are never written to disk.
    in_memory_experts: Annotated[list[ExpertConfig], Field(default_factory=list)]

    learner_model: Model
    reward_model: Model
    strategy_model: Model

    task_generator_model: Model

    def get_experts(self) -> list[ExpertConfig]:
        experts_by_name: dict[str, ExpertConfig] = {}

        for yaml_file in Path(self.expert_save_dir).glob("*.yaml"):
            try:
                expert_config: ExpertConfig = ExpertConfig.from_yaml(yaml_file)
                experts_by_name[expert_config.name] = expert_config
            except Exception:
                logger.warning(
                    "Could not load expert config (%s)",
                    yaml_file.absolute(),
                    exc_info=True,
                )

        # Experts trained with auto_save disabled only exist in memory;
        # on a name clash they take precedence over saved ones.
        for expert_config in self.in_memory_experts:
            experts_by_name[expert_config.name] = expert_config

        return list(experts_by_name.values())

    def __generate_synth_learning_tasks(self, expert_task):
        task_agent = Agent(
            model=self.task_generator_model,
            output_schema=TrainingTaskList,
            system_message=CAMPUS_AGENT_SYSTEM_PROMPT,
        )
        logger.debug(
            "Campus curriculum system prompt:\n%s", CAMPUS_AGENT_SYSTEM_PROMPT
        )

        run_output: RunOutput = task_agent.run(f"""Global task: {self.global_task}

                    Expert task: {expert_task}""")

        task_list: TrainingTaskList = TrainingTaskList.model_validate(
            run_output.content
        )

        for task in task_list.tasks:
            logger.debug(
                "Training task: %s | Relevance: %s",
                task.task,
                task.relevance_justification,
            )

        return task_list

    def has_expert(self, expert_name: str) -> bool:
        """True if an expert with this (already normalized) name exists on
        disk or in memory."""
        if any(e.name == expert_name for e in self.in_memory_experts):
            return True
        return (Path(self.expert_save_dir) / f"{expert_name}.yaml").is_file()

    def train_new_expert(
        self, expert_name: str, expert_task: str, description: str
    ) -> str:
        """Use this method to train a new expert for the campus.

        Args:
            expert_name (str): Name of the expert. Used later for task allocation
            expert_task (str): Task the expert is the specialist for.

        Returns:
            The normalized expert name (the assignable ID).
        """

        expert_name = expert_name.replace(" ", "_").lower()

        # Retraining an existing expert wastes tokens and silently overwrites
        # its learned state — reuse it instead. LLM callers cannot be trusted
        # to check the pool reliably, so this guard is enforced in code.
        if self.has_expert(expert_name):
            logger.info(
                "Expert '%s' already exists — skipping training, reusing it.",
                expert_name,
            )
            return expert_name

        logger.info("Training new expert: %s", expert_name)
        task_list: TrainingTaskList = self.__generate_synth_learning_tasks(expert_task)

        learner = ICRLLearner(
            learner_model=self.learner_model,
            reward_model=self.reward_model,
            strategy_model=self.strategy_model,
            task_description=EXPERT_TASK_DESCRIPTION_PROMPT.format(
                global_task=self.global_task, expert_task=expert_task
            ),
            tasks=task_list.task_prompts,
        )

        start = time.perf_counter()
        learner.auto_learn()
        learner.update_strategy()
        elapsed = time.perf_counter() - start
        logger.info(
            "Trained new expert '%s' on %d task(s) in %.1fs",
            expert_name,
            len(task_list.tasks),
            elapsed,
        )

        agent_config: ExpertConfig = ExpertConfig(
            name=expert_name,
            description=description,
            **learner.agent_save_state.model_dump(),
        )
        if self.auto_save:
            save_path = agent_config.to_yaml(self.expert_save_dir + f"/{expert_name}")
            logger.info("Expert '%s' saved to %s", expert_name, save_path)
        else:
            self.in_memory_experts.append(agent_config)
            logger.info(
                "Auto-save disabled, expert '%s' is kept in memory only",
                expert_name,
            )

        return expert_name
