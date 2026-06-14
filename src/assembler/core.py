from agno.run.agent import RunOutput
from agno.workflow import StepOutput
from agno.workflow import StepInput
from pydantic import BaseModel
from agno.models.base import Model
from assembler.prompts import ASSEMBLER_AGENT_PROMPT
from agno.agent import Agent
import logging

ASSEMBLER_INPUT_PROMPT = """
<user_input>
{user_input}
</user_input>

<sub_agent_outputs>
{sub_agent_outputs}
</sub_agent_outputs>

"""

LOGGER = logging.getLogger(__name__)


class Assembler(BaseModel):

    model: Model

    def assemble(self, step_input: StepInput) -> StepOutput:

        assembler_agent: Agent = AssemblerAgent(model=self.model)

        assembler_input_prompt: str = ASSEMBLER_INPUT_PROMPT.format(
            user_input=step_input.get_input_as_string(),
            sub_agent_outputs=step_input.get_last_step_content().to_xml(),
        )

        LOGGER.info(assembler_input_prompt)

        assembler_out: RunOutput = assembler_agent.run(assembler_input_prompt)

        return StepOutput(content=assembler_out.content)


class AssemblerAgent(Agent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.system_message = ASSEMBLER_AGENT_PROMPT
