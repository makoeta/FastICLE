from agno.run import RunContext
from agno.run.agent import RunOutput
from agno.workflow import StepOutput
from agno.workflow import StepInput
from pydantic import BaseModel
from agno.models.base import Model
from icle.assembler.prompts import ASSEMBLER_AGENT_PROMPT
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

logger = logging.getLogger(__name__)


class AssemblerAgent(Agent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.system_message = ASSEMBLER_AGENT_PROMPT


class Assembler:

    assembler_agent: AssemblerAgent

    def __init__(self, model: Model):
        self.assembler_agent = AssemblerAgent(model=model)

    def assemble(self, step_input: StepInput, run_context: RunContext) -> StepOutput:
        logger.debug("Starting assembling...")
        user_input = run_context.session_state.get("user_input", "")

        assembler_input_prompt: str = ASSEMBLER_INPUT_PROMPT.format(
            user_input=user_input,
            sub_agent_outputs=step_input.get_last_step_content().to_xml(),
        )

        assembler_out: RunOutput = self.assembler_agent.run(assembler_input_prompt)

        if assembler_out.metrics:
            usage = run_context.session_state.setdefault(
                "token_usage", {"input_tokens": 0, "output_tokens": 0}
            )
            usage["input_tokens"] += assembler_out.metrics.input_tokens or 0
            usage["output_tokens"] += assembler_out.metrics.output_tokens or 0

        logger.debug("Assembling finished.")
        return StepOutput(content=assembler_out.content)
