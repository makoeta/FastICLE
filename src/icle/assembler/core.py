from agno.run import RunContext
from agno.run.agent import RunOutput
from agno.workflow import StepOutput
from agno.workflow import StepInput
from agno.models.base import Model
from icle.assembler.prompts import ASSEMBLER_AGENT_PROMPT
from icle.models.tasks import RuntimeTaskList
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
        logger.info("Starting assembling...")

        runtime_task_list: RuntimeTaskList = step_input.get_last_step_content()

        # Single-node graph: there is nothing to synthesize across. Return the
        # sole task's output verbatim instead of paying an extra LLM call that
        # would, at best, paraphrase an already-complete answer.
        if len(runtime_task_list.task_list) == 1:
            logger.info("Single task detected — skipping synthesis, returning output directly.")
            return StepOutput(content=runtime_task_list.task_list[0].task_output)

        # StepInput.input carries the workflow's original input into every step.
        user_input = str(step_input.input) if step_input.input else ""

        assembler_input_prompt: str = ASSEMBLER_INPUT_PROMPT.format(
            user_input=user_input,
            sub_agent_outputs=runtime_task_list.to_xml(),
        )

        logger.debug("Assembler input:\n%s", assembler_input_prompt)

        assembler_out: RunOutput = self.assembler_agent.run(assembler_input_prompt)

        if assembler_out.metrics:
            usage = run_context.session_state.setdefault(
                "token_usage", {"input_tokens": 0, "output_tokens": 0}
            )
            usage["input_tokens"] += assembler_out.metrics.input_tokens or 0
            usage["output_tokens"] += assembler_out.metrics.output_tokens or 0

        logger.info("Assembling finished.")
        logger.debug("Assembled output:\n%s", assembler_out.content)
        return StepOutput(content=assembler_out.content)
