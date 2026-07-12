from icle.dispatcher.prompts import DISPATCHER_SYSTEM_PROMPT


class TestDispatcherSystemPrompt:
    def test_prompt_not_empty(self):
        assert DISPATCHER_SYSTEM_PROMPT.strip()

    def test_requires_self_contained_descriptions(self):
        """Downstream agents only ever see the task description, so the
        prompt must forbid lossy references to the original request."""
        assert "SELF-CONTAINED" in DISPATCHER_SYSTEM_PROMPT
        assert "as specified" in DISPATCHER_SYSTEM_PROMPT

    def test_mentions_dependency_declaration(self):
        assert "depends_on" in DISPATCHER_SYSTEM_PROMPT

    def test_requires_one_task_per_deliverable(self):
        """Multi-deliverable requests must be decomposed — self-containment
        must not collapse everything into a single task."""
        assert "one task per deliverable" in DISPATCHER_SYSTEM_PROMPT
        assert "NEVER a reason to merge" in DISPATCHER_SYSTEM_PROMPT
