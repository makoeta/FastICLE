from icle.caster.prompts import (
    CASTING_AGENT_SYSTEM_PROMPT,
    CASTING_MULTI_MODE_SUB_PROMPT,
    CASTING_SINGLE_MODE_SUB_PROMPT,
    build_casting_prompt,
)


class TestBuildCastingPromptSingleMode:
    def test_contains_global_task(self):
        prompt = build_casting_prompt(
            global_task="Write poems",
            available_experts="",
            multi_expert_mode=False,
        )
        assert "Write poems" in prompt

    def test_contains_single_mode_sub_prompt(self):
        prompt = build_casting_prompt(
            global_task="task",
            available_experts="",
            multi_expert_mode=False,
        )
        assert "SINGLE-EXPERT MODE" in prompt

    def test_does_not_contain_multi_mode_sub_prompt(self):
        prompt = build_casting_prompt(
            global_task="task",
            available_experts="",
            multi_expert_mode=False,
        )
        assert "MULTI-EXPERT MODE" not in prompt

    def test_single_mode_assignment_rule(self):
        prompt = build_casting_prompt(
            global_task="task",
            available_experts="",
            multi_expert_mode=False,
        )
        assert "exactly ONE specialist ID" in prompt

    def test_contains_available_experts(self):
        experts = "<expert><id>poet</id></expert>"
        prompt = build_casting_prompt(
            global_task="task",
            available_experts=experts,
            multi_expert_mode=False,
        )
        assert "poet" in prompt

    def test_no_unfilled_format_placeholders(self):
        prompt = build_casting_prompt(
            global_task="Write poems",
            available_experts="<expert>...</expert>",
            multi_expert_mode=False,
        )
        # After formatting, no raw Python format placeholders should remain
        import re
        assert not re.search(r"\{[a-z_]+\}", prompt)


class TestBuildCastingPromptMultiMode:
    def test_contains_global_task(self):
        prompt = build_casting_prompt(
            global_task="Travel blog",
            available_experts="",
            multi_expert_mode=True,
        )
        assert "Travel blog" in prompt

    def test_contains_multi_mode_sub_prompt(self):
        prompt = build_casting_prompt(
            global_task="task",
            available_experts="",
            multi_expert_mode=True,
        )
        assert "MULTI-EXPERT MODE" in prompt

    def test_does_not_contain_single_mode_sub_prompt(self):
        prompt = build_casting_prompt(
            global_task="task",
            available_experts="",
            multi_expert_mode=True,
        )
        assert "SINGLE-EXPERT MODE" not in prompt

    def test_multi_mode_requires_general_expert(self):
        prompt = build_casting_prompt(
            global_task="task",
            available_experts="",
            multi_expert_mode=True,
        )
        assert "general expert" in prompt.lower()

    def test_no_unfilled_format_placeholders(self):
        prompt = build_casting_prompt(
            global_task="Write poems",
            available_experts="<expert>...</expert>",
            multi_expert_mode=True,
        )
        import re
        assert not re.search(r"\{[a-z_]+\}", prompt)


class TestPromptConstants:
    def test_single_mode_sub_prompt_not_empty(self):
        assert len(CASTING_SINGLE_MODE_SUB_PROMPT) > 0

    def test_multi_mode_sub_prompt_not_empty(self):
        assert len(CASTING_MULTI_MODE_SUB_PROMPT) > 0

    def test_system_prompt_contains_required_placeholders(self):
        # Verify the template has all expected format keys
        assert "{mode_sub_prompt}" in CASTING_AGENT_SYSTEM_PROMPT
        assert "{global_task}" in CASTING_AGENT_SYSTEM_PROMPT
        assert "{available_experts}" in CASTING_AGENT_SYSTEM_PROMPT
        assert "{task_scope}" in CASTING_AGENT_SYSTEM_PROMPT
        assert "{assignment_rule}" in CASTING_AGENT_SYSTEM_PROMPT
