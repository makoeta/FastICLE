import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from models.tasks import CasterTask, CasterTaskList, RuntimeTask
from runtime.core import Runtime

LOGGER = __import__("logging").getLogger(__name__)

DUMMY_EXPERTS_DIR = "./tests/data/dummy_experts"


# --- Helpers ---

def make_task(task_id: str, description: str = "task", depends_on: list[str] | None = None, agent_ids: list[str] | None = None) -> CasterTask:
    return CasterTask(task_id=task_id, description=description, depends_on=depends_on or [], agent_ids=agent_ids or [])


def make_step_input(tasks: list[CasterTask]):
    step_input = MagicMock()
    step_input.get_last_step_content.return_value = CasterTaskList(task_list=tasks)
    return step_input


def fake_run_task(task: CasterTask, _prior_context: str = "") -> RuntimeTask:
    return RuntimeTask(**task.model_dump(), task_output=f"output_for_{task.description}")


@pytest.fixture
def runtime(g_data):
    return Runtime(expert_save_dir=DUMMY_EXPERTS_DIR, model=g_data["model"])


# --- Unit tests ---

def test_independent_tasks_run_concurrently(runtime):
    """Tasks with no shared dependencies must execute in parallel threads."""
    barrier = threading.Barrier(2, timeout=3.0)

    def rendezvous_task(task, _prior_context: str = ""):
        # Both threads must reach this point before either can proceed —
        # raises BrokenBarrierError if tasks run sequentially.
        barrier.wait()
        return RuntimeTask(**task.model_dump(), task_output="done")

    tasks = [make_task("T1", "a"), make_task("T2", "b")]
    step_input = make_step_input(tasks)

    with patch.object(runtime, "_run_task", side_effect=rendezvous_task):
        result = runtime.runtime(step_input)

    assert len(result.content.task_list) == 2


def test_dependent_task_waits_for_dependencies(runtime):
    """T3 (depends on T1, T2) must not start until both T1 and T2 finish."""
    completed_ids: list[str] = []
    lock = threading.Lock()

    def tracking_task(task, _prior_context: str = ""):
        time.sleep(0.02)
        with lock:
            completed_ids.append(task.task_id)
        return RuntimeTask(**task.model_dump(), task_output="done")

    tasks = [
        make_task("T1", "first_a"),
        make_task("T2", "first_b"),
        make_task("T3", "second", depends_on=["T1", "T2"]),
    ]
    step_input = make_step_input(tasks)

    with patch.object(runtime, "_run_task", side_effect=tracking_task):
        runtime.runtime(step_input)

    t3_index = completed_ids.index("T3")
    assert "T1" in completed_ids[:t3_index]
    assert "T2" in completed_ids[:t3_index]


def test_output_respects_dependency_order(runtime):
    """Each task must appear in the output after all its dependencies."""
    tasks = [
        make_task("T3", "third",  depends_on=["T1", "T2"]),
        make_task("T1", "first",  depends_on=[]),
        make_task("T2", "second", depends_on=["T1"]),
    ]
    step_input = make_step_input(tasks)

    with patch.object(runtime, "_run_task", side_effect=fake_run_task):
        result = runtime.runtime(step_input)

    task_ids = [t.task_id for t in result.content.task_list]
    for task in tasks:
        task_index = task_ids.index(task.task_id)
        for dep in task.depends_on:
            assert task_ids.index(dep) < task_index


def test_output_preserves_task_output(runtime):
    """RuntimeTask.task_output must carry through from _run_task."""
    tasks = [make_task("T1", "x"), make_task("T2", "y")]
    step_input = make_step_input(tasks)

    with patch.object(runtime, "_run_task", side_effect=fake_run_task):
        result = runtime.runtime(step_input)

    outputs = {t.task_output for t in result.content.task_list}
    assert outputs == {"output_for_x", "output_for_y"}


def test_cycle_raises(runtime):
    """A dependency cycle must raise ValueError rather than deadlock."""
    tasks = [
        make_task("T1", "a", depends_on=["T2"]),
        make_task("T2", "b", depends_on=["T1"]),
    ]
    step_input = make_step_input(tasks)

    with pytest.raises(ValueError, match="cycle"):
        runtime.runtime(step_input)


def test_single_task(runtime):
    tasks = [make_task("T1", "only")]
    step_input = make_step_input(tasks)

    with patch.object(runtime, "_run_task", side_effect=fake_run_task):
        result = runtime.runtime(step_input)

    assert len(result.content.task_list) == 1
    assert result.content.task_list[0].task_output == "output_for_only"


def test_empty_task_list(runtime):
    step_input = make_step_input([])

    result = runtime.runtime(step_input)

    assert result.content.task_list == []


# --- API test ---

@pytest.mark.api
def test_runtime_run_parallel(g_data):
    """T1 and T2 run concurrently; T3 waits for both and receives their outputs as context."""
    runtime = Runtime(expert_save_dir=DUMMY_EXPERTS_DIR, model=g_data["model"])

    tasks = [
        make_task("T1", "Write a short nature poem.",       depends_on=[],          agent_ids=["nature_poem_writer"]),
        make_task("T2", "Write a short relationship poem.", depends_on=[],          agent_ids=["relationship_poem_writer"]),
        make_task("T3", "Write a general poem that continues and combines the poems written above.", depends_on=["T1", "T2"], agent_ids=["general_poem_writer"]),
    ]
    step_input = make_step_input(tasks)

    result = runtime.runtime(step_input)

    task_list = result.content.task_list
    assert len(task_list) == 3
    assert all(t.task_output for t in task_list)

    task_ids = [t.task_id for t in task_list]
    assert task_ids.index("T3") > task_ids.index("T1")
    assert task_ids.index("T3") > task_ids.index("T2")

    LOGGER.info("Task outputs:")
    for t in task_list:
        LOGGER.info(f"  [{t.task_id}] {t.description[:40]} → {t.task_output[:60]}...")
