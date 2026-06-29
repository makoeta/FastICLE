import pytest
from pydantic import ValidationError

from icle.campus.models.training_task_list import TrainingTaskList


def test_creation_with_tasks():
    tl = TrainingTaskList(tasks=["Task A", "Task B"])
    assert len(tl.tasks) == 2


def test_tasks_content_preserved():
    tl = TrainingTaskList(tasks=["Do X", "Do Y", "Do Z"])
    assert "Do X" in tl.tasks
    assert "Do Y" in tl.tasks
    assert "Do Z" in tl.tasks


def test_single_task_is_valid():
    tl = TrainingTaskList(tasks=["Only task"])
    assert len(tl.tasks) == 1


def test_empty_task_list_fails_validation():
    with pytest.raises(ValidationError):
        TrainingTaskList(tasks=[])


def test_tasks_field_is_list_of_strings():
    tl = TrainingTaskList(tasks=["Write a poem", "Refine the poem"])
    assert all(isinstance(t, str) for t in tl.tasks)
