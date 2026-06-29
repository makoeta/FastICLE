import xml.etree.ElementTree as ET

import pytest
from pydantic import ValidationError

from icle.models.tasks import (
    CasterTask,
    CasterTaskList,
    DispatcherTask,
    DispatcherTaskList,
    RuntimeTask,
    RuntimeTaskList,
)


class TestDispatcherTask:
    def test_creation(self):
        task = DispatcherTask(task_id="T1", description="Test task")
        assert task.task_id == "T1"
        assert task.description == "Test task"

    def test_default_depends_on_empty(self):
        task = DispatcherTask(task_id="T1", description="Test task")
        assert task.depends_on == []

    def test_depends_on_set(self):
        task = DispatcherTask(task_id="T2", description="Test task", depends_on=["T1"])
        assert task.depends_on == ["T1"]

    def test_str_contains_class_name(self):
        task = DispatcherTask(task_id="T1", description="Test task")
        assert "DispatcherTask" in str(task)

    def test_str_contains_task_id(self):
        task = DispatcherTask(task_id="T42", description="Test task")
        assert "task_id: T42" in str(task)

    def test_str_contains_description(self):
        task = DispatcherTask(task_id="T1", description="My description")
        assert "description: My description" in str(task)

    def test_repr_equals_str(self):
        task = DispatcherTask(task_id="T1", description="Test task")
        assert repr(task) == str(task)

    def test_to_xml_wraps_in_task_tag(self):
        task = DispatcherTask(task_id="T1", description="Test task")
        xml = task.to_xml()
        assert xml.startswith("<task>")
        assert xml.endswith("</task>")

    def test_to_xml_contains_task_id(self):
        task = DispatcherTask(task_id="T7", description="Test")
        assert "<task_id>T7</task_id>" in task.to_xml()

    def test_to_xml_contains_description(self):
        task = DispatcherTask(task_id="T1", description="Do something")
        assert "<description>Do something</description>" in task.to_xml()

    def test_to_xml_contains_depends_on(self):
        task = DispatcherTask(task_id="T2", description="Test", depends_on=["T1"])
        assert "<depends_on>T1</depends_on>" in task.to_xml()

    def test_to_xml_is_parseable(self):
        task = DispatcherTask(task_id="T1", description="A & B < C > D")
        ET.fromstring(task.to_xml())  # raises ParseError if XML is malformed

    def test_to_xml_preserves_content_when_parsed(self):
        task = DispatcherTask(task_id="T1", description="A & B < C > D")
        root = ET.fromstring(task.to_xml())
        assert root.find("description").text == "A & B < C > D"

    def test_requires_task_id(self):
        with pytest.raises(ValidationError):
            DispatcherTask(description="Missing task_id")

    def test_requires_description(self):
        with pytest.raises(ValidationError):
            DispatcherTask(task_id="T1")


class TestCasterTask:
    def test_inherits_dispatcher_task(self):
        task = CasterTask(task_id="T1", description="Cast task")
        assert isinstance(task, DispatcherTask)

    def test_default_agent_ids_empty(self):
        task = CasterTask(task_id="T1", description="Cast task")
        assert task.agent_ids == []

    def test_agent_ids_set(self):
        task = CasterTask(task_id="T1", description="Cast task", agent_ids=["a1", "a2"])
        assert task.agent_ids == ["a1", "a2"]

    def test_to_xml_inherited(self):
        task = CasterTask(task_id="T1", description="Cast task", agent_ids=["agent"])
        xml = task.to_xml()
        assert "<task>" in xml
        assert "<description>Cast task</description>" in xml

    def test_str_contains_agent_ids(self):
        task = CasterTask(task_id="T1", description="Cast task", agent_ids=["agent1"])
        assert "agent_ids" in str(task)


class TestRuntimeTask:
    def test_inherits_caster_task(self):
        task = RuntimeTask(task_id="T1", description="Run", agent_ids=[], task_output="out")
        assert isinstance(task, CasterTask)
        assert isinstance(task, DispatcherTask)

    def test_task_output_field(self):
        task = RuntimeTask(task_id="T1", description="Run", agent_ids=["a"], task_output="result")
        assert task.task_output == "result"

    def test_requires_task_output(self):
        with pytest.raises(ValidationError):
            RuntimeTask(task_id="T1", description="Run", agent_ids=[])

    def test_to_xml_inherited(self):
        task = RuntimeTask(task_id="T1", description="Run task", agent_ids=[], task_output="out")
        xml = task.to_xml()
        assert "<task>" in xml


class TestDispatcherTaskList:
    def test_empty_list(self):
        tl = DispatcherTaskList(task_list=[])
        assert tl.task_list == []

    def test_multiple_tasks(self):
        tasks = [DispatcherTask(task_id=f"T{i}", description=f"Task {i}") for i in range(3)]
        tl = DispatcherTaskList(task_list=tasks)
        assert len(tl.task_list) == 3

    def test_tasks_preserved_in_order(self):
        ids = ["T1", "T2", "T3", "T4", "T5"]
        tasks = [DispatcherTask(task_id=tid, description=f"Task {tid}") for tid in ids]
        tl = DispatcherTaskList(task_list=tasks)
        for tid, task in zip(ids, tl.task_list):
            assert task.task_id == tid


class TestCasterTaskList:
    def test_creation(self):
        tasks = [CasterTask(task_id="T1", description="t", agent_ids=["a"])]
        tl = CasterTaskList(task_list=tasks)
        assert len(tl.task_list) == 1

    def test_empty_list(self):
        tl = CasterTaskList(task_list=[])
        assert tl.task_list == []


class TestRuntimeTaskList:
    def test_default_is_empty(self):
        tl = RuntimeTaskList()
        assert tl.task_list == []

    def test_to_xml_empty_list(self):
        tl = RuntimeTaskList()
        xml = tl.to_xml()
        assert xml.startswith("<tasks>")
        assert xml.endswith("</tasks>")

    def test_to_xml_wraps_all_tasks(self):
        tasks = [
            RuntimeTask(task_id="T1", description="First", agent_ids=["a"], task_output="out1"),
            RuntimeTask(task_id="T2", description="Second", agent_ids=["b"], task_output="out2"),
        ]
        tl = RuntimeTaskList(task_list=tasks)
        xml = tl.to_xml()
        assert xml.count("<task>") == 2
        assert xml.count("</task>") == 2

    def test_to_xml_contains_task_content(self):
        tasks = [RuntimeTask(task_id="T1", description="Do work", agent_ids=[], task_output="done")]
        tl = RuntimeTaskList(task_list=tasks)
        xml = tl.to_xml()
        assert "Do work" in xml

    def test_to_xml_empty_is_parseable(self):
        tl = RuntimeTaskList()
        root = ET.fromstring(tl.to_xml())
        assert root.tag == "tasks"
        assert len(root) == 0

    def test_append_task(self):
        tl = RuntimeTaskList()
        task = RuntimeTask(task_id="T1", description="t", agent_ids=[], task_output="o")
        tl.task_list.append(task)
        assert len(tl.task_list) == 1
