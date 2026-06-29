from agno.agent import Agent
from agno.models.base import Model

from icle.dispatcher.core import DispatcherAgent
from icle.models.tasks import DispatcherTaskList


def test_instance_name(mock_model):
    dispatcher = DispatcherAgent(mock_model)
    assert dispatcher.name == "Dispatcher"


def test_instance_description(mock_model):
    dispatcher = DispatcherAgent(mock_model)
    assert dispatcher.description
    assert "Dispatcher" in dispatcher.description


def test_instance_model_is_set(mock_model):
    dispatcher = DispatcherAgent(mock_model)
    assert dispatcher.model is not None


def test_instance_model_is_correct_type(mock_model):
    dispatcher = DispatcherAgent(mock_model)
    assert isinstance(dispatcher.model, Model)


def test_instance_is_agent(mock_model):
    dispatcher = DispatcherAgent(mock_model)
    assert isinstance(dispatcher, Agent)


def test_output_schema_is_dispatcher_task_list(mock_model):
    dispatcher = DispatcherAgent(mock_model)
    assert dispatcher.output_schema == DispatcherTaskList
