from agno.models.base import Model
from agno.agent import Agent

from dispatcher.dispatcher import DispatcherFactory

def test_instance_name(g_data):
    dispatcher: Agent = DispatcherFactory.create_dispatcher(g_data["model"])
    assert dispatcher.name

def test_instance_description(g_data):
    dispatcher: Agent = DispatcherFactory.create_dispatcher(g_data["model"])
    assert dispatcher.description

def test_instance_model(g_data):
    dispatcher: Agent = DispatcherFactory.create_dispatcher(g_data["model"])
    assert dispatcher.model
    
def test_instance_is_instance(g_data):
    dispatcher: Agent = DispatcherFactory.create_dispatcher(g_data["model"])
    assert isinstance(dispatcher.model, Model)

