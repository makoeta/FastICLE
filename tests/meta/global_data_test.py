
from agno.models.base import Model


def test_model(g_data):
    model: Model = g_data["model"]
    
    assert model