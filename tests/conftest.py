import os

from agno.models.base import Model
from agno.models.openai import OpenAIResponses
from dotenv import load_dotenv
import pytest

@pytest.fixture
def g_data() -> dict:
    return {"model": __get_model(provider_key=os.getenv("PROVIDER_KEY", "OPENAI"))}


def __get_model(provider_key: str) -> Model:
    load_dotenv()

    api_key = os.getenv(f"{provider_key}_TEST_API_KEY")

    assert api_key

    match provider_key:
        case "OPENAI":
            return OpenAIResponses(id="gpt-4.1-mini", api_key=api_key)

        case _:
            raise NotImplementedError(f"Provider {provider_key} not supported (yet)!")
