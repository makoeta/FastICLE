from _pytest.mark import MarkDecorator
import os
from pathlib import Path

from agno.models.base import Model
from agno.models.openai import OpenAIResponses
from dotenv import load_dotenv
from fasticrl.models.attempt import Attempt
import pytest

from icle.campus.models.expert_config import ExpertConfig

DUMMY_EXPERTS_DIR = Path(__file__).parent / "data" / "dummy_experts"


def _make_dummy_expert(name: str, description: str, task_description: str) -> ExpertConfig:
    return ExpertConfig(
        name=name,
        description=description,
        task_description=task_description,
        strategy=f"Focus on {task_description.lower()} Be vivid, concise, and evocative.",
        buffer=[
            Attempt(task=f"{task_description} (example 1)", output="An example poem.", reward=1),
            Attempt(task=f"{task_description} (example 2)", output="Another example poem.", reward=1),
        ],
    )


# Static, deterministic experts that the non-API unit tests load from disk.
# The real experts are produced by the live-API caster/campus training runs,
# but those are skipped without --run-api, so we synthesize equivalent fixtures
# here. tests/data is gitignored, so this must run on every session.
_DUMMY_EXPERTS = [
    _make_dummy_expert("general_poem_writer", "Writes poems on any subject.", "Write a poem."),
    _make_dummy_expert("nature_poem_writer", "Writes poems about nature.", "Write a nature poem."),
    _make_dummy_expert("relationship_poem_writer", "Writes poems about relationships.", "Write a relationship poem."),
]


@pytest.fixture(scope="session", autouse=True)
def dummy_experts() -> list[ExpertConfig]:
    """Generate the dummy expert YAML fixtures on disk before any test runs."""
    DUMMY_EXPERTS_DIR.mkdir(parents=True, exist_ok=True)
    for expert in _DUMMY_EXPERTS:
        expert.to_yaml(str(DUMMY_EXPERTS_DIR / expert.name))
    return _DUMMY_EXPERTS


@pytest.fixture(scope="session")
def mock_model() -> Model:
    """Lightweight model fixture for unit tests — no real API key required."""
    return OpenAIResponses(id="gpt-4.1-mini", api_key="test-key-unit-tests-only")


@pytest.fixture
def g_data() -> dict:
    return {"model": __get_model(provider_key=os.getenv("PROVIDER_KEY", "OPENAI"))}


def pytest_addoption(parser):
    parser.addoption(
        "--run-api",
        action="store_true",
        default=False,
        help="Runs tests, which require API usage",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "api: marks tests, which require API usage")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-api"):
        return

    skip_api: MarkDecorator = pytest.mark.skip(reason="Needs '--run-api' flag.")
    for item in items:
        if "api" in item.keywords:
            item.add_marker(skip_api)


def __get_model(provider_key: str) -> Model:
    load_dotenv()

    api_key = os.getenv(f"{provider_key}_TEST_API_KEY")

    assert api_key

    match provider_key:
        case "OPENAI":
            return OpenAIResponses(id="gpt-4.1-mini", api_key=api_key)

        case _:
            raise NotImplementedError(f"Provider {provider_key} not supported (yet)!")
