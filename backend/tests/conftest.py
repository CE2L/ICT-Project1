from unittest.mock import MagicMock,patch
import pytest
from fastapi.testclient import TestClient
from main import app
from services import BaseAIService,EvaluationService,OpenAIService


class MockAIService(BaseAIService):
    """
    A fake AI service class that returns fixed values without calling external APIs.
    """
    def fetch_chat_completion(self,prompt:str)->str:
        return "Mocked LLM Analysis Result"

    def get_embedding(self,text:str)->list[float]:
        return [0.1]*1536

    def get_provider_name(self)->str:
        return "Mock AI Service"


@pytest.fixture
def client():
    """
    Injects the mock service only during tests using FastAPI dependency overrides.
    This approach is safer because it does not rely on the internal structure of routes.py
    (such as whether it uses global variables).
    """
    mock_ai=MockAIService()

    import routes

    original_providers = routes.AI_PROVIDERS.copy()

    routes.AI_PROVIDERS["openai"] = mock_ai
    routes.AI_PROVIDERS["friendli"] = mock_ai
    routes.AI_PROVIDERS["gemini"] = mock_ai

    try:
        with TestClient(app) as c:
            yield c
    finally:
        routes.AI_PROVIDERS.update(original_providers)


@pytest.fixture
def evaluation_service():
    """
    For unit tests: uses the real OpenAIService structure,
    while allowing internal logic to be patched per test.
    """
    return EvaluationService(ai_service=OpenAIService())


@pytest.fixture
def mock_ai_service():
    """
    A fixture for cases where manual patching is required.
    """
    with patch("services.OpenAIService.get_embedding") as mock_embed:
        with patch("services.OpenAIService.fetch_chat_completion") as mock_chat:
            mock_embed.return_value=[0.1]*1536

            mock_response=MagicMock()
            mock_response.choices=[
                MagicMock(
                    message=MagicMock(
                        content="Mocked LLM Analysis Result"
                    )
                )
            ]
            mock_chat.return_value=mock_response

            yield{
                "embed":mock_embed,
                "chat":mock_chat
            }