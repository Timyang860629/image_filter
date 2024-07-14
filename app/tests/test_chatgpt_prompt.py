import openai
import pytest

from app.chatgpt.chatgpt_prompt import generate_response


# Mocking openai.Completion.create to avoid actual API calls during testing
class MockResponse:
    def __init__(self, text):
        self.text = text

    @property
    def choices(self):
        return [self]


def mock_openai_completion_create(engine, prompt, max_tokens, temperature, top_p):
    return MockResponse(f"Mock response for prompt: {prompt}")


def test_generate_response(monkeypatch):
    # Use monkeypatch to replace openai.Completion.create with our mock function
    monkeypatch.setattr(openai.Completion, "create", mock_openai_completion_create)

    prompt = "Explain the concept of machine learning in simple terms."
    expected_response = "Mock response for prompt: Explain the concept of machine learning in simple terms."

    response = generate_response(prompt)
    assert response == expected_response
