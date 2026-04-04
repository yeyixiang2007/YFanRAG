import pytest

from yfanrag.chat_providers import (
    ChatApiClient,
    ChatApiSettings,
    ChatMessage,
    parse_json_dict,
)


def test_openai_compatible_request_shape():
    client = ChatApiClient()
    settings = ChatApiSettings(
        provider="openai_compatible",
        endpoint="https://example.com/v1/chat/completions",
        model="test-model",
        api_key="secret",
        system_prompt="sys",
        temperature=0.3,
        max_tokens=256,
    )
    endpoint, headers, payload = client.build_request(
        settings,
        [ChatMessage(role="user", content="hello")],
    )

    assert endpoint == "https://example.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert payload["model"] == "test-model"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["content"] == "hello"
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 256


def test_deepseek_request_shape():
    client = ChatApiClient()
    settings = ChatApiSettings(
        provider="deepseek",
        endpoint="https://api.deepseek.com/chat/completions",
        model="deepseek-chat",
        api_key="secret",
    )
    endpoint, headers, payload = client.build_request(
        settings,
        [ChatMessage(role="user", content="hello deepseek")],
    )
    assert endpoint == "https://api.deepseek.com/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert payload["model"] == "deepseek-chat"
    assert payload["messages"][0]["content"] == "hello deepseek"


def test_openai_responses_request_shape():
    client = ChatApiClient()
    settings = ChatApiSettings(
        provider="openai_responses",
        endpoint="https://example.com/v1/responses",
        model="resp-model",
    )
    _, _, payload = client.build_request(
        settings,
        [ChatMessage(role="user", content="hello")],
    )
    assert payload["model"] == "resp-model"
    assert payload["input"][0]["role"] == "user"
    assert payload["input"][0]["content"][0]["type"] == "input_text"


def test_anthropic_request_shape():
    client = ChatApiClient()
    settings = ChatApiSettings(
        provider="anthropic",
        endpoint="https://example.com/v1/messages",
        model="claude-test",
        api_key="secret",
        system_prompt="sys",
    )
    _, headers, payload = client.build_request(
        settings,
        [ChatMessage(role="user", content="hello")],
    )
    assert headers["x-api-key"] == "secret"
    assert headers["anthropic-version"] == "2023-06-01"
    assert payload["system"] == "sys"
    assert payload["messages"][0]["role"] == "user"


def test_extract_openai_choices_text():
    client = ChatApiClient()
    text = client.extract_text(
        "openai_compatible",
        {"choices": [{"message": {"content": "hello world"}}]},
    )
    assert text == "hello world"


def test_extract_openai_responses_text():
    client = ChatApiClient()
    text = client.extract_text(
        "openai_responses",
        {
            "output": [
                {"content": [{"type": "output_text", "text": "hello"}]},
                {"content": [{"type": "output_text", "text": "world"}]},
            ]
        },
    )
    assert "hello" in text
    assert "world" in text


def test_extract_anthropic_text():
    client = ChatApiClient()
    text = client.extract_text(
        "anthropic",
        {"content": [{"type": "text", "text": "reply from anthropic"}]},
    )
    assert text == "reply from anthropic"


def test_extract_stream_delta_openai_compatible():
    client = ChatApiClient()
    delta = client.extract_stream_delta(
        "openai_compatible",
        {"choices": [{"delta": {"content": "hello "}}]},
    )
    assert delta == "hello"


def test_extract_stream_delta_openai_responses():
    client = ChatApiClient()
    delta = client.extract_stream_delta(
        "openai_responses",
        {"type": "response.output_text.delta", "delta": "hi"},
    )
    assert delta == "hi"


def test_extract_stream_delta_anthropic():
    client = ChatApiClient()
    delta = client.extract_stream_delta(
        "anthropic",
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hey"}},
    )
    assert delta == "hey"


def test_parse_json_dict():
    assert parse_json_dict('{"a": 1}', "X") == {"a": 1}
    assert parse_json_dict("", "X") == {}
    with pytest.raises(ValueError):
        parse_json_dict("[]", "X")
