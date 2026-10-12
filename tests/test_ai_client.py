import asyncio
import socket

import httpx
import pytest

from backend.app import ai_client
from backend.app.schemas import ApiConfig, ChatMessage
from backend.app.version import __version__


def mock_dns(monkeypatch, addresses):
    async def resolve(self, host, port, **kwargs):
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, port),
            )
            for address in addresses
        ]

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolve)


def capture_requests(monkeypatch, sent):
    def make_transport(**kwargs):
        def handler(request):
            sent.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        return httpx.MockTransport(handler)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", make_transport)


def config(base_url: str, model: str = "deepseek-v4.1-flash") -> ApiConfig:
    return ApiConfig(
        provider_name="OpenCode Go",
        base_url=base_url,
        api_key="TEST_ONLY_API_KEY",
        model=model,
    )


async def test_ai_requests_identify_kaobuddy_instead_of_the_http_library(monkeypatch):
    mock_dns(monkeypatch, ["93.184.216.34"])
    sent = []
    capture_requests(monkeypatch, sent)

    await ai_client.chat_completion(config("https://api.deepseek.com"), [ChatMessage(role="user", content="hi")])

    assert sent[0].headers["user-agent"] == f"KaoBuddy/{__version__}"
    assert "python-httpx" not in sent[0].headers["user-agent"]


async def test_opencode_gateway_receives_the_session_id(monkeypatch):
    mock_dns(monkeypatch, ["93.184.216.34"])
    sent = []
    capture_requests(monkeypatch, sent)

    await ai_client.chat_completion(
        config("https://opencode.ai/zen/v1"),
        [ChatMessage(role="user", content="hi")],
        "project_5c36b277",
    )

    assert sent[0].headers["x-opencode-session"] == "project_5c36b277"
    assert sent[0].headers["host"] == "opencode.ai"


async def test_other_providers_never_see_the_session_id(monkeypatch):
    mock_dns(monkeypatch, ["93.184.216.34"])
    sent = []
    capture_requests(monkeypatch, sent)

    await ai_client.chat_completion(
        config("https://api.moonshot.cn/v1", model="kimi-k2.6"),
        [ChatMessage(role="user", content="hi")],
        "project_5c36b277",
    )

    assert "x-opencode-session" not in sent[0].headers


async def test_streaming_cards_also_carry_the_session_id(monkeypatch):
    mock_dns(monkeypatch, ["93.184.216.34"])
    sent = []

    def make_transport(**kwargs):
        def handler(request):
            sent.append(request)
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text='data: {"choices":[{"delta":{"content":"卡"}}]}\n\ndata: [DONE]\n\n',
            )

        return httpx.MockTransport(handler)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", make_transport)

    chunks = [
        chunk
        async for chunk in ai_client.chat_completion_stream(
            config("https://opencode.ai/zen/v1"), [ChatMessage(role="user", content="hi")], "project_abc"
        )
    ]

    assert chunks == ["卡"]
    assert sent[0].headers["x-opencode-session"] == "project_abc"


@pytest.mark.parametrize("base_url", ["https://opencode.ai", "https://opencode.ai/zen/v1"])
async def test_gateway_host_matching_accepts_the_plain_host_and_subdomains(monkeypatch, base_url):
    assert ai_client._is_opencode_gateway(base_url) is True


@pytest.mark.parametrize("base_url", ["https://notopencode.ai/zen/v1", "https://zen.opencode.ai.evil.test/v1"])
async def test_gateway_host_matching_rejects_lookalike_hosts(base_url):
    assert ai_client._is_opencode_gateway(base_url) is False
