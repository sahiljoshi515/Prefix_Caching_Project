"""One retry when a pooled connection closes before /tokenize (spec 003)."""
import io

import httpx
import pytest

from client_simulator import client_sim as client

pytestmark = pytest.mark.m3


def transport(drops):
    calls = []

    def respond(req):
        calls.append(req.url.path)
        if req.url.path == "/tokenize" and len(calls) <= drops:
            raise httpx.RemoteProtocolError("Server disconnected", request=req)
        return httpx.Response(200, json={"count": 3, "max_model_len": 64})

    return httpx.MockTransport(respond), calls


@pytest.mark.asyncio
async def test_a_dropped_connection_is_retried_once():
    mock, calls = transport(drops=1)
    async with httpx.AsyncClient(transport=mock) as http:
        replay = client.ReplayClient(http, "http://test/v1", "test",
                                     io.StringIO())
        assert (await replay.tokenize(prompt="hi"))["count"] == 3
    assert calls == ["/tokenize", "/tokenize"]


@pytest.mark.asyncio
async def test_a_second_drop_is_still_an_error():
    mock, calls = transport(drops=2)
    async with httpx.AsyncClient(transport=mock) as http:
        replay = client.ReplayClient(http, "http://test/v1", "test",
                                     io.StringIO())
        with pytest.raises(httpx.RemoteProtocolError):
            await replay.tokenize(prompt="hi")
    assert calls == ["/tokenize", "/tokenize"]
