"""Fixtures shared by the eviction-policy tests."""
import pytest

from vllm.core.block import cache_events


@pytest.fixture
def request_context():
    """The module's request context, which the leaf evictors switch on and
    tests set, restored after the test; no event time is frozen during it
    (`set_current_request` freezes one)."""
    ctx = cache_events._context
    saved = ctx.enabled, ctx.request_id, ctx.op, ctx.time
    ctx.time = None
    yield ctx
    ctx.enabled, ctx.request_id, ctx.op, ctx.time = saved
