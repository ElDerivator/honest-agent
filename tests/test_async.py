import asyncio
import os
import tempfile

from honest_agent import EventLog, gated, honest_episode_async


def _log() -> EventLog:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return EventLog(path)


def test_async_context_manager_unverified():
    log = _log()

    async def go():
        async with honest_episode_async(vertical="t", trigger="x", log=log) as o:
            pass
        return o.status

    assert asyncio.run(go()) == "UNVERIFIED"


def test_async_context_manager_with_proof():
    log = _log()

    async def go():
        async with honest_episode_async(vertical="t", trigger="x", log=log) as o:
            o.verify("unit_test", "tests/")
        return o.status

    assert asyncio.run(go()) == "COMPLETED"


def test_async_context_manager_exception_is_failed():
    log = _log()

    async def go():
        try:
            async with honest_episode_async(vertical="t", trigger="x", log=log) as o:
                raise RuntimeError("boom")
        except RuntimeError:
            return o.status

    assert asyncio.run(go()) == "FAILED"


def test_async_gated_decorator():
    log = _log()

    @gated(vertical="t", trigger="x", log=log)
    async def work(ok: bool):
        return {"type": "x", "ref": "y", "ok": True} if ok else None

    assert asyncio.run(work(True))[1] == "COMPLETED"
    assert asyncio.run(work(False))[1] == "UNVERIFIED"
