"""Regression coverage for consecutive auxiliary endpoint timeouts (#141016)."""

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import auxiliary_client as aux


PRIMARY_URL = "http://127.0.0.1:8001/v1"
MESSAGES = [{"role": "user", "content": "describe the image"}]


def _response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _full_budget_timeout(clock: list[float]):
    def fail(**_kwargs):
        clock[0] += 1.0
        raise TimeoutError("Request timed out.")
    return fail


def _async_full_budget_timeout(clock: list[float]):
    async def fail(**_kwargs):
        clock[0] += 1.0
        raise TimeoutError("Request timed out.")
    return fail


@pytest.fixture(autouse=True)
def _reset_auxiliary_health():
    aux._reset_aux_unhealthy_cache()
    yield
    aux._reset_aux_unhealthy_cache()


def test_consecutive_timeout_quarantine_skips_endpoint_and_success_resets(monkeypatch):
    """Two full-budget timeouts trip the breaker, while one timeout after a success does not."""
    now = [0.0]
    monotonic = [0.0]
    monkeypatch.setattr(aux.time, "time", lambda: now[0])
    monkeypatch.setattr(aux.time, "monotonic", lambda: monotonic[0])

    primary = MagicMock(base_url=PRIMARY_URL)
    primary.chat.completions.create.side_effect = _full_budget_timeout(monotonic)
    fallback = MagicMock(base_url="https://fallback.example/v1")
    fallback.chat.completions.create.return_value = _response("fallback")

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "local-model", PRIMARY_URL, "local", None),
        ),
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, "local-model"),
        ) as get_primary,
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(fallback, "fallback-model", "fallback_chain[0](custom)"),
        ),
    ):
        for _ in range(2):
            result = aux.call_llm(
                task="compression", messages=MESSAGES, timeout=1.0
            )
            assert result.choices[0].message.content == "fallback"

        assert primary.chat.completions.create.call_count == 2

        # The third call is served by the fallback without dialing the quarantined endpoint.
        aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)
        assert primary.chat.completions.create.call_count == 2
        assert get_primary.call_count == 2

        # Raw streaming stays on its requested route and outside the completion breaker.
        marker = object()
        with patch("agent.auxiliary_client._relay_sync_stream", return_value=marker):
            assert aux.call_llm(
                task="compression", messages=MESSAGES, timeout=1.0, stream=True
            ) is marker
        assert get_primary.call_count == 3

        # A successful half-open probe clears the timeout streak.
        now[0] = 61.0
        primary.chat.completions.create.side_effect = None
        primary.chat.completions.create.return_value = _response("primary")
        result = aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)
        assert result.choices[0].message.content == "primary"

        # After that success, one timeout is ambiguous and must not quarantine the endpoint.
        primary.chat.completions.create.side_effect = _full_budget_timeout(monotonic)
        aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)
        assert primary.chat.completions.create.call_count == 4
        aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)
        assert primary.chat.completions.create.call_count == 5

        # The second consecutive timeout re-trips the breaker.
        aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)
        assert primary.chat.completions.create.call_count == 5


@pytest.mark.asyncio
async def test_expired_quarantine_allows_one_probe_then_retrips(monkeypatch):
    """A timed-out half-open probe re-trips immediately on the shared async vision path."""
    now = [0.0]
    monotonic = [0.0]
    monkeypatch.setattr(aux.time, "time", lambda: now[0])
    monkeypatch.setattr(aux.time, "monotonic", lambda: monotonic[0])

    primary = MagicMock(base_url=PRIMARY_URL)
    primary.chat.completions.create = AsyncMock(side_effect=_async_full_budget_timeout(monotonic))
    fallback = MagicMock(base_url="https://fallback.example/v1")
    fallback.chat.completions.create = AsyncMock(return_value=_response("fallback"))

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "local-model", PRIMARY_URL, "local", None),
        ),
        patch(
            "agent.auxiliary_client.resolve_vision_provider_client",
            return_value=("custom", primary, "local-model"),
        ) as resolve_primary,
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(fallback, "fallback-model", "fallback_chain[0](custom)"),
        ),
        patch(
            "agent.auxiliary_client._to_async_client",
            return_value=(fallback, "fallback-model"),
        ),
    ):
        for _ in range(2):
            result = await aux.async_call_llm(
                task="vision", messages=MESSAGES, timeout=1.0
            )
            assert result.choices[0].message.content == "fallback"

        await aux.async_call_llm(task="vision", messages=MESSAGES, timeout=1.0)
        assert primary.chat.completions.create.call_count == 2
        assert resolve_primary.call_count == 2

        # Cooldown expiry permits exactly one probe. Its timeout re-trips immediately.
        now[0] = 61.0
        await aux.async_call_llm(task="vision", messages=MESSAGES, timeout=1.0)
        assert primary.chat.completions.create.call_count == 3
        await aux.async_call_llm(task="vision", messages=MESSAGES, timeout=1.0)
        assert primary.chat.completions.create.call_count == 3


def test_cooldown_expiry_admits_exactly_one_concurrent_probe(monkeypatch):
    """Concurrent calls during half-open state share one probe instead of stampeding the endpoint."""
    now = [0.0]
    monotonic = [0.0]
    monkeypatch.setattr(aux.time, "time", lambda: now[0])
    monkeypatch.setattr(aux.time, "monotonic", lambda: monotonic[0])
    primary = MagicMock(base_url=PRIMARY_URL)
    primary.chat.completions.create.side_effect = _full_budget_timeout(monotonic)
    fallback = MagicMock(base_url="https://fallback.example/v1")
    fallback.chat.completions.create.return_value = _response("fallback")
    probe_started = threading.Event()
    release_probe = threading.Event()
    probe_mode = [False]
    probe_resolutions = [0]

    def resolve_primary(*_args, **_kwargs):
        if probe_mode[0]:
            probe_resolutions[0] += 1
            probe_started.set()
            assert release_probe.wait(timeout=2.0)
        return primary, "local-model"

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("custom", "local-model", PRIMARY_URL, "local", None),
        ),
        patch("agent.auxiliary_client._get_cached_client", side_effect=resolve_primary),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(fallback, "fallback-model", "fallback_chain[0](custom)"),
        ),
    ):
        for _ in range(2):
            aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)

        now[0] = 61.0
        probe_mode[0] = True
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                aux.call_llm, task="compression", messages=MESSAGES, timeout=1.0
            )
            assert probe_started.wait(timeout=2.0)
            second = pool.submit(
                aux.call_llm, task="compression", messages=MESSAGES, timeout=1.0
            )
            try:
                second_result = second.result(timeout=2.0)
            finally:
                release_probe.set()
            first_result = first.result(timeout=2.0)

    assert first_result.choices[0].message.content == "fallback"
    assert second_result.choices[0].message.content == "fallback"
    assert probe_resolutions == [1]


def test_fast_timeout_does_not_trip_full_budget_breaker(monkeypatch):
    """An immediate transport timeout never quarantines a slow-capable local endpoint."""
    monkeypatch.setattr(aux.time, "time", lambda: 0.0)
    monkeypatch.setattr(aux.time, "monotonic", lambda: 0.0)
    primary = MagicMock(base_url=PRIMARY_URL)
    primary.chat.completions.create.side_effect = TimeoutError("Request timed out.")
    fallback = MagicMock(base_url="https://fallback.example/v1")
    fallback.chat.completions.create.return_value = _response("fallback")

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("custom", "local-model", PRIMARY_URL, "local", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "local-model")),
        patch("agent.auxiliary_client._try_configured_fallback_chain",
              return_value=(fallback, "fallback-model", "fallback_chain[0](custom)")),
    ):
        for _ in range(3):
            result = aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)
            assert result.choices[0].message.content == "fallback"

    assert primary.chat.completions.create.call_count == 3


def test_payment_health_hold_does_not_masquerade_as_timeout_quarantine(monkeypatch):
    """Legacy payment health state does not reroute an explicitly requested primary."""
    monkeypatch.setattr(aux.time, "time", lambda: 0.0)
    primary = MagicMock(base_url=PRIMARY_URL)
    primary.chat.completions.create.return_value = _response("primary")
    fallback = MagicMock(base_url="https://fallback.example/v1")
    fallback.chat.completions.create.return_value = _response("fallback")
    aux._mark_provider_unhealthy(
        "custom", base_url=PRIMARY_URL, ttl=600.0, reason="payment error")

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("custom", "local-model", PRIMARY_URL, "local", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "local-model")),
        patch("agent.auxiliary_client._try_configured_fallback_chain",
              return_value=(fallback, "fallback-model", "fallback_chain[0](custom)")),
    ):
        result = aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)

    assert result.choices[0].message.content == "primary"
    assert primary.chat.completions.create.call_count == 1


def test_non_timeout_half_open_failure_releases_probe_claim(monkeypatch):
    """A half-open request that fails another way never strands the endpoint probe lease."""
    now = [0.0]
    monotonic = [0.0]
    monkeypatch.setattr(aux.time, "time", lambda: now[0])
    monkeypatch.setattr(aux.time, "monotonic", lambda: monotonic[0])
    primary = MagicMock(base_url=PRIMARY_URL)
    primary.chat.completions.create.side_effect = _full_budget_timeout(monotonic)
    fallback = MagicMock(base_url="https://fallback.example/v1")
    fallback.chat.completions.create.return_value = _response("fallback")

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("custom", "local-model", PRIMARY_URL, "local", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "local-model")),
        patch("agent.auxiliary_client._try_configured_fallback_chain",
              return_value=(fallback, "fallback-model", "fallback_chain[0](custom)")),
    ):
        for _ in range(2):
            aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)

        now[0] = 61.0
        primary.chat.completions.create.side_effect = ValueError("malformed response")
        with pytest.raises(ValueError, match="malformed response"):
            aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)

        primary.chat.completions.create.side_effect = None
        primary.chat.completions.create.return_value = _response("primary")
        result = aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0)

    assert result.choices[0].message.content == "primary"
    assert primary.chat.completions.create.call_count == 4


def test_recovery_rung_success_clears_timeout_streak(monkeypatch):
    """A validated same-endpoint parameter-recovery success breaks timeout consecutiveness."""
    monotonic = [0.0]
    monkeypatch.setattr(aux.time, "time", lambda: 0.0)
    monkeypatch.setattr(aux.time, "monotonic", lambda: monotonic[0])
    outcomes = ["timeout", ValueError("Unsupported parameter: temperature"),
                _response("recovered"), "timeout", "timeout"]

    def create(**kwargs):
        outcome = outcomes.pop(0)
        if outcome == "timeout":
            monotonic[0] += float(kwargs["timeout"])
            raise TimeoutError("Request timed out.")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    primary = MagicMock(base_url=PRIMARY_URL)
    primary.chat.completions.create.side_effect = create
    fallback = MagicMock(base_url="https://fallback.example/v1")
    fallback.chat.completions.create.return_value = _response("fallback")

    with (
        patch("agent.auxiliary_client._resolve_task_provider_model",
              return_value=("custom", "local-model", PRIMARY_URL, "local", None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "local-model")),
        patch("agent.auxiliary_client._try_configured_fallback_chain",
              return_value=(fallback, "fallback-model", "fallback_chain[0](custom)")),
    ):
        assert aux.call_llm(
            task="compression", messages=MESSAGES, timeout=1.0, temperature=0.2
        ).choices[0].message.content == "fallback"
        assert aux.call_llm(
            task="compression", messages=MESSAGES, timeout=1.0, temperature=0.2
        ).choices[0].message.content == "recovered"
        for _ in range(2):
            assert aux.call_llm(
                task="compression", messages=MESSAGES, timeout=1.0, temperature=0.2
            ).choices[0].message.content == "fallback"
        aux.call_llm(task="compression", messages=MESSAGES, timeout=1.0, temperature=0.2)

    assert primary.chat.completions.create.call_count == 5
