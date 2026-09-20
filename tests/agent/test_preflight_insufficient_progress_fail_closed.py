"""Regression coverage for #116472: an over-window session whose preflight compression
cannot make progress must fail closed immediately instead of re-waiting to the compression
ceiling (which blocks the Desktop UI and gets the renderer SIGKILLed)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import (
    PreflightCompressionTimedOut,
    _fail_closed_on_insufficient_progress,
)
from agent.turn_context_compaction import CompactionOutcome, _run_preflight_passes


def _agent(*, context_length: int | None):
    compressor = SimpleNamespace(
        protect_first_n=3, protect_last_n=3, threshold_tokens=65_536,
        context_length=context_length, summary_target_ratio=0.5,
    )
    return SimpleNamespace(
        context_compressor=compressor, session_id="s1", model="m",
        max_compression_attempts=3, _emit_status=MagicMock(),
    )


def test_over_window_insufficient_progress_fails_closed_with_new_session_hint():
    agent = _agent(context_length=131_072)

    with pytest.raises(PreflightCompressionTimedOut, match="Start a new session"):
        _fail_closed_on_insufficient_progress(agent, 356_113)


def test_insufficient_progress_below_or_unknown_window_proceeds():
    # Over threshold but the request fits the window: send it (the cooldown-blocked behaviour).
    _fail_closed_on_insufficient_progress(_agent(context_length=1_000_000), 356_113)
    # Window unknown: keep the conservative existing behaviour, never guess.
    _fail_closed_on_insufficient_progress(_agent(context_length=None), 356_113)


def test_preflight_pass_fails_closed_when_compression_cannot_shrink_over_window_session():
    """The wiring: a no-op preflight pass on a request above the window must raise, not
    silently continue toward the ceiling."""
    agent = _agent(context_length=131_072)
    messages = [{"role": "user", "content": "hi"}]
    agent._compress_context = lambda msgs, system, **kw: (msgs, system)  # no progress
    out = CompactionOutcome(
        messages=messages, active_system_prompt="sys", conversation_history=None,
        current_turn_user_idx=0,
    )

    with patch("agent.turn_context._preflight_request_tokens", return_value=356_113), patch(
        "agent.turn_context_compaction.automatic_compaction_status_message", return_value=""
    ):
        with pytest.raises(PreflightCompressionTimedOut, match="Start a new session"):
            _run_preflight_passes(
                agent, out, agent.context_compressor, 356_113, "sys", "t",
            )
