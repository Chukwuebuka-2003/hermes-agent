"""#116472: a compression whose request is already over the model window must not let a
trickling summary hold the host toward the generic ceiling. The wait is capped to a dedicated
over-window budget (never more than the configured idle), so the deterministic fallback
summary carries the compaction instead of freezing the UI.
"""

import time

from agent.conversation_compression import (
    _OVER_WINDOW_COMPRESSION_CEILING_SECONDS,
    run_compress_context_with_progress_timeout,
)


def _observed_ceiling(*, idle: float, total_ceiling: float, request_exceeds_window: bool) -> float:
    captured = {}

    def worker(fence):
        # The fence carries the enforced ceiling; capture it while the worker still holds the run.
        captured["ceiling"] = (fence.deadline_monotonic or 0.0) - time.monotonic()
        return [{"role": "user", "content": "hi"}], "sys"

    run_compress_context_with_progress_timeout(
        worker=worker, messages=[{"role": "user", "content": "hi"}], system_prompt_fallback="sys",
        idle_timeout_seconds=idle, total_ceiling_seconds=total_ceiling,
        request_exceeds_window=request_exceeds_window,
    )

    return captured["ceiling"]


def test_over_window_wait_is_capped_to_the_dedicated_budget():
    # idle (300s, the aux-clamped default) is far above the over-window budget → use the budget.
    ceiling = _observed_ceiling(idle=300.0, total_ceiling=600.0, request_exceeds_window=True)

    assert abs(ceiling - _OVER_WINDOW_COMPRESSION_CEILING_SECONDS) < 2.0
    assert ceiling < 300.0


def test_over_window_respects_a_smaller_configured_idle():
    # A configured inactivity budget below the dedicated budget stays authoritative.
    ceiling = _observed_ceiling(idle=15.0, total_ceiling=600.0, request_exceeds_window=True)

    assert abs(ceiling - 15.0) < 2.0


def test_a_fitting_request_keeps_the_full_ceiling():
    ceiling = _observed_ceiling(idle=300.0, total_ceiling=600.0, request_exceeds_window=False)

    assert abs(ceiling - 600.0) < 2.0
