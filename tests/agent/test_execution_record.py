"""Tests for per-conversation execution records (issue #102190).

The sidecar (``execution-record.db``) records one row per ``run_conversation``
execution with lifecycle status plus per-``(model, task)`` usage deltas that
must reconcile with the canonical session accounting for the same calls.
"""
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import execution_record as er
from hermes_state import SessionDB


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    yield h


@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


def _official_totals(db, session_id):
    with db._lock:
        rows = db._conn.execute(
            "SELECT api_call_count, input_tokens, output_tokens, "
            "reasoning_tokens, cache_read_tokens, cache_write_tokens, "
            "estimated_cost_usd FROM session_model_usage WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    totals = {
        "api_call_count": 0, "input_tokens": 0, "output_tokens": 0,
        "reasoning_tokens": 0, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "estimated_cost_usd": 0.0,
    }
    for r in rows:
        d = dict(r)
        for k in totals:
            totals[k] += float(d[k] or 0.0) if k == "estimated_cost_usd" else int(d[k] or 0)
    return totals


def _sidecar_totals(execution_id):
    totals = {
        "api_call_count": 0, "input_tokens": 0, "output_tokens": 0,
        "reasoning_tokens": 0, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "estimated_cost_usd": 0.0,
    }
    for r in er.get_execution_usage(execution_id):
        totals["api_call_count"] += int(r["api_call_count"] or 0)
        totals["input_tokens"] += int(r["input_tokens"] or 0)
        totals["output_tokens"] += int(r["output_tokens"] or 0)
        totals["reasoning_tokens"] += int(r["reasoning_tokens"] or 0)
        totals["cache_read_tokens"] += int(r["cache_read_tokens"] or 0)
        totals["cache_write_tokens"] += int(r["cache_write_tokens"] or 0)
        totals["estimated_cost_usd"] += float(r["estimated_cost_usd"] or 0.0)
    return totals


class TestLifecycle:
    def test_start_creates_running_finish_completes(self, home):
        eid = er.start_execution("s1", correlation_id="turn-1")
        assert eid
        row = er.get_execution(eid)
        assert row["status"] == "running"
        assert row["session_id"] == "s1"
        assert row["correlation_id"] == "turn-1"
        assert row["finished_at"] is None
        assert row["started_at"] > 0

        assert er.finish_execution(eid, "completed") is True
        row = er.get_execution(eid)
        assert row["status"] == "completed"
        assert row["finished_at"] is not None
        assert row["finished_at"] >= row["started_at"]

    def test_terminal_states_are_immutable(self, home):
        eid = er.start_execution("s1")
        assert er.finish_execution(eid, "failed") is True
        assert er.finish_execution(eid, "completed") is False
        assert er.get_execution(eid)["status"] == "failed"

    def test_finish_rejects_bad_status(self, home):
        eid = er.start_execution("s1")
        assert er.finish_execution(eid, "success") is False
        assert er.finish_execution("", "completed") is False
        assert er.finish_execution(None, "completed") is False
        assert er.get_execution(eid)["status"] == "running"

    def test_unfinished_row_stays_running(self, home):
        eid = er.start_execution("s1")
        assert er.get_execution(eid)["status"] == "running"

    def test_start_without_session_is_noop(self, home):
        assert er.start_execution("") is None
        assert er.list_executions() == []

    def test_derive_status(self):
        assert er.derive_status() == "completed"
        assert er.derive_status(failed=True) == "failed"
        assert er.derive_status(interrupted=True) == "cancelled"
        assert er.derive_status(interrupted=True, failed=True) == "cancelled"
        assert er.derive_status(exit_reason="interrupted_by_user") == "cancelled"
        assert er.derive_status(exit_reason="error_near_max_iterations(x)") == "failed"


class TestUsageAttribution:
    def test_main_and_aux_deltas_aggregate_and_conserve(self, home, db):
        db.create_session("s1", source="cli")
        eid = er.start_execution("s1", correlation_id="t1")
        token = er.set_current_execution(eid, "s1")
        try:
            db.queue_token_counts(
                "s1", model="main-model", billing_provider="nous",
                input_tokens=100, output_tokens=10, api_call_count=1,
                estimated_cost_usd=0.01,
            )
            db.record_auxiliary_usage(
                "s1", "vision", model="aux-model",
                billing_provider="openrouter", input_tokens=50, output_tokens=5,
                estimated_cost_usd=0.002,
            )
        finally:
            er.reset_current_execution(token)
        db.flush_token_counts()
        assert er.finish_execution(eid, "completed") is True

        rows = er.get_execution_usage(eid)
        by_key = {(r["model"], r["task"]): r for r in rows}
        assert by_key[("main-model", "")]["input_tokens"] == 100
        assert by_key[("main-model", "")]["output_tokens"] == 10
        assert by_key[("main-model", "")]["api_call_count"] == 1
        assert by_key[("aux-model", "vision")]["input_tokens"] == 50
        assert by_key[("aux-model", "vision")]["api_call_count"] == 1

        assert _sidecar_totals(eid) == _official_totals(db, "s1")

    def test_call_only_delta_is_counted(self, home, db):
        # Codex-style turn with no token usage still counts the call.
        db.create_session("s1", source="cli")
        eid = er.start_execution("s1")
        token = er.set_current_execution(eid, "s1")
        try:
            db.queue_token_counts(
                "s1", model="codex-model", billing_provider="openai-codex",
                billing_mode="subscription_included", api_call_count=1,
            )
        finally:
            er.reset_current_execution(token)
        db.flush_token_counts()
        rows = er.get_execution_usage(eid)
        assert len(rows) == 1
        assert rows[0]["api_call_count"] == 1
        assert rows[0]["input_tokens"] == 0

    def test_absolute_deltas_are_skipped(self, home, db):
        db.create_session("s1", source="cli")
        eid = er.start_execution("s1")
        token = er.set_current_execution(eid, "s1")
        try:
            db.queue_token_counts(
                "s1", input_tokens=100, api_call_count=1, absolute=True,
            )
        finally:
            er.reset_current_execution(token)
        db.flush_token_counts()
        assert er.get_execution_usage(eid) == []

    def test_no_context_records_nothing(self, home, db):
        db.create_session("s1", source="cli")
        assert er.get_current_execution() is None
        db.queue_token_counts("s1", model="m", input_tokens=10, api_call_count=1)
        db.flush_token_counts()
        assert er.list_executions(session_id="s1") == []

    def test_session_mismatch_does_not_cross_attribute(self, home, db):
        db.create_session("s1", source="cli")
        db.create_session("s2", source="cli")
        eid = er.start_execution("s1")
        token = er.set_current_execution(eid, "s1")
        try:
            db.queue_token_counts("s2", model="m", input_tokens=10, api_call_count=1)
            db.record_auxiliary_usage("s2", "vision", model="m", input_tokens=5)
        finally:
            er.reset_current_execution(token)
        db.flush_token_counts()
        # Parent execution untouched; official s2 accounting intact.
        assert er.get_execution_usage(eid) == []
        assert _official_totals(db, "s2")["input_tokens"] == 15


class TestFailOpen:
    def test_broken_sidecar_does_not_break_official(self, tmp_path, monkeypatch, db):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        # A directory where the sqlite file must live breaks every connect.
        (home / "execution-record.db").mkdir()

        db.create_session("s1", source="cli")
        assert er.start_execution("s1") is None

        token = er.set_current_execution("bogus", "s1")
        try:
            db.queue_token_counts("s1", model="m", input_tokens=10, api_call_count=1)
            db.record_auxiliary_usage("s1", "vision", model="m", input_tokens=5)
        finally:
            er.reset_current_execution(token)
        db.flush_token_counts()

        official = _official_totals(db, "s1")
        assert official["input_tokens"] == 15
        assert official["api_call_count"] == 2

    def test_unknown_schema_version_fails_open(self, home):
        eid = er.start_execution("s1")
        assert eid
        conn = sqlite3.connect(str(home / "execution-record.db"))
        try:
            conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
            conn.commit()
        finally:
            conn.close()

        assert er.start_execution("s1") is None
        token = er.set_current_execution(eid, "s1")
        try:
            assert er.note_canonical_delta(
                "s1", model="m", task="", api_call_count=1, input_tokens=1,
            ) is False
            assert er.finish_execution(eid, "completed") is False
        finally:
            er.reset_current_execution(token)
        # Pre-existing row untouched.
        assert er.get_execution(eid)["status"] == "running"


class TestProfileIsolation:
    def test_multiplexed_profiles_write_distinct_sidecars(self, tmp_path, monkeypatch):
        from hermes_constants import get_hermes_home

        home_a = tmp_path / "a"
        home_b = tmp_path / "b"
        home_a.mkdir()
        home_b.mkdir()

        monkeypatch.setenv("HERMES_HOME", str(home_a))
        assert get_hermes_home() == home_a
        ea = er.start_execution("s-shared")
        assert ea

        monkeypatch.setenv("HERMES_HOME", str(home_b))
        assert get_hermes_home() == home_b
        eb = er.start_execution("s-shared")
        assert eb
        assert eb != ea

        assert (home_a / "execution-record.db").exists()
        assert (home_b / "execution-record.db").exists()
        assert [r["execution_id"] for r in er.list_executions()] == [eb]

        monkeypatch.setenv("HERMES_HOME", str(home_a))
        assert [r["execution_id"] for r in er.list_executions()] == [ea]


def _stub_agent(session_id="s-fwd"):
    return SimpleNamespace(
        session_id=session_id,
        _session_db=None,
        platform="cli",
        model="stub-model",
        _parent_session_id="",
        _relay_pending_turn_id=None,
        _conversation_root_id=lambda: "root",
        _reset_activity_labels_after_turn=lambda: None,
    )


def _patch_turn(monkeypatch, inner):
    import agent.conversation_loop as cl
    import agent.relay_runtime as rr
    import hermes_cli.observability.relay_shared_metrics as rsm

    monkeypatch.setattr(cl, "run_conversation", inner)
    monkeypatch.setattr(rsm, "start_task_run", lambda **kw: None)
    monkeypatch.setattr(rsm, "finish_task_run", lambda **kw: None)

    class _Turn:
        relay_enabled = False

    class _Coordinator:
        def acquire_conversation(self, **kw):
            return object()

        def begin_turn(self, lease, **kw):
            return _Turn()

        def finish_logical_calls(self, turn, **kw):
            pass

        def end_turn(self, turn, **kw):
            pass

        def release_conversation(self, lease):
            pass

    monkeypatch.setattr(rr, "SESSION_COORDINATOR", _Coordinator())


class TestRunConversationWiring:
    def test_successful_turn_records_completed(self, monkeypatch, home):
        from run_agent import AIAgent

        _patch_turn(
            monkeypatch,
            lambda agent, *a, **k: {
                "final_response": "hi", "interrupted": False, "failed": False,
            },
        )
        result = AIAgent.run_conversation(_stub_agent(), "hello")
        assert result["final_response"] == "hi"

        rows = er.list_executions(session_id="s-fwd")
        assert len(rows) == 1
        assert rows[0]["status"] == "completed"
        assert rows[0]["finished_at"] >= rows[0]["started_at"]
        assert er.get_current_execution() is None

    def test_interrupted_turn_records_cancelled(self, monkeypatch, home):
        from run_agent import AIAgent

        _patch_turn(
            monkeypatch,
            lambda agent, *a, **k: {
                "final_response": "", "interrupted": True, "failed": False,
            },
        )
        AIAgent.run_conversation(_stub_agent(), "hello")
        rows = er.list_executions(session_id="s-fwd")
        assert len(rows) == 1
        assert rows[0]["status"] == "cancelled"

    def test_raising_turn_records_failed(self, monkeypatch, home):
        from run_agent import AIAgent

        def _boom(agent, *a, **k):
            raise RuntimeError("boom")

        _patch_turn(monkeypatch, _boom)
        with pytest.raises(RuntimeError):
            AIAgent.run_conversation(_stub_agent(), "hello")
        rows = er.list_executions(session_id="s-fwd")
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert er.get_current_execution() is None
