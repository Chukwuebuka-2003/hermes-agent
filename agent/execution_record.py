"""Per-conversation execution records with usage attribution (issue #102190).

Hermes has session-level usage (``sessions`` + ``session_model_usage``) and
cron execution records, but operators need a durable, queryable record for
each ``run_conversation`` execution: lifecycle status, elapsed time, and
usage grouped by actual ``(model, task)`` calls. Session totals alone cannot
attribute work to a user-visible turn once auxiliary models, delegation,
retries, and fallbacks are involved.

This module implements the versioned, per-profile sidecar SQLite database
(``execution-record.db``) proposed in the issue. Design constraints:

* Never alter official ``state.db`` tables, indexes, or transactions. All
  writes here are best-effort and isolated in their own connections.
* Sidecar failures, lock contention, or unknown schema versions fail open
  and never block agent execution. Every public entry point swallows its
  own exceptions (debug-logged) and returns ``None``/``False``.
* The profile home is resolved dynamically via ``get_hermes_home()`` on
  every connection so multiplexed profiles never cross-write.
* Rows left ``running`` when a process dies stay ``running``; finalization
  never fabricates success. Terminal states are immutable.
* Conservation: deltas mirrored here are the same canonical deltas fed to
  ``SessionDB`` (``queue_token_counts`` / ``record_auxiliary_usage``), so
  aggregated execution usage reconciles with session usage for the same
  calls. Absolute (cumulative-overwrite) deltas are skipped because they
  cannot be split back into per-execution attributions.

Usage is grouped by ``(model, task)`` per the issue's minimal schema;
``''`` (empty) task is the main agent loop, auxiliary calls record their
task name (``vision``, ``compression``, ...).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_DB_FILENAME = "execution-record.db"

_TERMINAL_STATUSES = ("completed", "failed", "cancelled")
_ALL_STATUSES = ("running",) + _TERMINAL_STATUSES

_DB_LOCK = threading.Lock()

# (execution_id, session_id) for the active run_conversation turn, or None.
_current: ContextVar[Optional[tuple]] = ContextVar(
    "execution_record_context", default=None
)


def _db_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _DB_FILENAME


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        try:
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label=_DB_FILENAME)
        except Exception:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except Exception:
                pass
        conn.execute("PRAGMA busy_timeout=5000")
        _ensure_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS execution(
             execution_id TEXT PRIMARY KEY,
             correlation_id TEXT,
             session_id TEXT NOT NULL,
             started_at REAL NOT NULL,
             finished_at REAL,
             status TEXT NOT NULL CHECK(status IN ('running','completed','failed','cancelled'))
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS execution_usage(
             execution_id TEXT NOT NULL,
             model TEXT NOT NULL,
             task TEXT NOT NULL DEFAULT '',
             api_call_count INTEGER NOT NULL DEFAULT 0,
             input_tokens INTEGER NOT NULL DEFAULT 0,
             output_tokens INTEGER NOT NULL DEFAULT 0,
             reasoning_tokens INTEGER NOT NULL DEFAULT 0,
             cache_read_tokens INTEGER NOT NULL DEFAULT 0,
             cache_write_tokens INTEGER NOT NULL DEFAULT 0,
             estimated_cost_usd REAL NOT NULL DEFAULT 0,
             actual_cost_usd REAL NOT NULL DEFAULT 0,
             PRIMARY KEY (execution_id, model, task)
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_execution_session "
        "ON execution(session_id, started_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_execution_usage_execution "
        "ON execution_usage(execution_id)"
    )
    row = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
    conn.commit()


def _schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            return _SCHEMA_VERSION
        return int(row["value"] if isinstance(row, sqlite3.Row) else row[0])
    except Exception:
        return _SCHEMA_VERSION


def _schema_supported(conn: sqlite3.Connection) -> bool:
    return _schema_version(conn) == _SCHEMA_VERSION


def set_current_execution(execution_id: str, session_id: str) -> Optional[Token]:
    """Publish the active execution for ambient delta attribution."""
    if not execution_id or not session_id:
        return None
    try:
        return _current.set((str(execution_id), str(session_id)))
    except Exception:
        return None


def get_current_execution() -> Optional[tuple]:
    """Return ``(execution_id, session_id)`` for the active turn, or ``None``."""
    try:
        return _current.get()
    except Exception:
        return None


def reset_current_execution(token: Optional[Token]) -> None:
    """Restore the previous execution context (pair with ``set_...``)."""
    if token is None:
        return
    try:
        _current.reset(token)
    except Exception:
        try:
            _current.set(None)
        except Exception:
            pass


def derive_status(
    *,
    interrupted: bool = False,
    failed: bool = False,
    exit_reason: Optional[str] = None,
) -> str:
    """Map turn outcome flags to a terminal execution status."""
    if interrupted:
        return "cancelled"
    if failed:
        return "failed"
    reason = str(exit_reason or "").lower()
    if "interrupt" in reason or "cancel" in reason:
        return "cancelled"
    if "fail" in reason or "error" in reason or "exception" in reason:
        return "failed"
    return "completed"


def start_execution(
    session_id: str,
    correlation_id: Optional[str] = None,
) -> Optional[str]:
    """Create a ``running`` execution row. Fail-open: returns ``None``."""
    if not session_id:
        return None
    execution_id = uuid.uuid4().hex
    try:
        with _DB_LOCK:
            conn = _connect()
            try:
                if not _schema_supported(conn):
                    logger.debug(
                        "execution_record: unsupported schema version; skipping start"
                    )
                    return None
                now = time.time()
                with conn:
                    conn.execute(
                        """INSERT INTO execution(
                             execution_id, correlation_id, session_id,
                             started_at, finished_at, status
                           ) VALUES (?, ?, ?, ?, NULL, 'running')""",
                        (
                            execution_id,
                            str(correlation_id or ""),
                            str(session_id),
                            now,
                        ),
                    )
            finally:
                conn.close()
        return execution_id
    except Exception:
        logger.debug("execution_record: start failed (non-fatal)", exc_info=True)
        return None


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def note_canonical_delta(
    session_id: str,
    *,
    model: Optional[str] = None,
    task: Optional[str] = None,
    api_call_count: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    estimated_cost_usd: Optional[float] = None,
    actual_cost_usd: Optional[float] = None,
) -> bool:
    """Mirror one canonical usage delta into the active execution. Fail-open.

    Only records when an execution context is published for the same
    ``session_id``; delegation children publish their own execution so each
    ``run_conversation`` attributes its own calls by actual ``(model, task)``.
    """
    try:
        if not session_id:
            return False
        ctx = get_current_execution()
        if ctx is None:
            return False
        execution_id, exec_session_id = ctx
        if not execution_id or str(exec_session_id) != str(session_id):
            return False
        api_call_count = _coerce_int(api_call_count)
        input_tokens = _coerce_int(input_tokens)
        output_tokens = _coerce_int(output_tokens)
        reasoning_tokens = _coerce_int(reasoning_tokens)
        cache_read_tokens = _coerce_int(cache_read_tokens)
        cache_write_tokens = _coerce_int(cache_write_tokens)
        estimated = _coerce_float(estimated_cost_usd)
        actual = _coerce_float(actual_cost_usd)
        if not (
            api_call_count
            or input_tokens
            or output_tokens
            or reasoning_tokens
            or cache_read_tokens
            or cache_write_tokens
            or estimated
            or actual
        ):
            return False
        model_name = str(model or "") or "unknown"
        task_name = str(task or "")
        with _DB_LOCK:
            conn = _connect()
            try:
                if not _schema_supported(conn):
                    return False
                with conn:
                    conn.execute(
                        """INSERT INTO execution_usage(
                             execution_id, model, task, api_call_count,
                             input_tokens, output_tokens, reasoning_tokens,
                             cache_read_tokens, cache_write_tokens,
                             estimated_cost_usd, actual_cost_usd
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(execution_id, model, task) DO UPDATE SET
                             api_call_count = api_call_count + excluded.api_call_count,
                             input_tokens = input_tokens + excluded.input_tokens,
                             output_tokens = output_tokens + excluded.output_tokens,
                             reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                             cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                             cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                             estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd,
                             actual_cost_usd = actual_cost_usd + excluded.actual_cost_usd""",
                        (
                            execution_id,
                            model_name,
                            task_name,
                            api_call_count,
                            input_tokens,
                            output_tokens,
                            reasoning_tokens,
                            cache_read_tokens,
                            cache_write_tokens,
                            estimated,
                            actual,
                        ),
                    )
            finally:
                conn.close()
        return True
    except Exception:
        logger.debug(
            "execution_record: delta recording failed (non-fatal)", exc_info=True
        )
        return False


def note_queued_delta(session_id: str, kwargs: Dict[str, Any]) -> bool:
    """Mirror a ``queue_token_counts`` delta. Skips absolute overwrites."""
    try:
        if not isinstance(kwargs, dict):
            return False
        if kwargs.get("absolute"):
            return False
        return note_canonical_delta(
            session_id,
            model=kwargs.get("model"),
            task=kwargs.get("task", ""),
            api_call_count=kwargs.get("api_call_count", 0),
            input_tokens=kwargs.get("input_tokens", 0),
            output_tokens=kwargs.get("output_tokens", 0),
            reasoning_tokens=kwargs.get("reasoning_tokens", 0),
            cache_read_tokens=kwargs.get("cache_read_tokens", 0),
            cache_write_tokens=kwargs.get("cache_write_tokens", 0),
            estimated_cost_usd=kwargs.get("estimated_cost_usd"),
            actual_cost_usd=kwargs.get("actual_cost_usd"),
        )
    except Exception:
        logger.debug(
            "execution_record: queued delta failed (non-fatal)", exc_info=True
        )
        return False


def finish_execution(execution_id: Optional[str], status: str) -> bool:
    """Write a terminal status once. Terminal states are immutable. Fail-open."""
    if not execution_id or status not in _TERMINAL_STATUSES:
        return False
    try:
        with _DB_LOCK:
            conn = _connect()
            try:
                if not _schema_supported(conn):
                    return False
                with conn:
                    cur = conn.execute(
                        """UPDATE execution
                           SET status=?, finished_at=?
                           WHERE execution_id=? AND status='running'""",
                        (status, time.time(), str(execution_id)),
                    )
                    return cur.rowcount == 1
            finally:
                conn.close()
    except Exception:
        logger.debug(
            "execution_record: finish failed (non-fatal)", exc_info=True
        )
        return False


def get_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Return one execution row, or ``None``. Fail-open."""
    try:
        with _DB_LOCK:
            conn = _connect()
            try:
                row = conn.execute(
                    "SELECT * FROM execution WHERE execution_id=?",
                    (str(execution_id),),
                ).fetchone()
                return dict(row) if row is not None else None
            finally:
                conn.close()
    except Exception:
        logger.debug("execution_record: read failed (non-fatal)", exc_info=True)
        return None


def get_execution_usage(execution_id: str) -> List[Dict[str, Any]]:
    """Return per-``(model, task)`` usage rows. Fail-open (empty on fault)."""
    try:
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM execution_usage WHERE execution_id=? "
                    "ORDER BY model, task",
                    (str(execution_id),),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
    except Exception:
        logger.debug("execution_record: usage read failed (non-fatal)", exc_info=True)
        return []


def list_executions(
    *, session_id: Optional[str] = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """Return newest-first executions, optionally filtered. Fail-open."""
    try:
        limit = max(1, min(int(limit), 500))
        with _DB_LOCK:
            conn = _connect()
            try:
                if session_id:
                    rows = conn.execute(
                        "SELECT * FROM execution WHERE session_id=? "
                        "ORDER BY started_at DESC, execution_id DESC LIMIT ?",
                        (str(session_id), limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM execution "
                        "ORDER BY started_at DESC, execution_id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
    except Exception:
        logger.debug("execution_record: list failed (non-fatal)", exc_info=True)
        return []
