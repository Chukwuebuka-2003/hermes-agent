"""Short-lived, single-use handles for model-blind verification codes.

Trusted retrieval paths park a raw SMS/email code in a profile-scoped encrypted
sidecar and return only an opaque ``otp_…`` handle. The sidecar is protected by
a cross-process lock, so a CLI or skill subprocess can mint a handle that the
agent process later consumes exactly once. Raw bytes never enter model context,
tool results, logs, or session storage.
"""

from __future__ import annotations

import re
import secrets
import time
from pathlib import Path
from typing import Any, Optional

#: Default lifetime: one SMS/email delivery window.
CODE_TTL_S = 300
MIN_TTL_S = 30
MAX_TTL_S = 900
#: Per-profile bound; oldest entries are evicted first.
MAX_PER_PROFILE = 16
HANDLE_PREFIX = "otp_"

_CODE_KEYWORD = re.compile(r"\b(code|otp|pin|verification|verify)\b", re.I)
_DIGIT_RUN = re.compile(r"\b(\d{4,8})\b")
#: Alphanumeric OTPs (e.g. GitHub's 8-char codes): mixed case/digits, 6-10 chars.
_ALNUM_RUN = re.compile(
    r"\b(?=[A-Za-z0-9]{6,10}\b)(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9]{6,10}\b"
)


def extract_verification_code(text: str) -> Optional[str]:
    """Pull a likely OTP out of an SMS/email body. Never returns the body."""
    if not isinstance(text, str) or not text.strip():
        return None
    for line in text.splitlines():
        if _CODE_KEYWORD.search(line):
            match = _DIGIT_RUN.search(line) or _ALNUM_RUN.search(line)
            if match:
                return match.group(0)
    match = _DIGIT_RUN.search(text)
    if match:
        return match.group(1)
    if _CODE_KEYWORD.search(text):
        match = _ALNUM_RUN.search(text)
        if match:
            return match.group(0)
    return None


def _scope() -> str:
    from hermes_constants import get_hermes_home

    return str(get_hermes_home())


def _store():
    from agent.vault_store import EncryptedDocumentStore

    return EncryptedDocumentStore(
        Path(_scope()) / "cache" / "code-handles",
        filename="handles.json.enc",
        key_filename="handles.key",
        lock_filename=".handles.lock",
        label="verification-code handle store",
    )


def _clamp_ttl(ttl_s: float) -> float:
    try:
        ttl = float(ttl_s)
    except (TypeError, ValueError):
        return float(CODE_TTL_S)
    return min(max(ttl, float(MIN_TTL_S)), float(MAX_TTL_S))


def _entries(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = document.get("entries")
    if not isinstance(entries, dict):
        entries = {}
        document["entries"] = entries
    document["version"] = 1
    return entries


def _prune_expired(entries: dict[str, dict[str, Any]], now: float) -> None:
    for handle, entry in list(entries.items()):
        if not isinstance(entry, dict) or now > float(entry.get("expires_at") or 0):
            entries.pop(handle, None)


def mint(
    code: str,
    *,
    origin: Optional[str] = None,
    source: str = "",
    ttl_s: float = CODE_TTL_S,
) -> dict[str, Any]:
    """Encrypt ``code`` behind an opaque handle visible across local processes."""
    from agent.redact import register_vault_redaction_value

    if not isinstance(code, str) or not code.strip():
        raise ValueError("code is required")
    raw = code.strip()

    normalized_origin: Optional[str] = None
    if origin:
        from agent.vault_store import normalize_origin

        normalized_origin = normalize_origin(origin)

    register_vault_redaction_value(raw)
    handle = HANDLE_PREFIX + secrets.token_urlsafe(16)
    expires_at = time.time() + _clamp_ttl(ttl_s)
    entry = {
        "code": raw,
        "origin": normalized_origin,
        "source": (source or "")[:32],
        "expires_at": expires_at,
    }
    with _store().locked_document() as document:
        entries = _entries(document)
        _prune_expired(entries, time.time())
        entries[handle] = entry
        while len(entries) > MAX_PER_PROFILE:
            entries.pop(next(iter(entries)))
    return {
        "code_handle": handle,
        "expires_at": expires_at,
        "source": entry["source"],
        "origin": normalized_origin,
    }


def consume_entry(handle: str, *, origin: Optional[str] = None) -> Optional[dict[str, str]]:
    """Atomically burn and resolve one handle, returning its code and source.

    Expired, unknown and origin-mismatched handles are indistinguishable. A
    matching row is removed and durably committed before its code is returned,
    so concurrent processes cannot both win.
    """
    if not isinstance(handle, str) or not handle:
        return None
    from agent.vault_store import VaultError

    requested_origin: Optional[str] = None
    if origin:
        from agent.vault_store import normalize_origin

        try:
            requested_origin = normalize_origin(origin)
        except Exception:
            requested_origin = None

    try:
        entry: Optional[dict[str, Any]] = None
        with _store().locked_document() as document:
            entries = _entries(document)
            _prune_expired(entries, time.time())
            candidate = entries.pop(handle, None)
            if isinstance(candidate, dict):
                entry = candidate
    except (OSError, ValueError, VaultError):
        return None

    if entry is None:
        return None
    bound = entry.get("origin")
    if bound and requested_origin != bound:
        return None
    code = entry.get("code")
    if not isinstance(code, str) or not code:
        return None

    # Mint-time redaction happened in a different process. Register again in
    # the consumer before these bytes reach any browser or model-egress path.
    from agent.redact import register_vault_redaction_value

    register_vault_redaction_value(code)
    return {"code": code, "source": str(entry.get("source") or "")}


def consume(handle: str, *, origin: Optional[str] = None) -> Optional[str]:
    """Return a handle's raw code once, or ``None``."""
    entry = consume_entry(handle, origin=origin)
    return entry["code"] if entry else None


def peek_source(handle: str) -> Optional[str]:
    """Mint-time source label for a still-live handle (diagnostics only)."""
    if not isinstance(handle, str) or not handle:
        return None
    from agent.vault_store import VaultError

    try:
        document = _store().read_document()
    except (OSError, ValueError, VaultError):
        return None
    entry = _entries(document).get(handle)
    if not isinstance(entry, dict) or time.time() > float(entry.get("expires_at") or 0):
        return None
    return str(entry.get("source") or "") or None


def clear_codes() -> None:
    """Drop the current profile's parked codes (profile teardown / tests)."""
    from agent.vault_store import VaultError

    try:
        with _store().locked_document() as document:
            document.clear()
            document.update({"version": 1, "entries": {}})
    except (OSError, ValueError, VaultError):
        return
