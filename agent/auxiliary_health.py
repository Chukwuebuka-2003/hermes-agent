"""Endpoint identity for auxiliary custom-provider health checks."""
import contextlib
import contextvars
import threading
from typing import Any, Callable, Optional

from hermes_cli.route_identity import normalize_route_base_url

_AUX_TIMEOUT_TRIP_COUNT = 2
_AUX_TIMEOUT_QUARANTINE_SECONDS = 60.0
_aux_timeout_streaks: dict[Any, int] = {}
_aux_timeout_blocked_until: dict[Any, float] = {}
_aux_timeout_probe_claims: set[Any] = set()
_aux_timeout_streaks_lock = threading.Lock()
_aux_timeout_walk_exclusions: contextvars.ContextVar[frozenset[Any]] = contextvars.ContextVar(
    "aux_timeout_walk_exclusions", default=frozenset()
)
_aux_timeout_call_claims: contextvars.ContextVar[Optional[frozenset[Any]]] = contextvars.ContextVar(
    "aux_timeout_call_claims", default=None
)
_aux_timeout_breaker_bypassed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "aux_timeout_breaker_bypassed", default=False
)

def _unhealthy_cache_key(provider: str, base_url: Optional[str] = None) -> Any:
    """Provider-wide key, or endpoint-specific key for an explicit custom endpoint — prefixed with the
    active profile home: a 402 on profile A's account must not hide the provider from profile B's
    (differently funded) account in the same multiplexed process."""
    from agent.auxiliary_client import _normalize_chain_label
    from hermes_constants import hermes_home_key
    label = _normalize_chain_label(provider)
    endpoint = normalize_route_base_url(_custom_health_base_url(provider, base_url))
    home_key = hermes_home_key()
    if endpoint:
        return home_key, "custom-endpoint", endpoint
    return home_key, label


def _custom_health_base_url(provider: str, explicit_base_url: Optional[str] = None) -> str:
    """Return the concrete custom endpoint used to scope health and failed-route checks."""
    from agent.auxiliary_client import _current_custom_base_url
    explicit = str(explicit_base_url or "").strip()
    from agent.auxiliary_client import _normalize_chain_label
    label = _normalize_chain_label(provider)
    if label == "local/custom":
        return explicit or _current_custom_base_url()
    if label.startswith("custom:") and explicit:
        return explicit
    with contextlib.suppress(ImportError):
        from hermes_cli.runtime_provider import _get_named_custom_provider, _resolves_to_custom
        if _resolves_to_custom(label):
            return explicit or _current_custom_base_url()
        entry = _get_named_custom_provider(provider)
        if entry:
            return explicit or str(entry.get("base_url") or "").strip()
    return ""


def note_auxiliary_timeout(
    provider: str, base_url: Optional[str] = None, *, now: float,
) -> bool:
    """Record a full-budget timeout; True means this endpoint must be quarantined.

    Counts remain saturated after the first trip so a failed half-open probe re-trips
    immediately. A successful response clears the entry through
    :func:`clear_auxiliary_timeout_streak`.
    """
    key = _unhealthy_cache_key(provider, base_url)
    with _aux_timeout_streaks_lock:
        count = min(_AUX_TIMEOUT_TRIP_COUNT, _aux_timeout_streaks.get(key, 0) + 1)
        _aux_timeout_streaks[key] = count
        _aux_timeout_probe_claims.discard(key)
        tripped = count >= _AUX_TIMEOUT_TRIP_COUNT
        if tripped:
            _aux_timeout_blocked_until[key] = now + _AUX_TIMEOUT_QUARANTINE_SECONDS
    claims = _aux_timeout_call_claims.get()
    if claims is not None:
        _aux_timeout_call_claims.set(claims - {key})
    return tripped


def auxiliary_timeout_breaker_blocks(
    provider: str, base_url: Optional[str] = None, *, now: float,
) -> bool:
    """Whether this call must skip the endpoint, atomically admitting one half-open probe."""
    if _aux_timeout_breaker_bypassed.get():
        return False
    key = _unhealthy_cache_key(provider, base_url)
    if key in _aux_timeout_walk_exclusions.get():
        return True
    with _aux_timeout_streaks_lock:
        if _aux_timeout_streaks.get(key, 0) < _AUX_TIMEOUT_TRIP_COUNT:
            return False
        if now < _aux_timeout_blocked_until.get(key, 0.0):
            return True
        claims = _aux_timeout_call_claims.get()
        if key in _aux_timeout_probe_claims:
            return claims is None or key not in claims
        if claims is None:
            return False
        _aux_timeout_probe_claims.add(key)
        _aux_timeout_call_claims.set(claims | {key})
        return False


def begin_auxiliary_timeout_call(*, bypass: bool = False) -> tuple[contextvars.Token, contextvars.Token]:
    """Track half-open claims for one logical call so every exit releases unfinished probes."""
    return (
        _aux_timeout_call_claims.set(frozenset()),
        _aux_timeout_breaker_bypassed.set(bypass),
    )


def end_auxiliary_timeout_call(tokens: tuple[contextvars.Token, contextvars.Token]) -> None:
    """Release probe claims not completed by success or a full-budget timeout."""
    claims = _aux_timeout_call_claims.get() or ()
    with _aux_timeout_streaks_lock:
        _aux_timeout_probe_claims.difference_update(claims)
    _aux_timeout_breaker_bypassed.reset(tokens[1])
    _aux_timeout_call_claims.reset(tokens[0])


def begin_auxiliary_timeout_walk() -> contextvars.Token:
    """Start one fallback walk's local endpoint exclusions."""
    return _aux_timeout_walk_exclusions.set(frozenset())


def exclude_auxiliary_timeout_for_walk(provider: str, base_url: Optional[str] = None) -> None:
    """Skip this endpoint for the rest of the current walk without cross-call quarantine."""
    key = _unhealthy_cache_key(provider, base_url)
    _aux_timeout_walk_exclusions.set(_aux_timeout_walk_exclusions.get() | {key})


def end_auxiliary_timeout_walk(token: contextvars.Token) -> None:
    """Restore the caller's fallback-walk exclusions."""
    _aux_timeout_walk_exclusions.reset(token)


def clear_auxiliary_timeout_streak(provider: str, base_url: Optional[str] = None) -> None:
    """Clear consecutive timeout evidence after this concrete endpoint succeeds."""
    key = _unhealthy_cache_key(provider, base_url)
    with _aux_timeout_streaks_lock:
        _aux_timeout_streaks.pop(key, None)
        _aux_timeout_blocked_until.pop(key, None)
        _aux_timeout_probe_claims.discard(key)
    claims = _aux_timeout_call_claims.get()
    if claims is not None:
        _aux_timeout_call_claims.set(claims - {key})


def reset_auxiliary_timeout_streaks() -> None:
    """Clear process-local timeout evidence (tests and explicit health resets)."""
    with _aux_timeout_streaks_lock:
        _aux_timeout_streaks.clear()
        _aux_timeout_blocked_until.clear()
        _aux_timeout_probe_claims.clear()
    _aux_timeout_walk_exclusions.set(frozenset())


def timed_auxiliary_callback(
    callback: Callable[[dict[str, Any]], Any], timeout: Any, monotonic: Callable[[], float],
) -> Callable[[dict[str, Any]], Any]:
    """Stamp sync provider exceptions with the physical attempt's elapsed time and budget."""
    def timed_sync(request: dict[str, Any]) -> Any:
        started = monotonic()
        try:
            return callback(request)
        except Exception as exc:
            object.__setattr__(exc, "_hermes_aux_elapsed", monotonic() - started)
            object.__setattr__(exc, "_hermes_aux_budget", timeout)
            raise
    return timed_sync


def timed_auxiliary_async_callback(
    callback: Callable[[dict[str, Any]], Any], timeout: Any, monotonic: Callable[[], float],
) -> Callable[[dict[str, Any]], Any]:
    """Stamp async provider exceptions with the physical attempt's elapsed time and budget."""
    async def timed_async(request: dict[str, Any]) -> Any:
        started = monotonic()
        try:
            return await callback(request)
        except Exception as exc:
            object.__setattr__(exc, "_hermes_aux_elapsed", monotonic() - started)
            object.__setattr__(exc, "_hermes_aux_budget", timeout)
            raise
    return timed_async


def is_full_budget_timeout(exc: Exception) -> bool:
    """True only when the failed physical attempt consumed its configured timeout budget."""
    if "no-progress timeout" in str(exc):
        return False
    elapsed = getattr(exc, "_hermes_aux_elapsed", None)
    budget = getattr(exc, "_hermes_aux_budget", None)
    return (
        isinstance(elapsed, (int, float)) and isinstance(budget, (int, float))
        and not isinstance(budget, bool) and budget > 0 and elapsed >= budget * 0.95
    )


def quarantined_primary_fallback(
    task: Optional[str], provider: str, base_url: Optional[str], *, async_mode: bool,
) -> Optional[tuple[Any, Optional[str], str, str]]:
    """Resolve a fallback without dialing an explicitly configured quarantined route."""
    if not provider or provider == "auto":
        return None
    from agent import auxiliary_client as aux
    health_base_url = _custom_health_base_url(provider, base_url)
    if not auxiliary_timeout_breaker_blocks(provider, health_base_url, now=aux.time.time()):
        return None
    aux._log_skip_unhealthy(provider, task, base_url=health_base_url)
    from agent.backend_identity import FailureScope
    fallback = (None, None, "")
    if task:
        fallback = aux._try_configured_fallback_chain(
            task, provider, reason="provider quarantined", failed_base_url=health_base_url,
            failure_scope=FailureScope.ENDPOINT,
        )
    if fallback[0] is None:
        fallback = aux._try_main_agent_model_fallback(
            provider, task, reason="provider quarantined", failed_base_url=health_base_url,
            failure_scope=FailureScope.ENDPOINT,
        )
    client, model, label = fallback
    if client is None:
        from agent.auxiliary_unavailable import AuxiliaryClientUnavailable
        raise AuxiliaryClientUnavailable(
            f"Auxiliary {task or 'call'} provider {provider} is temporarily quarantined after "
            "consecutive request timeouts and no fallback is available"
        )
    if async_mode:
        client, model = aux._to_async_client(client, model or "", is_vision=(task == "vision"))
    concrete_provider = aux._fallback_provider_from_label(label)
    return client, model, concrete_provider, concrete_provider




def fallback_candidate_unavailable_reason(exc: Exception) -> Optional[str]:
    """Why a fallback candidate cannot serve this walk (``_FALLBACK_REASONS`` label), or None.

    The same capacity classes that admitted the primary failure into the chain (payment/quota,
    rate limit, connection, route-incompatible model, malformed response) mean "this lane is out
    for now, try the next configured one"; anything else (a 400 request-shape error, a ValueError)
    is the caller's bug and must still propagate. Auth errors are excluded on purpose: they have
    their own refresh-then-quarantine path in the candidate helpers (#106367)."""
    from agent.auxiliary_client import _FALLBACK_REASONS
    return next(
        (label for predicate, label in _FALLBACK_REASONS if label != "auth error" and predicate(exc)),
        None,
    )


# Quarantine hold per unavailable-reason label. Payment/quota depletion and a dead credential
# last hours, so those keep the long default TTL (None); a per-minute 429, a dropped connection or
# a garbled body clears in seconds — holding the lane for 10 minutes process-wide would hide a
# healthy fallback from every aux task over one transient blip.
_TRANSIENT_CANDIDATE_QUARANTINE_SECONDS = 60.0
_CANDIDATE_QUARANTINE_TTL: dict[str, Optional[float]] = {
    "rate limit": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "connection error": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "request timed out": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "invalid provider response": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
}


def fallback_candidate_quarantine_ttl(reason: Optional[str]) -> Optional[float]:
    """Seconds to hide a fallback candidate for ``reason`` (a ``_FALLBACK_REASONS`` label, or None
    for a stale credential); None means the long default TTL."""
    return _CANDIDATE_QUARANTINE_TTL.get(reason or "")
