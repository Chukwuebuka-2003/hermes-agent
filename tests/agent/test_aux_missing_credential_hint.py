"""Missing-credential hints for explicit auxiliary providers (regression for #114405).

Naive ``provider_id.upper() + "_API_KEY"`` yields invalid env var names for
hyphenated provider IDs (``minimax-oauth`` → ``MINIMAX-OAUTH_API_KEY``,
``opencode-zen`` → ``OPENCODE-ZEN_API_KEY``). The hint must come from
``PROVIDER_REGISTRY.api_key_env_vars``; OAuth providers register no env vars
at all, so they get a sign-in hint instead of a fake variable.
"""

import pytest

from agent.auxiliary_client import _resolve_call_client
from agent.auxiliary_unavailable import AuxiliaryClientUnavailable


def _call_explicit(provider: str) -> str:
    """Run the explicit-provider-no-credentials arm and return the error text."""
    from unittest.mock import patch

    with patch("agent.auxiliary_client._get_cached_client", return_value=(None, None)), \
         patch("agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
               return_value=(None, None, None)):
        with pytest.raises(AuxiliaryClientUnavailable) as exc_info:
            _resolve_call_client(
                "compression", provider=provider, model=None, base_url=None,
                api_key=None, resolved_provider=provider, resolved_model=None,
                resolved_base_url=None, resolved_api_key=None,
                resolved_api_mode=None, main_runtime=None, async_mode=False,
            )
    return str(exc_info.value)


def test_hyphenated_api_key_provider_names_real_env_var():
    message = _call_explicit("opencode-zen")
    assert "OPENCODE_ZEN_API_KEY" in message
    assert "OPENCODE-ZEN_API_KEY" not in message


def test_oauth_provider_points_at_signin_not_fake_env_var():
    message = _call_explicit("minimax-oauth")
    assert "MINIMAX-OAUTH_API_KEY" not in message
    assert "hermes auth add minimax-oauth" in message
