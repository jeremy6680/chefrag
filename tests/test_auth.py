# ==============================================================================
# ChefRAG — Tests: app/auth.py
#
# Covers:
#   - verify_password (valid, invalid, empty, malformed hash)
#   - is_locked_out / remaining_lockout_seconds
#   - record_failed_attempt (counter, lockout trigger)
#   - reset_failed_attempts
#   - get_hashed_password_from_env
#   - logout
# ==============================================================================

import time

import bcrypt
import pytest

from auth import (
    LOCKOUT_DURATION_SECONDS,
    MAX_FAILED_ATTEMPTS,
    SESSION_KEY_AUTHENTICATED,
    SESSION_KEY_FAILED_ATTEMPTS,
    SESSION_KEY_LOCKOUT_UNTIL,
    get_hashed_password_from_env,
    is_locked_out,
    logout,
    record_failed_attempt,
    remaining_lockout_seconds,
    reset_failed_attempts,
    verify_password,
)

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

PLAIN_PASSWORD = "correct-horse-battery-staple"


@pytest.fixture()
def valid_hash() -> str:
    """Return a fresh bcrypt hash of PLAIN_PASSWORD."""
    return bcrypt.hashpw(PLAIN_PASSWORD.encode("utf-8"), bcrypt.gensalt()).decode()


@pytest.fixture()
def clean_session() -> dict:
    """Return a clean session state dict (no attempts, not locked)."""
    return {
        SESSION_KEY_AUTHENTICATED: False,
        SESSION_KEY_FAILED_ATTEMPTS: 0,
        SESSION_KEY_LOCKOUT_UNTIL: 0,
    }


@pytest.fixture()
def locked_session(clean_session: dict) -> dict:
    """Return a session that is currently locked out."""
    clean_session[SESSION_KEY_FAILED_ATTEMPTS] = MAX_FAILED_ATTEMPTS
    clean_session[SESSION_KEY_LOCKOUT_UNTIL] = time.time() + LOCKOUT_DURATION_SECONDS
    return clean_session


# ------------------------------------------------------------------------------
# verify_password
# ------------------------------------------------------------------------------


class TestVerifyPassword:
    def test_correct_password_returns_true(self, valid_hash: str) -> None:
        assert verify_password(PLAIN_PASSWORD, valid_hash) is True

    def test_wrong_password_returns_false(self, valid_hash: str) -> None:
        assert verify_password("wrong-password", valid_hash) is False

    def test_empty_password_returns_false(self, valid_hash: str) -> None:
        assert verify_password("", valid_hash) is False

    def test_malformed_hash_returns_false(self) -> None:
        # Should not raise — must return False gracefully
        assert verify_password(PLAIN_PASSWORD, "not-a-valid-hash") is False

    def test_empty_hash_returns_false(self) -> None:
        assert verify_password(PLAIN_PASSWORD, "") is False


# ------------------------------------------------------------------------------
# is_locked_out
# ------------------------------------------------------------------------------


class TestIsLockedOut:
    def test_clean_session_is_not_locked(self, clean_session: dict) -> None:
        assert is_locked_out(clean_session) is False

    def test_active_lockout_returns_true(self, locked_session: dict) -> None:
        assert is_locked_out(locked_session) is True

    def test_expired_lockout_returns_false(self, clean_session: dict) -> None:
        # Set lockout in the past
        clean_session[SESSION_KEY_LOCKOUT_UNTIL] = time.time() - 1
        assert is_locked_out(clean_session) is False

    def test_missing_key_returns_false(self) -> None:
        # Empty dict — no lockout key
        assert is_locked_out({}) is False


# ------------------------------------------------------------------------------
# remaining_lockout_seconds
# ------------------------------------------------------------------------------


class TestRemainingLockoutSeconds:
    def test_no_lockout_returns_zero(self, clean_session: dict) -> None:
        assert remaining_lockout_seconds(clean_session) == 0

    def test_active_lockout_returns_positive(self, locked_session: dict) -> None:
        remaining = remaining_lockout_seconds(locked_session)
        assert 0 < remaining <= LOCKOUT_DURATION_SECONDS

    def test_expired_lockout_returns_zero(self, clean_session: dict) -> None:
        clean_session[SESSION_KEY_LOCKOUT_UNTIL] = time.time() - 10
        assert remaining_lockout_seconds(clean_session) == 0


# ------------------------------------------------------------------------------
# record_failed_attempt
# ------------------------------------------------------------------------------


class TestRecordFailedAttempt:
    def test_increments_counter(self, clean_session: dict) -> None:
        result = record_failed_attempt(clean_session)
        assert result == 1
        assert clean_session[SESSION_KEY_FAILED_ATTEMPTS] == 1

    def test_multiple_increments(self, clean_session: dict) -> None:
        for i in range(1, 4):
            result = record_failed_attempt(clean_session)
            assert result == i

    def test_triggers_lockout_at_max_attempts(self, clean_session: dict) -> None:
        for _ in range(MAX_FAILED_ATTEMPTS):
            record_failed_attempt(clean_session)

        assert is_locked_out(clean_session) is True
        assert clean_session[SESSION_KEY_LOCKOUT_UNTIL] > time.time()

    def test_no_lockout_before_max_attempts(self, clean_session: dict) -> None:
        for _ in range(MAX_FAILED_ATTEMPTS - 1):
            record_failed_attempt(clean_session)

        assert is_locked_out(clean_session) is False


# ------------------------------------------------------------------------------
# reset_failed_attempts
# ------------------------------------------------------------------------------


class TestResetFailedAttempts:
    def test_resets_counter_to_zero(self, clean_session: dict) -> None:
        clean_session[SESSION_KEY_FAILED_ATTEMPTS] = 3
        reset_failed_attempts(clean_session)
        assert clean_session[SESSION_KEY_FAILED_ATTEMPTS] == 0

    def test_clears_lockout(self, locked_session: dict) -> None:
        reset_failed_attempts(locked_session)
        assert is_locked_out(locked_session) is False
        assert locked_session[SESSION_KEY_LOCKOUT_UNTIL] == 0


# ------------------------------------------------------------------------------
# get_hashed_password_from_env
# ------------------------------------------------------------------------------


class TestGetHashedPasswordFromEnv:
    def test_returns_hash_when_env_is_set(
        self, monkeypatch: pytest.MonkeyPatch, valid_hash: str
    ) -> None:
        monkeypatch.setenv("APP_PASSWORD", valid_hash)
        assert get_hashed_password_from_env() == valid_hash

    def test_raises_when_env_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("APP_PASSWORD", raising=False)
        with pytest.raises(RuntimeError, match="APP_PASSWORD"):
            get_hashed_password_from_env()

    def test_raises_when_env_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APP_PASSWORD", "   ")
        with pytest.raises(RuntimeError, match="APP_PASSWORD"):
            get_hashed_password_from_env()

    def test_strips_whitespace(
        self, monkeypatch: pytest.MonkeyPatch, valid_hash: str
    ) -> None:
        monkeypatch.setenv("APP_PASSWORD", f"  {valid_hash}  ")
        assert get_hashed_password_from_env() == valid_hash


# ------------------------------------------------------------------------------
# logout
# ------------------------------------------------------------------------------


class TestLogout:
    def test_clears_authenticated_flag(self, clean_session: dict) -> None:
        clean_session[SESSION_KEY_AUTHENTICATED] = True
        logout(clean_session)
        assert clean_session[SESSION_KEY_AUTHENTICATED] is False

    def test_resets_failed_attempts(self, clean_session: dict) -> None:
        clean_session[SESSION_KEY_FAILED_ATTEMPTS] = 3
        logout(clean_session)
        assert clean_session[SESSION_KEY_FAILED_ATTEMPTS] == 0

    def test_clears_lockout_on_logout(self, locked_session: dict) -> None:
        locked_session[SESSION_KEY_AUTHENTICATED] = True
        logout(locked_session)
        assert is_locked_out(locked_session) is False