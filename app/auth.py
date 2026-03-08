# ==============================================================================
# ChefRAG — Password-based authentication
#
# Single-user authentication via a bcrypt-hashed password stored in the
# APP_PASSWORD environment variable. Rate limiting is enforced via Streamlit
# session state (max 5 failed attempts before lockout).
# ==============================================================================

import os
import time

import bcrypt
import streamlit as st
from loguru import logger

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

MAX_FAILED_ATTEMPTS: int = 5
LOCKOUT_DURATION_SECONDS: int = 300  # 5 minutes

# Session state keys
SESSION_KEY_AUTHENTICATED: str = "authenticated"
SESSION_KEY_FAILED_ATTEMPTS: str = "failed_attempts"
SESSION_KEY_LOCKOUT_UNTIL: str = "lockout_until"


# ------------------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------------------


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plain-text password against a bcrypt hash.

    Args:
        plain_password: The password entered by the user.
        hashed_password: The bcrypt hash stored in the APP_PASSWORD env var.

    Returns:
        True if the password matches the hash, False otherwise.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception as exc:
        # Log the error but never expose details to the caller
        logger.error(f"Password verification error: {exc}")
        return False


def is_locked_out(session_state: dict) -> bool:
    """Check whether the current session is locked out.

    A session is locked out when the number of failed attempts has reached
    MAX_FAILED_ATTEMPTS and the lockout duration has not yet elapsed.

    Args:
        session_state: Streamlit st.session_state (or any dict-like object).

    Returns:
        True if the session is currently locked out, False otherwise.
    """
    lockout_until = session_state.get(SESSION_KEY_LOCKOUT_UNTIL, 0)
    return time.time() < lockout_until


def remaining_lockout_seconds(session_state: dict) -> int:
    """Return the number of seconds remaining in the current lockout.

    Args:
        session_state: Streamlit st.session_state (or any dict-like object).

    Returns:
        Seconds remaining (0 if not locked out).
    """
    lockout_until = session_state.get(SESSION_KEY_LOCKOUT_UNTIL, 0)
    remaining = lockout_until - time.time()
    return max(0, int(remaining))


def record_failed_attempt(session_state: dict) -> int:
    """Increment the failed attempt counter and apply lockout if threshold reached.

    Args:
        session_state: Streamlit st.session_state (or any dict-like object).

    Returns:
        The updated number of failed attempts.
    """
    attempts = session_state.get(SESSION_KEY_FAILED_ATTEMPTS, 0) + 1
    session_state[SESSION_KEY_FAILED_ATTEMPTS] = attempts

    if attempts >= MAX_FAILED_ATTEMPTS:
        session_state[SESSION_KEY_LOCKOUT_UNTIL] = (
            time.time() + LOCKOUT_DURATION_SECONDS
        )
        logger.warning(
            f"Login locked out after {attempts} failed attempts. "
            f"Lockout duration: {LOCKOUT_DURATION_SECONDS}s."
        )

    return attempts


def reset_failed_attempts(session_state: dict) -> None:
    """Reset the failed attempt counter and remove any lockout.

    Called after a successful login.

    Args:
        session_state: Streamlit st.session_state (or any dict-like object).
    """
    session_state[SESSION_KEY_FAILED_ATTEMPTS] = 0
    session_state[SESSION_KEY_LOCKOUT_UNTIL] = 0


def get_hashed_password_from_env() -> str:
    """Load the bcrypt-hashed password from the APP_PASSWORD environment variable.

    Returns:
        The bcrypt hash string.

    Raises:
        RuntimeError: If APP_PASSWORD is not set or is empty.
    """
    hashed = os.environ.get("APP_PASSWORD", "").strip()
    if not hashed:
        raise RuntimeError(
            "APP_PASSWORD environment variable is not set. "
            "Generate a hash with: "
            "python -c \"import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())\""
        )
    return hashed


# ------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------


def login_form(translations: dict) -> bool:
    """Render the Streamlit login form and handle authentication.

    Displays a password input field. On submission, verifies the password
    against the bcrypt hash in APP_PASSWORD. Enforces rate limiting.

    On success, sets st.session_state[SESSION_KEY_AUTHENTICATED] = True
    and triggers a page rerun.

    Args:
        translations: The i18n dict for the current language
                      (loaded from app/i18n/fr.json or en.json).

    Returns:
        True if the user is already authenticated, False otherwise.
    """
    # Initialise session state keys if not present
    if SESSION_KEY_AUTHENTICATED not in st.session_state:
        st.session_state[SESSION_KEY_AUTHENTICATED] = False
    if SESSION_KEY_FAILED_ATTEMPTS not in st.session_state:
        st.session_state[SESSION_KEY_FAILED_ATTEMPTS] = 0
    if SESSION_KEY_LOCKOUT_UNTIL not in st.session_state:
        st.session_state[SESSION_KEY_LOCKOUT_UNTIL] = 0

    # Already authenticated — nothing to render
    if st.session_state[SESSION_KEY_AUTHENTICATED]:
        return True

    # Build the login UI
    t_login = translations.get("login", {})

    st.title(t_login.get("title", "Login"))

    # --- Lockout state ---
    if is_locked_out(st.session_state):
        remaining = remaining_lockout_seconds(st.session_state)
        st.error(t_login.get("error_locked", "Too many attempts. Try again later."))
        st.caption(f"⏳ {remaining}s")
        logger.info(f"Login form displayed lockout message. {remaining}s remaining.")
        return False

    # --- Password form ---
    # aria-label is set via the label parameter (Streamlit renders it as <label>)
    password = st.text_input(
        label=t_login.get("password_label", "Password"),
        placeholder=t_login.get("password_placeholder", "Enter your password"),
        type="password",
        key="login_password_input",
    )

    submit = st.button(
        label=t_login.get("submit_button", "Sign in"),
        type="primary",
        use_container_width=True,
    )

    if submit:
        if not password:
            # Empty submission — count as a failed attempt
            attempts = record_failed_attempt(st.session_state)
            _show_attempt_error(t_login, attempts)
            return False

        try:
            hashed = get_hashed_password_from_env()
        except RuntimeError as exc:
            st.error(str(exc))
            logger.error(str(exc))
            return False

        if verify_password(password, hashed):
            # --- Success ---
            st.session_state[SESSION_KEY_AUTHENTICATED] = True
            reset_failed_attempts(st.session_state)
            logger.info("Successful login.")
            st.rerun()
        else:
            # --- Failure ---
            attempts = record_failed_attempt(st.session_state)
            _show_attempt_error(t_login, attempts)
            logger.warning(f"Failed login attempt #{attempts}.")

    return False


def logout(session_state: dict) -> None:
    """Log the user out by clearing authentication state.

    Args:
        session_state: Streamlit st.session_state (or any dict-like object).
    """
    session_state[SESSION_KEY_AUTHENTICATED] = False
    reset_failed_attempts(session_state)
    logger.info("User logged out.")


# ------------------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------------------


def _show_attempt_error(t_login: dict, attempts: int) -> None:
    """Display the appropriate error message based on attempt count.

    Args:
        t_login: The "login" sub-dict from the translations.
        attempts: The current number of failed attempts.
    """
    if attempts >= MAX_FAILED_ATTEMPTS:
        st.error(t_login.get("error_locked", "Too many attempts."))
    else:
        remaining = MAX_FAILED_ATTEMPTS - attempts
        error_template = t_login.get(
            "error_attempts_remaining", "Attempts remaining: {remaining}"
        )
        st.error(
            t_login.get("error_invalid", "Incorrect password.")
            + " "
            + error_template.format(remaining=remaining)
        )