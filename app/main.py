"""ChefRAG — Streamlit entry point.

Handles:
- i18n loading and language toggle (FR / EN)
- Password-based authentication (via auth.py)
- Chat interface with recipe category selector (via agent.py)
- Structured question rendering: clickable buttons for clarifying questions
- Loading state: inline "ChefRAG réfléchit..." message in the chat flow
- Error handling: error displayed in chat flow with a Retry button
- Responsive layout and WCAG 2.1 AA accessibility
"""
from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
from pathlib import Path
from typing import Any

import streamlit as st

from agent import build_agent, get_agent_response
from auth import is_locked_out, login_form, logout, SESSION_KEY_AUTHENTICATED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: list[str] = ["fr", "en"]
DEFAULT_LANGUAGE: str = os.getenv("APP_LANGUAGE_DEFAULT", "fr")
I18N_DIR: Path = Path(__file__).parent / "i18n"

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Category values passed to the agent
CATEGORY_OPTIONS: dict[str, str] = {
    "all": "all",
    "favorites": "favorites",
    "new": "new",
}

# Maximum visible chat turns before scrolling (display only, not a hard limit)
MAX_VISIBLE_TURNS: int = 50

# ---------------------------------------------------------------------------
# i18n helpers
# ---------------------------------------------------------------------------


def load_translations(lang: str) -> dict[str, str]:
    """Load static UI strings for the given language.

    Falls back to English if the requested language file is not found.

    Args:
        lang: Language code — ``"fr"`` or ``"en"``.

    Returns:
        Dictionary mapping translation keys to localized strings.
    """
    lang = lang if lang in SUPPORTED_LANGUAGES else "en"
    path = I18N_DIR / f"{lang}.json"

    if not path.exists():
        logger.warning("Translation file not found: %s — falling back to English.", path)
        path = I18N_DIR / "en.json"

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def t(key: str, **kwargs: Any) -> str:
    """Return a translated string from the current session translations.

    Supports simple placeholder substitution via ``{key}`` in the string.

    Args:
        key: Translation key (e.g. ``"login_title"``).
        **kwargs: Optional placeholder values (e.g. ``n=3``).

    Returns:
        Localized string, or the raw key if not found.
    """
    raw = st.session_state.get("translations", {}).get(key, key)
    return raw.format(**kwargs) if kwargs else raw


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialise all required Streamlit session state keys.

    Called once at startup. Safe to call multiple times — only sets
    keys that are not already present.
    """
    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE
    if "translations" not in st.session_state:
        st.session_state.translations = load_translations(st.session_state.language)
    if "chat_history" not in st.session_state:
        # List of {"role": "user"|"assistant", "content": str} dicts
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "recipe_category" not in st.session_state:
        st.session_state.recipe_category = "all"
    if "pending_question" not in st.session_state:
        # Holds the current structured question dict while waiting for user input.
        # Shape: {"key": str, "text": str, "options": list[str]} or None
        st.session_state.pending_question = None
    if "cuisine_other_active" not in st.session_state:
        # True when the user selected "Other" for cuisine and we show a text input
        st.session_state.cuisine_other_active = False
    if "needs_agent_call" not in st.session_state:
        st.session_state.needs_agent_call = False
    if "last_error" not in st.session_state:
        # Stores the last agent error message so the Retry button can re-surface it.
        # None when no error is pending.
        st.session_state.last_error = None


# ---------------------------------------------------------------------------
# Language toggle
# ---------------------------------------------------------------------------


def render_language_toggle() -> None:
    """Render the language toggle button in the top-right corner of the header.

    Switches between FR and EN, reloads translations into session state,
    and triggers a rerun so all strings update immediately.
    """
    current = st.session_state.language
    next_lang = "en" if current == "fr" else "fr"

    if st.button(
        t("lang_toggle"),
        key="lang_toggle_btn",
        help=t("aria_lang_toggle"),
        use_container_width=False,
    ):
        st.session_state.language = next_lang
        st.session_state.translations = load_translations(next_lang)
        st.rerun()


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------


def render_login_page() -> None:
    """Render the centered login form with language toggle."""
    col_spacer, col_content, col_spacer2 = st.columns([1, 2, 1])
    with col_content:
        login_form(translations=st.session_state.translations)
        st.divider()
        render_language_toggle()


# ---------------------------------------------------------------------------
# Chat interface helpers
# ---------------------------------------------------------------------------


def get_agent() -> Any:
    """Return the cached agent, building it if not yet initialised.

    The build is wrapped in a chat_message block so the loading indicator
    appears inline in the conversation flow rather than as a floating spinner.

    Returns:
        A fully initialised ChefRagAgent instance.
    """
    if st.session_state.agent is None:
        with st.chat_message("assistant"):
            with st.spinner(t("chat_thinking")):
                st.session_state.agent = build_agent(
                    chroma_host=os.getenv("CHROMA_HOST", "chefrag-chroma"),
                    chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
                    duckdb_path=os.getenv("DUCKDB_PATH", "storage/duckdb/chefrag.duckdb"),
                )
    return st.session_state.agent


def render_chat_message(role: str, content: str) -> None:
    """Render a single chat message with correct ARIA role labelling.

    Args:
        role: ``"user"`` or ``"assistant"``.
        content: Message text (may contain Markdown).
    """
    aria_label = t("aria_user_message") if role == "user" else t("aria_assistant_message")
    with st.chat_message(role):
        st.markdown(
            f'<span class="sr-only">{aria_label}:</span>',
            unsafe_allow_html=True,
        )
        st.markdown(content)


def _inject_user_answer(answer: str) -> None:
    """Record a button-selected answer as a user message and clear pending state.

    Appends the answer to chat_history as a user message, clears the
    pending_question so the agent is called again on next rerun.

    Args:
        answer: The option text the user clicked.
    """
    st.session_state.chat_history.append({"role": "user", "content": answer})
    st.session_state.pending_question = None
    st.session_state.cuisine_other_active = False
    st.session_state.needs_agent_call = True
    st.rerun()


def render_question_widget(question: dict) -> None:
    """Render a clarifying question as clickable option buttons.

    For questions with options: displays one button per option.
    For "Other" cuisine selection: shows a text input field instead.
    For free-text questions (empty options list): shows a text input.

    Args:
        question: Dict with keys "key", "text", "options" (list[str]).
    """
    options: list[str] = question.get("options", [])
    question_text: str = question.get("text", "")
    question_key: str = question.get("key", "")

    with st.chat_message("assistant"):
        st.markdown(
            f'<span class="sr-only">{t("aria_assistant_message")}:</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**{question_text}**")

        # Free-text input: empty options list, or "Other" was previously selected
        if not options or st.session_state.cuisine_other_active:
            _render_free_text_input(question_key)
            return

        # Button grid: one button per option
        cols = st.columns(len(options))
        for i, option in enumerate(options):
            with cols[i]:
                if st.button(
                    option,
                    key=f"q_{question_key}_{i}",
                    use_container_width=True,
                    help=option,
                ):
                    if question_key == "cuisine" and option.lower() == "other":
                        # Switch to free-text mode without injecting "Other" as answer
                        st.session_state.cuisine_other_active = True
                        st.rerun()
                    else:
                        _inject_user_answer(option)


def _render_free_text_input(question_key: str) -> None:
    """Render a free-text input field for open-ended question answers.

    Displays a text input + submit button. On submit, injects the typed
    value as a user message into the chat history.

    Args:
        question_key: The question key, used to namespace the widget key.
    """
    placeholder = t("chat_placeholder")
    value = st.text_input(
        label=t("chat_placeholder"),
        label_visibility="collapsed",
        key=f"free_text_{question_key}",
        placeholder=placeholder,
    )
    if st.button(
        t("chat_send") if "chat_send" in st.session_state.get("translations", {}) else "→",
        key=f"free_text_submit_{question_key}",
    ):
        if value.strip():
            _inject_user_answer(value.strip())
        else:
            st.warning(t("chat_no_input"))


def _render_error_in_chat(error_message: str) -> None:
    """Display an agent error inline in the chat flow with a Retry button.

    The error is rendered inside a ``st.chat_message("assistant")`` block so
    it appears at the correct position in the conversation. A Retry button
    sets ``needs_agent_call = True`` and reruns, allowing the agent to be
    called again without the user retyping their message.

    The last user message is already in ``chat_history`` at this point —
    the retry simply re-sends the existing history.

    Args:
        error_message: Localised error string to display.
    """
    with st.chat_message("assistant"):
        st.markdown(
            f'<span class="sr-only">{t("aria_assistant_message")}:</span>',
            unsafe_allow_html=True,
        )
        st.error(error_message, icon="⚠️")
        if st.button(t("chat_retry"), key="retry_btn"):
            # Clear the stored error and re-trigger the agent call
            st.session_state.last_error = None
            st.session_state.needs_agent_call = True
            st.rerun()


# ---------------------------------------------------------------------------
# Main chat page
# ---------------------------------------------------------------------------


def render_chat_page() -> None:
    """Render the main authenticated chat interface.

    Layout:
    - Top bar: title + category selector + logout
    - Chat history (scrollable)
    - Pending question widget OR sticky input bar

    Accessibility:
    - All interactive elements have ARIA labels
    - Focus is managed via Streamlit's native tab order
    - Color contrast meets WCAG 2.1 AA (4.5:1 for normal text)
    """
    # ---- Accessibility + responsive CSS ----
    st.markdown(
        """
        <style>
        /* Screen-reader-only utility */
        /* display: inline prevents Streamlit from creating an empty block node
           that shifts the visible message text downward. */
        .sr-only {
            display: inline;
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }

        /* ---- Responsive layout ---- */
        .stApp { max-width: 1100px; margin: 0 auto; }

        @media (max-width: 1023px) {
            .stApp { padding: 0 1rem; }
        }

        @media (max-width: 767px) {
            .stApp { padding: 0 0.5rem; }
            section[data-testid="stBottom"] {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: var(--background-color);
                padding: 0.5rem;
                z-index: 100;
                border-top: 1px solid rgba(255,255,255,0.08);
            }
        }

        /* ---- Focus indicator (WCAG 2.1 AA) ---- */
        :focus-visible {
            outline: 3px solid #f5a623;
            outline-offset: 2px;
        }

        /* ---- Chat message contrast ---- */
        div[data-testid="stChatMessage"][data-role="user"] {
            background-color: #1e3a5f;
            color: #f0f4f8;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stChatMessage"][data-role="assistant"] {
            background-color: #12263a;
            color: #e8edf2;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
        }

        /* ---- Question buttons ---- */
        /* Larger touch targets, clear visual distinction */
        div[data-testid="stHorizontalBlock"] .stButton button {
            border: 1px solid rgba(245, 166, 35, 0.5);
            border-radius: 8px;
            font-weight: 500;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }
        div[data-testid="stHorizontalBlock"] .stButton button:hover {
            border-color: #f5a623;
            background-color: rgba(245, 166, 35, 0.12);
        }

        hr { border-color: rgba(255,255,255,0.1); }

        /* ---- Fix: remove Streamlit's default 1rem gap inside chat bubbles ---- */
        /* The gap is set on the flex container wrapping sr-only + message content. */
        div[data-testid="stChatMessage"] div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Top bar ----
    col_title, col_category, col_lang, col_logout = st.columns([3, 2, 1, 1])

    with col_title:
        st.markdown(f"### 🍳 {t('app_title')}")

    with col_category:
        category_keys = list(CATEGORY_OPTIONS.keys())
        category_labels = [t(f"category_{k}") for k in category_keys]
        current_index = category_keys.index(st.session_state.recipe_category)
        selected_label = st.selectbox(
            label=t("category_label"),
            options=category_labels,
            index=current_index,
            key="category_selector",
            label_visibility="collapsed",
            help=t("aria_category_select"),
        )
        selected_key = category_keys[category_labels.index(selected_label)]
        if selected_key != st.session_state.recipe_category:
            st.session_state.recipe_category = selected_key
            st.session_state.agent = None
            st.session_state.pending_question = None
            st.rerun()

    with col_lang:
        render_language_toggle()

    with col_logout:
        if st.button(t("logout"), key="logout_btn", help=t("aria_logout_button")):
            st.session_state.chat_history = []
            st.session_state.agent = None
            st.session_state.pending_question = None
            st.session_state.cuisine_other_active = False
            st.session_state.last_error = None
            logout(st.session_state)
            st.rerun()

    st.divider()

    # ---- Chat history ----
    st.markdown(
        f'<div role="log" aria-label="{t("aria_chat_history")}" aria-live="polite">',
        unsafe_allow_html=True,
    )

    history = st.session_state.chat_history

    if not history:
        render_chat_message("assistant", t("chat_welcome"))

    for message in history[-MAX_VISIBLE_TURNS:]:
        render_chat_message(message["role"], message["content"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- New conversation button ----
    if history:
        if st.button(t("chat_clear"), key="clear_btn", help=t("aria_clear_button")):
            st.session_state.chat_history = []
            st.session_state.pending_question = None
            st.session_state.cuisine_other_active = False
            st.session_state.last_error = None
            st.rerun()

    # ---- Pending error: show error bubble + Retry in chat flow ----
    # Rendered before the pending_question check so a failed agent call
    # after a button answer also surfaces the error correctly.
    if st.session_state.last_error:
        _render_error_in_chat(st.session_state.last_error)
        return

    # ---- After a button answer is injected, call the agent on the next rerun ----
    if st.session_state.needs_agent_call:
        st.session_state.needs_agent_call = False
        _call_agent_and_handle_response()
        return

    # ---- Pending question OR text input ----
    # If the agent returned a clarifying question on the last turn, render
    # the question widget instead of the free-text input bar.
    if st.session_state.pending_question:
        render_question_widget(st.session_state.pending_question)
        # Do NOT render the chat_input while a question is pending —
        # the button widget is the expected interaction point.
        return

    # ---- Free-text input bar ----
    user_input = st.chat_input(
        placeholder=t("chat_placeholder"),
        key="chat_input",
    )

    if user_input:
        user_input = user_input.strip()
        if not user_input:
            st.warning(t("chat_no_input"))
            return

        # Append user message to history before calling the agent.
        # The message is already rendered above via the history loop on the next rerun —
        # we do NOT render it here to avoid a double display.
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        _call_agent_and_handle_response()


def _call_agent_and_handle_response() -> None:
    """Call the agent with the current history and handle its structured response.

    Loading state:
        Displays a "ChefRAG réfléchit..." message inline in the chat flow
        (inside a st.chat_message block) while waiting for the agent.
        The placeholder is replaced by the real response once available.

    Error handling:
        On exception, stores the localised error in ``last_error`` session key
        and resets ``pending_question`` to avoid a blocked UI. The error is
        rendered on the next rerun via ``_render_error_in_chat()``.

    Success — question:
        Stores the question dict in ``pending_question`` and reruns so the
        question is appended to the chat history and rendered as a widget.

    Success — message:
        Appends the assistant reply to history and reruns so the message
        appears in the chat history loop (single render path).
    """
    agent = get_agent()

    # ---- Inline loading indicator ----
    # Renders "ChefRAG réfléchit..." in an assistant chat bubble while the
    # API call is in flight. st.empty() lets us replace it on the same rerun
    # without leaving a ghost bubble behind.
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            f'<span class="sr-only">{t("aria_assistant_message")}:</span>'
            f"_{t('chat_thinking')}_",
            unsafe_allow_html=True,
        )

        try:
            response = get_agent_response(
                agent=agent,
                messages=st.session_state.chat_history,
                category=st.session_state.recipe_category,
                language=st.session_state.language,
            )
        except Exception as exc:
            logger.exception("Agent error: %s", exc)
            # Clear the thinking indicator — the error bubble replaces it
            thinking_placeholder.empty()
            # Store error for rendering on next rerun; reset pending state
            st.session_state.last_error = t("chat_error")
            st.session_state.pending_question = None
            st.rerun()
            return

        # Clear the thinking indicator before rendering the real response
        thinking_placeholder.empty()

    response_type = response.get("type", "message")

    if response_type == "question":
        # Append question text to history so Claude tracks its own flow
        question_text = response.get("text", "")
        if question_text:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": question_text}
            )
        st.session_state.pending_question = response
        st.rerun()

    else:
        # Append assistant message to history; rerun renders it via history loop
        text = response.get("text", "")
        st.session_state.chat_history.append(
            {"role": "assistant", "content": text}
        )
        st.rerun()


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit application entry point.

    Configures the page, initialises session state, and routes to either
    the login page or the authenticated chat interface depending on
    ``st.session_state.authenticated``.
    """
    st.set_page_config(
        page_title="ChefRAG",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "ChefRAG — Personal RAG-based culinary assistant.",
        },
    )

    init_session_state()

    # ---- Skip-to-main-content link (WCAG 2.4.1) ----
    st.markdown(
        """
        <a href="#main-content" class="sr-only focusable">
            Skip to main content
        </a>
        <style>
        .focusable:focus { position: static; width: auto; height: auto;
            clip: auto; white-space: normal; }
        </style>
        <main id="main-content">
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.get(SESSION_KEY_AUTHENTICATED, False):
        render_login_page()
    else:
        render_chat_page()

    # ---- Footer ----
    st.markdown("</main>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <footer style="text-align:center; padding: 2rem 0 1rem;
                       color: rgba(255,255,255,0.35); font-size: 0.75rem;">
            {t("footer")}
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()