"""ChefRAG — Streamlit entry point.

Handles:
- i18n loading and language toggle (FR / EN)
- Password-based authentication (via auth.py)
- Chat interface with recipe category selector (via agent.py)
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

from agent import build_agent, stream_agent_response
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
    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE
    if "translations" not in st.session_state:
        st.session_state.translations = load_translations(st.session_state.language)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "recipe_category" not in st.session_state:
        st.session_state.recipe_category = "all"


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
    col_spacer, col_content, col_spacer2 = st.columns([1, 2, 1])
    with col_content:
        login_form(translations=st.session_state.translations)
        st.divider()
        render_language_toggle()


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------


def get_agent() -> Any:
    if st.session_state.agent is None:
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
        # Streamlit chat_message renders as <div role="log"> — we add
        # a visually-hidden label for screen readers via a custom span.
        st.markdown(
            f'<span class="sr-only">{aria_label}:</span>',
            unsafe_allow_html=True,
        )
        st.markdown(content)


def render_chat_page() -> None:
    """Render the main authenticated chat interface.

    Layout:
    - Top bar: title + category selector + logout
    - Chat history (scrollable)
    - Sticky input bar at the bottom (mobile-friendly)

    Accessibility:
    - All interactive elements have ARIA labels
    - Focus is managed via Streamlit's native tab order
    - Color contrast meets WCAG 2.1 AA (4.5:1 for normal text)
    """
    # ---- Accessibility: screen-reader-only utility class ----
    st.markdown(
        """
        <style>
        /* Screen-reader-only utility */
        .sr-only {
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

        /* Desktop (≥1024px): standard margins */
        .stApp { max-width: 1100px; margin: 0 auto; }

        /* Tablet (768–1023px): tighter padding */
        @media (max-width: 1023px) {
            .stApp { padding: 0 1rem; }
        }

        /* Mobile (<768px): full-width, fixed bottom input bar */
        @media (max-width: 767px) {
            .stApp { padding: 0 0.5rem; }
            /* Streamlit chat_input is already sticky on mobile via JS —
               we enforce bottom pinning for browsers that don't support it */
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
        /* User bubble: dark background, white text — contrast > 4.5:1 */
        div[data-testid="stChatMessage"][data-role="user"] {
            background-color: #1e3a5f;
            color: #f0f4f8;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
        }
        /* Assistant bubble: slightly lighter, same contrast ratio */
        div[data-testid="stChatMessage"][data-role="assistant"] {
            background-color: #12263a;
            color: #e8edf2;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
        }

        /* ---- Divider accessibility ---- */
        hr { border-color: rgba(255,255,255,0.1); }
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
        category_labels = [
            t(f"category_{k}") for k in category_keys
        ]
        current_index = category_keys.index(st.session_state.recipe_category)
        selected_label = st.selectbox(
            label=t("category_label"),
            options=category_labels,
            index=current_index,
            key="category_selector",
            label_visibility="collapsed",
            # ARIA label injected via help parameter (rendered as title attr)
            help=t("aria_category_select"),
        )
        # Map label back to key
        selected_key = category_keys[category_labels.index(selected_label)]
        if selected_key != st.session_state.recipe_category:
            st.session_state.recipe_category = selected_key
            # Reset agent so it picks up the new category filter
            st.session_state.agent = None
            st.rerun()

    with col_lang:
        render_language_toggle()

    with col_logout:
        if st.button(t("logout"), key="logout_btn", help=t("aria_logout_button")):
            st.session_state.chat_history = []
            st.session_state.agent = None
            logout(st.session_state)  # gère authenticated + rate limit reset
            st.rerun()

    st.divider()

    # ---- Chat history ----
    # ARIA: wrap in a labelled region so screen readers announce it
    st.markdown(
        f'<div role="log" aria-label="{t("aria_chat_history")}" aria-live="polite">',
        unsafe_allow_html=True,
    )

    history = st.session_state.chat_history

    # Inject welcome message if history is empty
    if not history:
        render_chat_message("assistant", t("chat_welcome"))

    for message in history[-MAX_VISIBLE_TURNS:]:
        render_chat_message(message["role"], message["content"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- New conversation button ----
    if history:
        if st.button(
            t("chat_clear"),
            key="clear_btn",
            help=t("aria_clear_button"),
        ):
            st.session_state.chat_history = []
            st.rerun()

    # ---- Input bar ----
    # st.chat_input is rendered at the bottom by Streamlit automatically.
    # It is keyboard-accessible (Tab + Enter) and has a native ARIA role.
    user_input = st.chat_input(
        placeholder=t("chat_placeholder"),
        key="chat_input",
        # max_chars is intentionally unset — no artificial limit on queries
    )

    if user_input:
        user_input = user_input.strip()
        if not user_input:
            st.warning(t("chat_no_input"))
            return

        # Append user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        render_chat_message("user", user_input)

        # Build agent if needed and stream response
        agent = get_agent()

        with st.chat_message("assistant"):
            st.markdown(
                f'<span class="sr-only">{t("aria_assistant_message")}:</span>',
                unsafe_allow_html=True,
            )
            response_placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner(t("chat_thinking")):
                    for chunk in stream_agent_response(
                        agent=agent,
                        user_message=user_input,
                        category=st.session_state.recipe_category,
                        language=st.session_state.language,
                    ):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as exc:
                logger.exception("Agent error: %s", exc)
                error_msg = t("chat_error")
                response_placeholder.error(error_msg, icon="⚠️")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )


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