"""Admin tab for ChefRAG — recipe upload and indexing interface.

This module renders the administration tab in the Streamlit app.
It allows the user to upload Umami JSON exports and trigger indexing
into ChromaDB and DuckDB without any external orchestration tool.
"""

import os
import logging
from pathlib import Path
from typing import Callable

import streamlit as st

from indexer import run_indexing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported recipe categories and their target data directories.
CATEGORIES: dict[str, str] = {
    "favorites": "data/favorites",
    "new": "data/new",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def save_uploaded_file(file: st.runtime.uploaded_file_manager.UploadedFile, category: str) -> Path:
    """Save a Streamlit-uploaded file to the appropriate data directory.

    Args:
        file: The uploaded file object from st.file_uploader.
        category: Recipe category slug — "favorites" or "new".

    Returns:
        The absolute Path where the file was saved.

    Raises:
        ValueError: If the category is not recognized.
        OSError: If the file cannot be written to disk.
    """
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category!r}. Expected one of {list(CATEGORIES)}")

    target_dir = Path(CATEGORIES[category])
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / file.name
    target_path.write_bytes(file.getvalue())

    logger.info("Saved uploaded file: %s → %s", file.name, target_path)
    return target_path


# ---------------------------------------------------------------------------
# Admin tab renderer
# ---------------------------------------------------------------------------


def render_admin_tab(translations: dict[str, str], t: Callable[..., str]) -> None:
    """Render the Admin tab UI inside the authenticated Streamlit app.

    Displays two file uploaders (Favorites / New) and an indexing button.
    When the button is clicked, uploaded files are saved to disk and
    run_indexing() is called for each category that received new files.

    Args:
        translations: The active i18n dictionary (unused directly — t() is used instead).
        t: The translation helper function from main.py.
    """
    # --- Upload section: Favorites ---
    st.subheader(t("admin_section_favorites"))
    favorites_files = st.file_uploader(
        label=t("admin_upload_favorites_label"),
        type=["json"],
        accept_multiple_files=True,
        help=t("admin_upload_help"),
        key="upload_favorites",
    )

    if favorites_files:
        st.caption(t("admin_files_ready", count=len(favorites_files)))
    else:
        st.caption(t("admin_no_files"))

    st.divider()

    # --- Upload section: New ---
    st.subheader(t("admin_section_new"))
    new_files = st.file_uploader(
        label=t("admin_upload_new_label"),
        type=["json"],
        accept_multiple_files=True,
        help=t("admin_upload_help"),
        key="upload_new",
    )

    if new_files:
        st.caption(t("admin_files_ready", count=len(new_files)))
    else:
        st.caption(t("admin_no_files"))

    st.divider()

    # --- Indexing button ---
    has_files = bool(favorites_files or new_files)

    if not has_files:
        st.info(t("admin_index_no_files"))

    index_button = st.button(
        label=t("admin_index_button"),
        disabled=not has_files,
        use_container_width=True,
        type="primary",
    )

    if not index_button:
        return

    # --- Run indexing pipeline ---
    uploads_by_category: dict[str, list] = {
        "favorites": list(favorites_files or []),
        "new": list(new_files or []),
    }

    chroma_host = os.environ.get("CHROMA_HOST", "localhost")
    chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
    duckdb_path = os.environ.get("DUCKDB_PATH", "storage/duckdb/chefrag.duckdb")

    with st.status(t("admin_index_running"), expanded=True) as status:
        try:
            total_indexed = 0

            for category, files in uploads_by_category.items():
                if not files:
                    continue

                # Save each uploaded file to disk first
                for uploaded_file in files:
                    st.write(t("admin_index_saving", filename=uploaded_file.name))
                    save_uploaded_file(uploaded_file, category)
                    st.write(t("admin_index_saved", filename=uploaded_file.name))

                # Run the full indexing pipeline for this category
                st.write(t("admin_index_category_start", category=category))
                run_indexing(
                    data_dir=Path(CATEGORIES[category]),
                    source_category=category,
                    chroma_host=chroma_host,
                    chroma_port=chroma_port,
                    duckdb_path=Path(duckdb_path),
                )
                st.write(t("admin_index_category_done", category=category, count="?"))

            status.update(
                label=t("admin_index_done"),
                state="complete",
                expanded=False,
            )

        except Exception as exc:
            logger.exception("Indexing failed: %s", exc)
            st.error(t("admin_index_error", error=str(exc)))
            status.update(
                label=t("admin_index_error", error=str(exc)),
                state="error",
                expanded=True,
            )