"""Airflow DAG placeholder — replaced by Streamlit Admin tab (ADR-015).

This DAG was planned to watch data/favorites/ and data/new/ for new Umami
JSON exports and trigger automatic re-indexing into ChromaDB and DuckDB.

Decision: Airflow was removed from the project (see DECISIONS.md — ADR-015).
Re-indexing is now triggered manually via the Admin tab in the Streamlit UI.

This file is kept as a reference for a future Airflow integration if needed.
"""

# from airflow import DAG
# from airflow.sensors.filesystem import FileSensor
# ...
# Implementation deferred — see ADR-015.