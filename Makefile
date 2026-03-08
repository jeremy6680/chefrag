.PHONY: run test lint chroma

chroma:
	docker compose up chefrag-chroma -d

run:
	.venv/bin/streamlit run app/main.py

test:
	.venv/bin/pytest tests/

lint:
	.venv/bin/ruff check app/

index:
	.venv/bin/python -c "\
from pathlib import Path; \
from app.indexer import run_indexing; \
run_indexing(Path('data/favorites'), 'favorites', 'localhost', 8000, Path('storage/duckdb/chefrag.duckdb')); \
run_indexing(Path('data/new'), 'new', 'localhost', 8000, Path('storage/duckdb/chefrag.duckdb'))"