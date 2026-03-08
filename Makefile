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

# Docker targets
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-index:
	docker compose exec chefrag-ui python -c "\
from pathlib import Path; \
from app.indexer import run_indexing; \
run_indexing(Path('data/favorites'), 'favorites', 'chefrag-chroma', 8000, Path('storage/duckdb/chefrag.duckdb')); \
run_indexing(Path('data/new'), 'new', 'chefrag-chroma', 8000, Path('storage/duckdb/chefrag.duckdb'))"

docker-test:
	docker compose run --rm chefrag-ui python -m pytest tests/