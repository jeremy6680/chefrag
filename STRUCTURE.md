# ChefRAG — Structure

## Folder / file map

```
chefrag/
│
├── data/                          # Recipe JSON exports (gitignored)
│   ├── favorites/                 # Umami "Favorites" category
│   └── new/                       # Umami "New" category
│
├── app/                           # Application source code
│   ├── __init__.py
│   ├── main.py                    # Streamlit entry point + chat UI
│   ├── agent.py                   # LlamaIndex agent, RAG tools, system prompt
│   ├── indexer.py                 # JSON parser, ChromaDB writer, DuckDB writer
│   ├── auth.py                    # bcrypt auth, rate limiting
│   └── i18n/
│       ├── fr.json                # French static strings
│       └── en.json                # English static strings
│
├── dags/
│   └── reindex_dag.py             # Airflow DAG (Step 6)
│
├── storage/                       # Persistent runtime data (gitignored)
│   ├── chroma/                    # ChromaDB embeddings
│   └── duckdb/                    # chefrag.duckdb metadata database
│
├── tests/
│   ├── __init__.py
│   ├── test_auth.py               # Auth unit tests (Step 2)
│   ├── test_indexer.py            # Indexer unit tests (Step 3)
│   └── test_agent.py              # Agent unit tests (Step 4)
│
├── .env.example                   # Env vars template (safe to commit)
├── .gitignore
├── docker-compose.yml             # Services: chefrag-ui + chefrag-chroma
├── Dockerfile                     # chefrag-ui image
├── pyproject.toml                 # Black + Ruff + pytest config
├── requirements.txt
│
├── CONTEXT.md                     # Project overview (source of truth)
├── DECISIONS.md                   # Architectural decisions log
├── NEXT_STEPS.md                  # Task board
├── STRUCTURE.md                   # This file
└── README.md
```

## Key design decisions

### DuckDB — no dedicated container

DuckDB is a file-based database. It runs inside `chefrag-ui` via a mounted volume (`./storage/duckdb`). No separate Docker service is needed, which saves ~200MB RAM.

### ChromaDB — dedicated container

ChromaDB needs a persistent HTTP server that both the Streamlit app and the Airflow DAG can access. It runs as `chefrag-chroma` on port 8000.

### Airflow — deferred to Step 6

On CPX21 (4GB RAM), Airflow adds ~1.5–2GB overhead. It is deliberately excluded from the initial Docker Compose to keep the development environment lean. It will be added as a separate service in Step 6.

### Embeddings — local (HuggingFace)

Using `llama-index-embeddings-huggingface` (sentence-transformers) instead of the OpenAI embeddings API to avoid per-token costs on indexing. The model runs inside the `chefrag-ui` container.

## Data flow

```
Umami JSON export
       ↓
   indexer.py
   ├── parse Schema.org JSON
   ├── validate fields (Pydantic)
   ├── generate embeddings (HuggingFace)
   ├── store vectors → ChromaDB
   └── store metadata → DuckDB
       ↓
   agent.py (LlamaIndex)
   ├── RecipeSearchTool → ChromaDB (semantic)
   └── MetadataFilterTool → DuckDB (structured)
       ↓
   main.py (Streamlit)
   └── chat interface → user
```
