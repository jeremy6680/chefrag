# ChefRAG — Project Context

## What this project is

ChefRAG is a personal culinary assistant web app that uses RAG (Retrieval-Augmented Generation) on recipes exported from the Umami app (JSON Schema.org format). The user can chat in natural language to get recipe suggestions based on available ingredients, cooking time, difficulty, and cuisine preferences.

## Author context

- Senior web developer (12+ years, WordPress/PHP/JS) transitioning to data/analytics engineering
- Stack: Python, SQL, dbt, DuckDB, LlamaIndex, Airflow
- Deployed on personal Hetzner VPS (CPX21 — 4GB RAM, Ubuntu) managed via Coolify
- Project will be open-sourced on GitHub (private initially)

## Core user flow

1. User logs in with a password (single shared password, no multi-user)
2. User types what ingredients they have: e.g. "I have 2 eggs, grated cheese and ham"
3. Agent asks clarifying questions:
   - Difficulty: Easy / Medium / Hard / Any?
   - Cooking time: < 15 min / 30 min / 1h / Any?
   - Spice level: Mild / Medium / Spicy / Any?
   - Cuisine type: Asian / French / Italian / Any?
4. Agent retrieves matching recipes from the personal Umami collection
5. Agent explains why each suggestion matches the user's criteria

---

# Stack

| Layer            | Technology              | Role                                                 |
| ---------------- | ----------------------- | ---------------------------------------------------- |
| UI               | Streamlit               | Chat interface, auth, i18n (FR/EN)                   |
| LLM              | Anthropic Claude API    | Response generation, agent orchestration             |
| RAG / Agents     | LlamaIndex              | Semantic indexing, retrieval, conversational agent   |
| Vector store     | ChromaDB                | Persistent embeddings storage                        |
| Metadata         | DuckDB                  | Fast filtering (time, difficulty, cuisine, category) |
| Orchestration    | Apache Airflow          | DAG for automatic re-indexing on new JSON exports    |
| Containerization | Docker + Docker Compose | All services                                         |
| Infra            | Hetzner CPX21 + Coolify | VPS hosting, auto-deploy from GitHub, SSL            |
| Versioning       | GitHub                  | Public open-source repo                              |
| Language         | Python 3.11+            | All code documented in English                       |

---

# Project Structure

```
chefrag/
├── data/
│   ├── favorites/          # Umami JSON exports — "Favorites" category
│   └── new/                # Umami JSON exports — "New" category
├── app/
│   ├── main.py             # Streamlit entry point
│   ├── agent.py            # LlamaIndex agent + RAG logic
│   ├── indexer.py          # JSON parser + ChromaDB + DuckDB indexing
│   ├── auth.py             # Password-based authentication (bcrypt)
│   └── i18n/
│       ├── fr.json         # French static strings
│       └── en.json         # English static strings
├── dags/
│   └── reindex_dag.py      # Airflow DAG — auto re-indexing pipeline
├── storage/
│   ├── chroma/             # ChromaDB persistent volume
│   └── duckdb/             # DuckDB persistent volume
├── tests/
│   ├── test_indexer.py
│   ├── test_auth.py
│   └── test_agent.py
├── docker-compose.yml
├── Dockerfile
├── .env.example            # Template — never commit actual .env
├── requirements.txt
├── CONTEXT.md              # This file
├── NEXT_STEPS.md           # Task board
├── STRUCTURE.md            # Architecture details
└── README.md
```

---

# Docker Services

| Service           | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `chefrag-ui`      | Streamlit app — port 8501                               |
| `chefrag-airflow` | Airflow webserver + scheduler                           |
| `chefrag-chroma`  | ChromaDB server — port 8000                             |
| `chefrag-duck`    | DuckDB via mounted volume (no dedicated service needed) |

---

# Airflow DAG — `reindex_recipes_dag`

**Triggers:** FileSensor on `data/favorites/` and `data/new/` + manual trigger

**Tasks in order:**

1. `detect_new_exports` — check for new/modified JSON files
2. `parse_and_validate` — parse Schema.org JSON, validate required fields
3. `clear_old_index` — drop existing ChromaDB collection for that category
4. `embed_and_store` — generate embeddings, store in ChromaDB
5. `update_metadata` — upsert DuckDB table with recipe metadata
6. `notify_done` — log confirmation

---

# LlamaIndex Agent

**Two tools:**

- `RecipeSearchTool` — semantic search in ChromaDB (ingredients, description, instructions)
- `MetadataFilterTool` — filter on DuckDB (cook time, difficulty, cuisine type, category)

**System prompt behavior:**

- Ask clarifying questions before suggesting recipes
- Explain why each suggestion matches the user's available ingredients
- Respond in the language selected by the user (FR or EN)
- If no match found, say so clearly and suggest relaxing filters

---

# Recipe Data Model (from Umami JSON export)

```python
# Fields extracted from Schema.org JSON
{
    "name": str,
    "description": str,           # cuisine tags (e.g. "Side, Asian, Vegetarian")
    "url": str,
    "prepTime": str,               # ISO 8601 duration
    "cookTime": str,               # ISO 8601 duration
    "totalTime": str,              # ISO 8601 duration
    "recipeYield": str,
    "recipeCategory": str,         # "Favorites" or "New"
    "recipeCuisine": str,
    "recipeIngredient": list[str],
    "recipeInstructions": list[dict],
    "nutrition": dict,
    "keywords": str,
    "source_category": str,        # "favorites" or "new" (derived from folder)
}
```

---

# Environment Variables

See `.env.example` for all required variables. Never hardcode secrets.

| Variable                 | Description                                |
| ------------------------ | ------------------------------------------ |
| `ANTHROPIC_API_KEY`      | Anthropic API key                          |
| `APP_PASSWORD`           | bcrypt-hashed password for app access      |
| `APP_LANGUAGE_DEFAULT`   | Default UI language (`fr` or `en`)         |
| `CHROMA_HOST`            | ChromaDB host (default: `chefrag-chroma`)  |
| `CHROMA_PORT`            | ChromaDB port (default: `8000`)            |
| `DUCKDB_PATH`            | Path to .duckdb file                       |
| `AIRFLOW_ADMIN_USER`     | Airflow admin username                     |
| `AIRFLOW_ADMIN_PASSWORD` | Airflow admin password                     |
| `LOG_LEVEL`              | Logging level (`INFO`, `WARNING`, `ERROR`) |

---

# Security Rules

- Password stored as bcrypt hash in env var only — never in code
- Rate limiting: lock after 5 failed login attempts
- HTTPS enforced in production (Coolify + Let's Encrypt)
- `.env` is gitignored — only `.env.example` is committed
- No user query logging in production
- API key injected by Coolify at runtime

---

# UI Requirements

## Responsive breakpoints

- Desktop (≥ 1024px): standard layout
- Tablet (768–1023px): adjusted margins
- Mobile (< 768px): full-width, fixed input bar at bottom

## Accessibility (WCAG 2.1 AA)

- Color contrast minimum 4.5:1 (normal text), 3:1 (large text)
- Full keyboard navigation (Tab, Enter, Escape)
- ARIA attributes on all interactive components
- Alt text on all images and icons
- Visible focus indicator at all times

## i18n

- All static strings (labels, errors, placeholders, agent questions) in `app/i18n/fr.json` and `app/i18n/en.json`
- Recipe content stays in its original language (Umami export language)
- Language toggle in the header — persisted in session state

---

# Code Standards

- **Language:** All code, comments, docstrings, variable/function/class names in English
- **Formatting:** Black (auto-format), Ruff (linting)
- **Docstrings:** Google Style on all functions and classes
- **Type hints:** Required on all functions
- **No magic numbers:** All constants are named
- **Tests:** pytest, minimum 70% coverage

---

# Deployment (Hetzner + Coolify)

- Push to `main` → Coolify auto-deploys
- Environment variables set in Coolify UI (never in repo)
- Docker Compose volumes mounted for persistence (`data/`, `storage/`)
- SSL automatic via Let's Encrypt
- Target server: existing Hetzner CPX21

---

# Out of Scope (MVP)

- Multi-user / role management
- Direct Umami API import (manual JSON export only)
- Shopping list generation
- Recipe rating or comments
- Native mobile app
- Non-JSON export formats

---

# Recipe Categories

| Category  | Folder            | ChromaDB Collection |
| --------- | ----------------- | ------------------- |
| Favorites | `data/favorites/` | `recipes_favorites` |
| New       | `data/new/`       | `recipes_new`       |

User can query one category or both simultaneously.
