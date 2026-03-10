# 🍳 ChefRAG

**ChefRAG** is a personal culinary assistant that uses RAG (Retrieval-Augmented Generation) on your own recipe collection exported from [Umami](https://umami.recipes).

You describe what ingredients you have — ChefRAG finds matching recipes from your personal collection and explains why each one fits.

---

## 🌐 Live demo

A live instance is deployed at **[chefrag.lumafinch.com](https://chefrag.lumafinch.com)**.

> This instance runs on my personal recipe collection. To request a demo password, contact me at [hey@jeremymarchandeau.com](mailto:hey@jeremymarchandeau.com).

---

## Features

- 🔍 **Semantic search** on your own recipes (ingredients, descriptions, instructions)
- 🗂️ **Metadata filtering** by cook time, difficulty, cuisine type, and category
- 💬 **Conversational interface** with clarifying questions before suggesting recipes
- 🌍 **Bilingual UI** (French / English)
- 🔒 **Single-password authentication** with bcrypt + rate limiting
- ♿ **WCAG 2.1 AA** accessible interface

---

## Stack

| Layer          | Technology                 |
| -------------- | -------------------------- |
| UI             | Streamlit                  |
| LLM            | Anthropic Claude API       |
| RAG            | LlamaIndex                 |
| Vector store   | ChromaDB                   |
| Metadata       | DuckDB                     |
| Infrastructure | Docker + Hetzner + Coolify |

---

## Quick start (Docker)

The recommended way to run ChefRAG in production or for a full end-to-end test.

```bash
# 1. Clone the repo
git clone https://github.com/your-username/chefrag.git
cd chefrag

# 2. Set up environment variables
cp .env.example .env
# Edit .env:
#   - ANTHROPIC_API_KEY=your_key
#   - APP_PASSWORD=your_bcrypt_hash (see below)
#   - CHROMA_HOST=chefrag-chroma   ← use this value for Docker
#   - CHROMA_PORT=8000
#   - DUCKDB_PATH=storage/duckdb/chefrag.duckdb

# 3. Generate a bcrypt password hash
python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"
# Paste the output as APP_PASSWORD in .env

# 4. Add your Umami JSON exports
# Place favorites in data/favorites/
# Place new recipes in data/new/

# 5. Build and start all services
make docker-up
# App is available at http://localhost:8501

# 6. Index your recipes (first run, and after each new export)
make docker-index
```

---

## Local development (without Docker)

Faster iteration — Streamlit reloads on file save. ChromaDB still runs in Docker.

### Prerequisites

- Python 3.11+
- Docker (for ChromaDB only)
- `make`

### Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env:
#   - ANTHROPIC_API_KEY=your_key
#   - APP_PASSWORD=your_bcrypt_hash
#   - CHROMA_HOST=localhost   ← use this value for local dev
#   - CHROMA_PORT=8000
#   - DUCKDB_PATH=storage/duckdb/chefrag.duckdb

# 4. Create local storage directories
mkdir -p storage/duckdb storage/chroma
```

### Daily workflow

```bash
# Start ChromaDB (required before running the app or indexing)
make chroma

# Index your recipes (first run, and after each new export)
make index

# Launch the app — available at http://localhost:8501
make run

# Run tests
make test

# Lint
make lint
```

> **Note:** `make index` and `make run` use `CHROMA_HOST=localhost`.
> Switch to `make docker-index` when working inside Docker.

---

## Makefile reference

| Command             | Description                                  |
| ------------------- | -------------------------------------------- |
| `make run`          | Launch Streamlit app (local)                 |
| `make test`         | Run pytest test suite (local)                |
| `make lint`         | Run Ruff linter on `app/`                    |
| `make chroma`       | Start ChromaDB container in background       |
| `make index`        | Index recipes into ChromaDB + DuckDB (local) |
| `make docker-up`    | Build and start all Docker services          |
| `make docker-down`  | Stop all Docker services                     |
| `make docker-index` | Index recipes inside the Docker container    |
| `make docker-test`  | Run tests inside the Docker container        |

---

## Recipe exports

Export recipes from the Umami app as JSON (Schema.org format) and place them in:

- `data/favorites/` — your Favorites collection
- `data/new/` — your New recipes collection

Then run `make index` (local) or `make docker-index` (Docker).

> **Note:** `data/` is gitignored — your personal recipes are never committed to the repository.

---

## Environment variables

See `.env.example` for the full list. Key variables:

| Variable               | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `ANTHROPIC_API_KEY`    | Anthropic API key                                    |
| `APP_PASSWORD`         | bcrypt-hashed app password                           |
| `CHROMA_HOST`          | `localhost` (local dev) or `chefrag-chroma` (Docker) |
| `CHROMA_PORT`          | ChromaDB port (default: `8000`)                      |
| `DUCKDB_PATH`          | Path to `.duckdb` file                               |
| `APP_LANGUAGE_DEFAULT` | Default UI language (`fr` or `en`)                   |

---

## Project documentation

- [`CONTEXT.md`](CONTEXT.md) — project overview, goals, stack
- [`STRUCTURE.md`](STRUCTURE.md) — folder/file structure explained
- [`NEXT_STEPS.md`](NEXT_STEPS.md) — task board
- [`DECISIONS.md`](DECISIONS.md) — architectural and technical decisions log

---

## License

MIT — see [`LICENSE`](LICENSE)
