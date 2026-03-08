# 🍳 ChefRAG

**ChefRAG** is a personal culinary assistant that uses RAG (Retrieval-Augmented Generation) on your own recipe collection exported from [Umami](https://umami.recipes).

You describe what ingredients you have — ChefRAG finds matching recipes from your personal collection and explains why each one fits.

---

## Features

- 🔍 **Semantic search** on your own recipes (ingredients, descriptions, instructions)
- 🗂️ **Metadata filtering** by cook time, difficulty, cuisine type, and category
- 💬 **Conversational interface** with clarifying questions before suggesting recipes
- 🌍 **Bilingual UI** (French / English)
- 🔒 **Single-password authentication** with bcrypt + rate limiting
- ♿ **WCAG 2.1 AA** accessible interface

## Stack

| Layer          | Technology                 |
| -------------- | -------------------------- |
| UI             | Streamlit                  |
| LLM            | Anthropic Claude API       |
| RAG            | LlamaIndex                 |
| Vector store   | ChromaDB                   |
| Metadata       | DuckDB                     |
| Orchestration  | Apache Airflow _(Step 6)_  |
| Infrastructure | Docker + Hetzner + Coolify |

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/chefrag.git
cd chefrag

# 2. Set up environment variables
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY and APP_PASSWORD

# 3. Generate a bcrypt password hash
python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"
# Paste the output as APP_PASSWORD in .env

# 4. Add your Umami JSON exports
# Place favorites in data/favorites/
# Place new recipes in data/new/

# 5. Start the app
docker compose up --build
# Open http://localhost:8501
```

## Recipe exports

Export recipes from the Umami app as JSON (Schema.org format) and place them in:

- `data/favorites/` — your Favorites collection
- `data/new/` — your New recipes collection

## Development

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start ChromaDB (Docker required)
make chroma

# Index your recipes (run once, then after each new export)
make index

# Launch the app
make run

# Run tests
make test

# Lint
make lint
```

> **Note:** Set `CHROMA_HOST=localhost` in your `.env` for local development.
> Use `CHROMA_HOST=chefrag-chroma` when running inside Docker Compose.

## Project documentation

- [`CONTEXT.md`](CONTEXT.md) — project overview, goals, stack
- [`STRUCTURE.md`](STRUCTURE.md) — folder/file structure explained
- [`NEXT_STEPS.md`](NEXT_STEPS.md) — task board
- [`DECISIONS.md`](DECISIONS.md) — architectural and technical decisions log

## License

MIT
