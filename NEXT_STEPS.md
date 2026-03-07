# ChefRAG — Next Steps

## ✅ Step 1 — Scaffold

- [x] Project structure
- [x] `requirements.txt`
- [x] `.env.example`
- [x] `Dockerfile`
- [x] `docker-compose.yml` (chefrag-ui + chefrag-chroma)
- [x] `pyproject.toml` (Black + Ruff + pytest)
- [x] `app/i18n/fr.json` + `app/i18n/en.json`
- [x] Placeholder files for all modules
- [x] `README.md`, `STRUCTURE.md`, `NEXT_STEPS.md`, `DECISIONS.md`

---

## 🔲 Step 2 — Authentication (`app/auth.py`)

- [ ] `verify_password(plain, hashed)` — bcrypt check
- [ ] `check_rate_limit(session_state)` — lock after 5 failed attempts
- [ ] `login_form()` — Streamlit form with i18n strings
- [ ] `test_auth.py` — unit tests (valid pwd, invalid pwd, lockout)

---

## 🔲 Step 3 — Indexer (`app/indexer.py`)

- [ ] Pydantic model: `Recipe`
- [ ] `parse_recipe_json(filepath)` — Schema.org → Recipe
- [ ] `parse_iso_duration(duration_str)` → total minutes (int)
- [ ] `index_recipe(recipe, chroma_client, duckdb_conn)` — embed + store
- [ ] `run_indexing(data_dir, category)` — full pipeline for one category
- [ ] `test_indexer.py` — unit tests with fixture JSON files

---

## 🔲 Step 4 — Agent (`app/agent.py`)

- [ ] `RecipeSearchTool` — semantic search in ChromaDB via LlamaIndex
- [ ] `MetadataFilterTool` — structured filter on DuckDB
- [ ] `build_agent()` — LlamaIndex ReActAgent with system prompt
- [ ] System prompt: clarifying questions, bilingual, explain matches
- [ ] `test_agent.py` — unit tests (mocked tools)

---

## 🔲 Step 5 — UI (`app/main.py`)

- [ ] i18n loader: `load_translations(lang)`
- [ ] Language toggle in header (persisted in session state)
- [ ] Login page (calls `auth.py`)
- [ ] Chat interface (calls `agent.py`)
- [ ] Responsive layout (mobile, tablet, desktop)
- [ ] WCAG 2.1 AA compliance

---

## 🔲 Step 6 — Airflow DAG (`dags/reindex_dag.py`)

- [ ] Add Airflow service to `docker-compose.yml`
- [ ] `reindex_recipes_dag` with FileSensor
- [ ] Tasks: detect → parse → clear → embed → metadata → notify
- [ ] Memory/resource tuning for CPX21

---

## 🔲 Step 7 — Deployment

- [ ] Final Docker Compose validation
- [ ] Coolify environment variable setup guide
- [ ] SSL + domain configuration
- [ ] End-to-end smoke test on Hetzner CPX21
- [ ] `DECISIONS.md` finalized
