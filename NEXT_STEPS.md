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

## ✅ Step 2 — Authentication (`app/auth.py`)

- [x] `verify_password(plain, hashed)` — bcrypt check
- [x] `is_locked_out(session_state)` + `record_failed_attempt()` — rate limiting (5 attempts, 5 min lockout)
- [x] `login_form(translations)` — Streamlit form with i18n strings
- [x] `logout(session_state)` — clears auth state
- [x] `test_auth.py` — unit tests (valid pwd, invalid pwd, lockout)

---

## ✅ Step 3 — Indexer (`app/indexer.py`)

- [x] Pydantic model: `Recipe`
- [x] `parse_recipe_json(filepath, source_category)` — Schema.org → Recipe
- [x] `parse_iso_duration(duration_str)` → total minutes (int)
- [x] `extract_clean_ingredients(raw)` — bullet filter + narrative removal
- [x] `parse_cuisine_tags(description)` — tag extraction from description field
- [x] `build_recipe_document(recipe)` — text document for embedding
- [x] `index_recipe_in_chroma(recipe, client, embed_model)` — upsert to ChromaDB
- [x] `index_recipe_in_duckdb(recipe, conn)` — upsert to DuckDB
- [x] `index_recipe(recipe, chroma, duckdb, embed_model)` — combined entry point
- [x] `run_indexing(data_dir, category, ...)` — full pipeline for one category
- [x] `test_indexer.py` — unit tests with real Umami JSON fixture files
- [x] `tests/fixtures/` — real JSON exports used as test data

---

## ✅ Step 4 — Agent (`app/agent.py`)

- [x] `RecipeSearchTool` — semantic search in ChromaDB via LlamaIndex embeddings
- [x] `MetadataFilterTool` — structured filter on DuckDB (time, cuisine, category)
- [x] `ChefRagAgent` — stateless agent: `chat(messages, language, filters) -> str`
- [x] `build_agent(chroma_host, chroma_port, duckdb_path)` — factory
- [x] `stream_agent_response(agent, user_message, category, language)` — streaming wrapper for UI
- [x] System prompt: clarifying questions, bilingual (FR/EN), explain matches
- [x] RAG context injected into last user message before Claude API call
- [x] `test_agent.py` — unit tests with fully mocked dependencies

---

## ✅ Step 5 — UI (`app/main.py`)

- [x] `load_translations(lang)` — loads `app/i18n/{lang}.json`
- [x] `t(key, **kwargs)` — translation helper with placeholder support
- [x] `init_session_state()` — initialises all session state keys
- [x] Language toggle in header (persisted in session state, triggers rerun)
- [x] Login page — delegates entirely to `auth.login_form(translations)`
- [x] Chat interface — calls `agent.build_agent()` + `agent.stream_agent_response()`
- [x] Recipe category selector (all / favorites / new)
- [x] Logout button — calls `auth.logout()`
- [x] Responsive layout (mobile, tablet, desktop) via injected CSS
- [x] WCAG 2.1 AA: focus indicators, ARIA labels, sr-only labels, skip link
- [x] `app/i18n/fr.json` + `app/i18n/en.json` — complete strings for Step 5
- [x] `Makefile` — `make run`, `make test`, `make lint`, `make chroma`, `make index`
- [x] `python-dotenv` — `.env` loaded at startup via `load_dotenv()`

---

## 🔲 Step 5b — UX improvements (`app/main.py` + `app/agent.py`)

> To be scoped after first end-to-end test session.

- [ ] Review and improve agent response quality (relevance, tone, match explanation)
- [ ] Improve chat UX (message rendering, loading states, error handling)
- [ ] Review i18n strings for naturalness (FR especially)
- [ ] Add conversation history passed to `agent.chat()` (currently single-turn)

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
