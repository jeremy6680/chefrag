# ChefRAG — Decisions Log

Architectural and technical decisions, with rationale.

---

## ADR-001 — Airflow excluded from initial Docker Compose

**Date:** Step 1  
**Status:** Accepted

**Context:** The target server is a Hetzner CPX21 (4GB RAM). Airflow (webserver + scheduler) consumes approximately 1.5–2GB of RAM on its own.

**Decision:** Airflow is not included in `docker-compose.yml` for Steps 1–5. It will be added as a separate service in Step 6, once the core app is validated.

**Consequences:** The re-indexing pipeline must be triggered manually during development (by running `indexer.py` directly). This is acceptable for the MVP phase.

---

## ADR-002 — Local embeddings via HuggingFace (sentence-transformers)

**Date:** Step 1  
**Status:** Accepted

**Context:** LlamaIndex supports multiple embedding providers. The OpenAI embeddings API charges per token, which adds cost every time the recipe index is rebuilt.

**Decision:** Use `llama-index-embeddings-huggingface` with a local sentence-transformers model (e.g. `BAAI/bge-small-en-v1.5`). The model runs inside the `chefrag-ui` container.

**Consequences:** First container startup will download the model (~130MB). Embedding generation is CPU-bound but acceptable given the small recipe dataset. No API cost for indexing.

---

## ADR-003 — DuckDB without a dedicated container

**Date:** Step 1  
**Status:** Accepted

**Context:** DuckDB is a file-based, in-process database. It does not require a network server.

**Decision:** DuckDB runs as a library inside the `chefrag-ui` container. The `.duckdb` file is persisted via a mounted volume (`./storage/duckdb`).

**Consequences:** DuckDB cannot be accessed concurrently from multiple containers (e.g. Airflow + Streamlit simultaneously). This is mitigated by Airflow writing to DuckDB only during scheduled re-indexing (when the app is idle). Acceptable for a single-user personal app.

---

## ADR-004 — Single shared password authentication

**Date:** Step 1  
**Status:** Accepted

**Context:** ChefRAG is a personal app with a single user.

**Decision:** Authentication uses a single bcrypt-hashed password stored in an environment variable (`APP_PASSWORD`). No user table, no JWT, no session database. Rate limiting (5 attempts) implemented in Streamlit session state.

**Consequences:** No multi-user support (explicitly out of scope for MVP). Password rotation requires updating the env var and redeploying.

---

## ADR-005 — Dependency pinning strategy (revised)

**Date:** Step 1 — Revised Step 2 (setup)  
**Status:** Accepted

**History:**

- **Step 1 (initial):** llama-index packages pinned with loose constraints (`>=`) to handle frequent version churn. Stable packages (Streamlit, DuckDB, bcrypt) pinned exactly.
- **Step 2 (revised):** Loose constraints caused `ResolutionTooDeep` (200,000 rounds) — pip could not resolve the dependency graph. Strategy reversed.

**Final decision:** All packages in `requirements.txt` are pinned exactly (`==`). A `requirements.lock` file (`pip freeze > requirements.lock`) is committed alongside `requirements.txt` to guarantee reproducible installs.

**Consequences:** Upgrades require testing the full dependency set together and updating both files. Individual package bumps are not safe in isolation.

---

## ADR-006 — llama-index-llms-anthropic excluded; Anthropic SDK called directly

**Date:** Step 2 (setup)  
**Status:** Accepted

**Context:** `llama-index-llms-anthropic` declares `anthropic[bedrock,vertex]>=0.75.0` as a dependency — it forces the installation of AWS Bedrock and Google Vertex AI extras that ChefRAG does not need. These extras introduce transitive conflicts that pip cannot resolve regardless of version.

**Decision:** `llama-index-llms-anthropic` is not installed. The agent in `app/agent.py` calls the `anthropic` SDK directly for LLM generation. LlamaIndex is used only for its RAG capabilities (ChromaDB retrieval, embeddings, query engine).

**Consequences:** The LlamaIndex `ReActAgent` cannot use the built-in Anthropic LLM wrapper. The agent orchestration layer calls `anthropic.Anthropic().messages.create()` directly and passes retrieved context manually. This is slightly more code but gives full control over the prompt and avoids the dependency conflict entirely.

---

## ADR-007 — fastapi and pydantic versions validated at install time

**Date:** Step 2 (setup)  
**Status:** Accepted

**Context:** `pydantic==2.7.1` (originally specified) and a missing `fastapi` entry caused resolution failures. `llama-index-core` and `chromadb` both require more recent versions.

**Decision:** `pydantic` pinned to `2.11.5` and `fastapi` to `0.111.0`, as validated by a clean venv install. These are the versions pip resolved when all other constraints were satisfied.

**Consequences:** None — these are transitive dependencies that don't affect application code directly.

---

## ADR-008 — Ingredient parsing strategy: bullet filter + full-text embedding

**Date:** Step 3  
**Status:** Accepted

**Context:** Umami JSON exports mix real ingredient lines (prefixed with `•`) with narrative text (section headers, cooking notes). Both types of content are present in the same `recipeIngredient` array.

**Decision:** Two representations are maintained:

- `ingredients_raw` — the full unfiltered list, used for embedding in ChromaDB. Narrative context improves semantic search coverage for edge cases (e.g. "fermented black bean paste").
- `ingredients_clean` — only bullet-prefixed lines, stripped of the `•`. Used for display in the UI and stored in DuckDB for structured filtering.

For exports without any bullet characters (fallback), all lines above a minimum length threshold are kept in `ingredients_clean`.

**Consequences:** ChromaDB embeddings contain richer context at the cost of slight noise. DuckDB stores clean ingredient lists suitable for display. No information is lost.

---

## ADR-009 — Cuisine tags parsed from `description` field, not `recipeCuisine`

**Date:** Step 3  
**Status:** Accepted

**Context:** The Umami `recipeCuisine` field is unreliable — for example, the "One Pan Mexican Quinoa" recipe has `recipeCuisine: "Vegetarian"`, which is a dietary type, not a cuisine. The `description` field consistently contains a comma-separated list of accurate tags (e.g. `"Korean, Chinese, Main, Vegetarian-adaptable, Asian"`).

**Decision:** A `cuisine_tags` field is derived by splitting the `description` string on commas. Both `cuisine_tags` and the original `recipe_cuisine` are stored (the latter for backward compatibility with the Umami data model).

**Consequences:** Cuisine-based filtering and search rely on `cuisine_tags`. The `recipe_cuisine` field is preserved in DuckDB but treated as secondary.

---

## ADR-010 — Stateless agent; conversation history managed by Streamlit

**Date:** Step 4  
**Status:** Accepted

**Context:** The agent could manage its own conversational state (clarification phase → search phase → done) via an internal state machine. This would guarantee correct flow regardless of user input.

**Decision:** The agent is stateless. `ChefRagAgent.chat()` accepts the full conversation history on every call and returns a single reply. Streamlit session state in `main.py` owns the history. The system prompt instructs Claude to handle the clarification flow naturally.

**Consequences:** Simpler code, simpler tests, easier to debug. Claude handles the clarification flow reliably via the system prompt for single-user personal use. A state machine can be added in a future iteration if the conversational behaviour is found to be insufficient.

---

## ADR-011 — RAG: semantic search intersected with metadata filter

**Date:** Step 4  
**Status:** Accepted

**Context:** Two tools are available: ChromaDB (semantic) and DuckDB (structured). Using only one loses either semantic relevance or hard constraints like cook time.

**Decision:** Both tools are run in parallel. The agent intersects semantic results with metadata-filtered results. If the intersection is empty (no recipe matches both criteria), it falls back to semantic-only results rather than returning nothing.

**Consequences:** Users always get suggestions even when filters are strict. The fallback is clearly explained in the system prompt so Claude can tell the user which constraint could not be satisfied.

---

## ADR-012 — python-dotenv for local development

**Date:** Step 5
**Status:** Accepted

**Context:** Streamlit does not load `.env` files automatically. Environment
variables set in `.env` were not visible to the app when running locally,
causing `APP_PASSWORD` and other variables to appear unset.

**Decision:** `python-dotenv` is added to `requirements.txt`. `load_dotenv()`
is called at the top of `app/main.py` before any other imports that read env
vars. In production (Coolify), variables are injected directly by the platform
and no `.env` file is present — `load_dotenv()` is a no-op in that case.

**Consequences:** None in production. Local dev works without manual `export`
commands.

---

## ADR-013 — CHROMA_HOST differs between local dev and Docker

**Date:** Step 5
**Status:** Accepted

**Context:** ChromaDB runs as `chefrag-chroma` in Docker Compose. This
hostname is only resolvable inside the Docker network. When running Streamlit
locally (outside Docker) against a locally-exposed ChromaDB container,
`chefrag-chroma` cannot be resolved.

**Decision:** `.env.example` documents both values with a comment. Developers
must set `CHROMA_HOST=localhost` for local dev and `CHROMA_HOST=chefrag-chroma`
for Docker / production.

**Consequences:** One manual `.env` change required when switching between
local and Docker environments. Acceptable for a single-developer project.

---

## ADR-014 — Makefile for local development commands

**Date:** Step 5
**Status:** Accepted

**Context:** Several commands need to be run repeatedly during development:
starting ChromaDB, launching Streamlit via the venv, running the indexer,
running tests. Typing full paths each time is error-prone.

**Decision:** A `Makefile` is added at the project root with targets: `run`,
`test`, `lint`, `chroma`, `index`. All targets use `.venv/bin/` explicitly to
avoid pyenv shim interference.

**Consequences:** `make` must be available (standard on macOS/Linux). The
`index` target hardcodes `localhost` — must be updated if the ChromaDB host
changes.

---

## ADR-015 — Airflow remplacé par une interface admin Streamlit

**Date:** Step 6  
**Status:** Accepted

**Context:** Airflow (webserver + scheduler) consomme ~1.5–2GB RAM sur un CPX21 (4GB total).
L'usage prévu était un FileSensor sur `data/favorites/` et `data/new/` pour déclencher
le ré-indexation après chaque export Umami. Or, Streamlit dispose d'un widget
`st.file_uploader` natif qui permet d'uploader des JSON directement depuis le navigateur.

**Decision:** Airflow est supprimé du projet. Un onglet "Admin" est ajouté à l'interface
Streamlit existante. Il permet d'uploader des fichiers JSON Umami par catégorie et de
lancer `run_indexing()` directement depuis l'UI, avec retour visuel en temps réel via
`st.status()`. Le fichier `dags/reindex_dag.py` est conservé comme placeholder commenté
pour référence future.

**Consequences:** Pas de ré-indexation automatique (FileSensor supprimé). L'indexation
est déclenchée manuellement par l'utilisateur via l'UI — ce qui est acceptable pour un
usage solo. Économie de ~1.5GB RAM sur le VPS. Zéro dépendance supplémentaire.

---

## ADR-016 — Public GitHub repository

**Date:** Step 7
**Status:** Accepted

**Context:** ChefRAG was initially developed in a private GitHub repository.
After completing the MVP (Steps 1–7), the project is ready to be open-sourced.

**Decision:** The repository is made public under the MIT License. Personal data
is protected by `.gitignore` (`.env`, `data/`, `storage/`). A live demo is
accessible at chefrag.lumafinch.com — a password is required and available on
request.

**Consequences:** The codebase, architecture decisions, and documentation are
publicly visible. Secrets and personal recipe data remain private via
`.gitignore`. The `tests/fixtures/` directory contains sample Umami JSON exports
used for testing — these are non-sensitive recipe data included intentionally as
examples for contributors.
