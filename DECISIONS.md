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
