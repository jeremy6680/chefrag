"""
ChefRAG — Agent module.

Implements the conversational culinary assistant using:
- ChromaDB for semantic recipe search (via LlamaIndex HuggingFace embeddings)
- DuckDB for structured metadata filtering (cook time, cuisine, category)
- Anthropic Claude API called directly for response generation (ADR-006)

The agent is stateless: conversation history is passed in full on every call.
State management (session history, language) is handled by main.py (Streamlit).

Public API:
    build_agent(chroma_host, chroma_port, duckdb_path) -> ChefRagAgent
    agent.chat(messages, language) -> str
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import chromadb
import duckdb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.indexer import CHROMA_COLLECTION_PREFIX, DUCKDB_TABLE_NAME, EMBEDDING_MODEL_NAME

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Claude model used for response generation
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Maximum tokens for a single Claude response
MAX_TOKENS = 1024

# Number of semantic search results to retrieve from ChromaDB
CHROMA_TOP_K = 5

# Maximum DuckDB results returned to the agent as context
DUCKDB_MAX_RESULTS = 10

# Supported language codes
SUPPORTED_LANGUAGES = ("fr", "en")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RecipeResult:
    """A single recipe result returned by a search or filter tool.

    Attributes:
        id: Unique recipe identifier (Umami URL slug).
        name: Recipe display name.
        url: Original Umami recipe URL.
        cuisine_tags: List of cuisine/type tags.
        total_time_minutes: Total cook + prep time in minutes (0 = unknown).
        source_category: "favorites" or "new".
        ingredients_clean: Clean ingredient list for display.
        score: Semantic similarity score (ChromaDB results only).
    """

    id: str
    name: str
    url: str
    cuisine_tags: list[str]
    total_time_minutes: int
    source_category: str
    ingredients_clean: str
    score: float = 0.0


@dataclass
class SearchFilters:
    """Optional filters for the MetadataFilterTool.

    All fields are optional — None means "no filter applied" (= Any).

    Attributes:
        max_time_minutes: Maximum total cook time in minutes.
        cuisine_tag: A single cuisine tag to filter on (case-insensitive).
        source_category: "favorites", "new", or None for both.
    """

    max_time_minutes: Optional[int] = None
    cuisine_tag: Optional[str] = None
    source_category: Optional[str] = None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# The system prompt is the core of the agent's behaviour.
# It instructs Claude to:
#   1. Ask clarifying questions before searching
#   2. Use the retrieved recipe context to justify suggestions
#   3. Respond in the user's chosen language
#   4. Be honest when no match is found

SYSTEM_PROMPT_TEMPLATE = """You are ChefRAG, a friendly and knowledgeable personal culinary assistant.
You help the user find recipes from their personal Umami recipe collection based on the ingredients they have available.

## Your behaviour

1. When the user tells you what ingredients they have, ALWAYS ask the following clarifying questions \
BEFORE searching for recipes (you can ask them all at once in a friendly way):
   - Preferred difficulty: Easy / Medium / Hard / Any?
   - Available cooking time: < 15 min / 30 min / 1 hour / Any?
   - Spice level: Mild / Medium / Spicy / Any?
   - Cuisine type preference: Asian / French / Italian / Mexican / Any?
   - Recipe category: Favorites only / New recipes only / Both?

2. Once you have the user's preferences, you will receive a list of matching recipes from their collection. \
Use ONLY these recipes to make suggestions — do not invent or suggest recipes that are not in the provided context.

3. For each recipe you suggest, ALWAYS explain:
   - Which of the user's available ingredients it uses
   - Why it matches their stated preferences (time, cuisine, etc.)
   - A link to the full recipe (use the URL provided)

4. If no recipes match, say so clearly and suggest which filters the user could relax to get results.

5. Keep your tone warm, concise and practical. Avoid unnecessary filler phrases.

## Language

Respond exclusively in {language}. All your messages must be in {language}, \
including clarifying questions, recipe suggestions, and error messages.
Recipe names and ingredient lists may remain in their original language.

## Context format

When recipes are available, they will be provided to you in this format:
---
RECIPE CONTEXT:
[list of matching recipes with their metadata]
---

Base your suggestions strictly on this context."""


def build_system_prompt(language: str) -> str:
    """Build the system prompt for the given language.

    Args:
        language: Language code ("fr" or "en").

    Returns:
        Formatted system prompt string.
    """
    lang_label = "French" if language == "fr" else "English"
    return SYSTEM_PROMPT_TEMPLATE.format(language=lang_label)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class RecipeSearchTool:
    """Semantic recipe search using ChromaDB vector embeddings.

    Uses the HuggingFace embedding model (BAAI/bge-small-en-v1.5) to embed
    the user's query and retrieve the most semantically similar recipes from
    ChromaDB. Searches both collections (favorites + new) by default.

    Attributes:
        chroma_client: Initialised ChromaDB HTTP client.
        embed_model: LlamaIndex HuggingFaceEmbedding instance.
        top_k: Number of results to return per collection.
    """

    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        embed_model: HuggingFaceEmbedding,
        top_k: int = CHROMA_TOP_K,
    ) -> None:
        """Initialise the RecipeSearchTool.

        Args:
            chroma_client: An initialised ChromaDB client.
            embed_model: A LlamaIndex HuggingFaceEmbedding instance.
            top_k: Number of results to retrieve per collection.
        """
        self.chroma_client = chroma_client
        self.embed_model = embed_model
        self.top_k = top_k
        self._logger = logging.getLogger(__name__)

    def search(
        self,
        query: str,
        source_category: Optional[str] = None,
    ) -> list[RecipeResult]:
        """Perform semantic search across ChromaDB recipe collections.

        Embeds the query and queries the ChromaDB collection(s). If
        source_category is None, both "favorites" and "new" collections
        are searched and results are merged and de-duplicated by recipe ID.

        Args:
            query: Natural language query (e.g. "pasta with eggs and cheese").
            source_category: Optional — "favorites", "new", or None for both.

        Returns:
            List of RecipeResult sorted by similarity score (highest first).
        """
        query_embedding = self.embed_model.get_text_embedding(query)

        # Determine which collections to query
        if source_category:
            categories = [source_category]
        else:
            categories = ["favorites", "new"]

        results: dict[str, RecipeResult] = {}

        for category in categories:
            collection_name = f"{CHROMA_COLLECTION_PREFIX}_{category}"
            try:
                collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                self._logger.warning(
                    "ChromaDB collection '%s' not found — skipping.", collection_name
                )
                continue

            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k,
                include=["metadatas", "distances"],
            )

            ids = response.get("ids", [[]])[0]
            metadatas = response.get("metadatas", [[]])[0]
            distances = response.get("distances", [[]])[0]

            for recipe_id, metadata, distance in zip(ids, metadatas, distances):
                # Convert ChromaDB distance to a similarity score (lower = better)
                score = 1.0 - distance

                # De-duplicate: keep the higher score if seen in both collections
                if recipe_id in results and results[recipe_id].score >= score:
                    continue

                cuisine_tags = [
                    t.strip()
                    for t in metadata.get("cuisine_tags", "").split(",")
                    if t.strip()
                ]

                results[recipe_id] = RecipeResult(
                    id=recipe_id,
                    name=metadata.get("name", ""),
                    url=metadata.get("url", ""),
                    cuisine_tags=cuisine_tags,
                    total_time_minutes=int(metadata.get("total_time_minutes", 0)),
                    source_category=metadata.get("source_category", ""),
                    ingredients_clean=metadata.get("ingredients_clean", ""),
                    score=score,
                )

        sorted_results = sorted(results.values(), key=lambda r: r.score, reverse=True)
        self._logger.info(
            "RecipeSearchTool: %d results for query '%s'.", len(sorted_results), query
        )
        return sorted_results


class MetadataFilterTool:
    """Structured recipe filter using DuckDB metadata.

    Queries the DuckDB `recipes` table with optional filters on:
    - Maximum total cooking time
    - Cuisine tag (partial match on the cuisine_tags column)
    - Source category (favorites / new)

    This tool complements RecipeSearchTool: semantic search finds
    thematically relevant recipes; this tool enforces hard constraints
    (e.g. "must be under 30 minutes").

    Attributes:
        duckdb_conn: An open DuckDB connection.
    """

    def __init__(self, duckdb_conn: duckdb.DuckDBPyConnection) -> None:
        """Initialise the MetadataFilterTool.

        Args:
            duckdb_conn: An open DuckDB connection to the chefrag database.
        """
        self.duckdb_conn = duckdb_conn
        self._logger = logging.getLogger(__name__)

    def filter(self, filters: SearchFilters) -> list[RecipeResult]:
        """Filter recipes from DuckDB based on structured criteria.

        Builds a dynamic WHERE clause from the non-None fields of SearchFilters.
        All filters are combined with AND.

        Args:
            filters: A SearchFilters dataclass with optional filter values.

        Returns:
            List of RecipeResult matching all applied filters.
        """
        conditions: list[str] = []
        params: list[object] = []

        if filters.max_time_minutes is not None:
            # Include recipes with unknown time (0) — they may still be valid
            conditions.append(
                "(total_time_minutes <= ? OR total_time_minutes = 0)"
            )
            params.append(filters.max_time_minutes)

        if filters.cuisine_tag:
            # Case-insensitive partial match on the cuisine_tags string
            conditions.append("LOWER(cuisine_tags) LIKE LOWER(?)")
            params.append(f"%{filters.cuisine_tag}%")

        if filters.source_category:
            conditions.append("source_category = ?")
            params.append(filters.source_category)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT
                id, name, url, cuisine_tags,
                total_time_minutes, source_category, ingredients_clean
            FROM {DUCKDB_TABLE_NAME}
            {where_clause}
            LIMIT {DUCKDB_MAX_RESULTS}
        """

        try:
            rows = self.duckdb_conn.execute(query, params).fetchall()
        except Exception as exc:
            self._logger.error("DuckDB filter query failed: %s", exc)
            return []

        results = [
            RecipeResult(
                id=row[0],
                name=row[1],
                url=row[2],
                cuisine_tags=[t.strip() for t in (row[3] or "").split(",") if t.strip()],
                total_time_minutes=row[4] or 0,
                source_category=row[5] or "",
                ingredients_clean=row[6] or "",
            )
            for row in rows
        ]

        self._logger.info(
            "MetadataFilterTool: %d results with filters %s.", len(results), filters
        )
        return results


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------


def format_recipe_context(recipes: list[RecipeResult]) -> str:
    """Format a list of RecipeResult into a text block for the Claude prompt.

    The formatted context is injected into the user message so Claude can
    base its suggestions on real data from the recipe collection.

    Args:
        recipes: List of RecipeResult to include in context.

    Returns:
        Formatted multi-line string, or a "no results" message if empty.
    """
    if not recipes:
        return "No matching recipes found in the collection."

    lines = ["RECIPE CONTEXT:", ""]
    for i, recipe in enumerate(recipes, start=1):
        time_str = (
            f"{recipe.total_time_minutes} min"
            if recipe.total_time_minutes > 0
            else "time not specified"
        )
        cuisine_str = ", ".join(recipe.cuisine_tags) if recipe.cuisine_tags else "not specified"
        lines += [
            f"{i}. {recipe.name}",
            f"   Category: {recipe.source_category}",
            f"   Cuisine: {cuisine_str}",
            f"   Total time: {time_str}",
            f"   Ingredients: {recipe.ingredients_clean}",
            f"   URL: {recipe.url}",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ChefRagAgent:
    """Conversational culinary assistant agent.

    Combines semantic search (RecipeSearchTool) and structured filtering
    (MetadataFilterTool) to retrieve relevant recipes, then calls the
    Anthropic Claude API directly to generate a natural language response.

    The agent is stateless — the full conversation history must be passed
    on every call. State is managed by main.py via Streamlit session state.

    Attributes:
        search_tool: RecipeSearchTool for ChromaDB semantic search.
        filter_tool: MetadataFilterTool for DuckDB structured filtering.
        anthropic_client: Anthropic SDK client instance.
    """

    def __init__(
        self,
        search_tool: RecipeSearchTool,
        filter_tool: MetadataFilterTool,
        anthropic_client: anthropic.Anthropic,
    ) -> None:
        """Initialise the ChefRagAgent.

        Args:
            search_tool: An initialised RecipeSearchTool.
            filter_tool: An initialised MetadataFilterTool.
            anthropic_client: An initialised Anthropic SDK client.
        """
        self.search_tool = search_tool
        self.filter_tool = filter_tool
        self.anthropic_client = anthropic_client
        self._logger = logging.getLogger(__name__)

    def _retrieve_recipes(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
    ) -> list[RecipeResult]:
        """Retrieve recipes using both tools and merge the results.

        Runs semantic search first, then applies metadata filters to the
        semantic results. If filters are provided but reduce the result set
        to zero, falls back to semantic-only results.

        Args:
            query: The user's ingredient/preference query string.
            filters: Optional structured filters to apply.

        Returns:
            Merged and de-duplicated list of RecipeResult.
        """
        # Step 1 — semantic search across all collections
        semantic_results = self.search_tool.search(
            query=query,
            source_category=filters.source_category if filters else None,
        )

        if not filters:
            return semantic_results

        # Step 2 — structured filter from DuckDB
        filtered_results = self.filter_tool.filter(filters)

        if not filtered_results:
            self._logger.info(
                "MetadataFilterTool returned 0 results — using semantic results only."
            )
            return semantic_results

        # Step 3 — intersect: keep semantic results that also appear in filtered set
        filtered_ids = {r.id for r in filtered_results}
        intersected = [r for r in semantic_results if r.id in filtered_ids]

        if not intersected:
            self._logger.info(
                "Intersection is empty — returning semantic results as fallback."
            )
            return semantic_results

        return intersected

    def chat(
        self,
        messages: list[dict[str, str]],
        language: str = "en",
        filters: Optional[SearchFilters] = None,
    ) -> str:
        """Generate a response to the latest user message.

        Takes the full conversation history, optionally retrieves matching
        recipes as RAG context, and calls the Claude API to generate a reply.

        The last message in `messages` must have role="user". If recipe
        context is available, it is injected into the last user message
        before being sent to Claude.

        Args:
            messages: Full conversation history as a list of
                {"role": "user"|"assistant", "content": str} dicts.
            language: Language code for the response ("fr" or "en").
            filters: Optional structured filters extracted from user preferences.

        Returns:
            The assistant's response as a plain string.

        Raises:
            ValueError: If messages is empty or the last message is not from the user.
        """
        if not messages:
            raise ValueError("messages must not be empty.")

        last_message = messages[-1]
        if last_message.get("role") != "user":
            raise ValueError("The last message must have role='user'.")

        if language not in SUPPORTED_LANGUAGES:
            self._logger.warning(
                "Unsupported language '%s' — falling back to 'en'.", language
            )
            language = "en"

        # Retrieve recipe context using the user's latest message as the query
        user_query = last_message["content"]
        recipes = self._retrieve_recipes(query=user_query, filters=filters)
        recipe_context = format_recipe_context(recipes)

        # Inject recipe context into the last user message
        augmented_content = f"{user_query}\n\n---\n{recipe_context}\n---"
        augmented_messages = messages[:-1] + [
            {"role": "user", "content": augmented_content}
        ]

        system_prompt = build_system_prompt(language)

        response = self.anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=augmented_messages,
        )

        reply = response.content[0].text
        self._logger.info(
            "Agent replied (%d chars) in language='%s'.", len(reply), language
        )
        return reply


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_agent(
    chroma_host: str,
    chroma_port: int,
    duckdb_path: str,
) -> ChefRagAgent:
    """Build and return a fully initialised ChefRagAgent.

    Initialises all dependencies: ChromaDB client, DuckDB connection,
    HuggingFace embedding model, Anthropic client, and both tools.

    The Anthropic API key is read from the ANTHROPIC_API_KEY environment
    variable — never hardcoded (security rule).

    Args:
        chroma_host: ChromaDB server hostname (e.g. "chefrag-chroma").
        chroma_port: ChromaDB server port (e.g. 8000).
        duckdb_path: Filesystem path to the .duckdb file.

    Returns:
        A fully initialised ChefRagAgent ready to receive chat() calls.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it in your .env file or Coolify environment."
        )

    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    duckdb_conn = duckdb.connect(duckdb_path)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    anthropic_client = anthropic.Anthropic(api_key=api_key)

    search_tool = RecipeSearchTool(chroma_client=chroma_client, embed_model=embed_model)
    filter_tool = MetadataFilterTool(duckdb_conn=duckdb_conn)

    return ChefRagAgent(
        search_tool=search_tool,
        filter_tool=filter_tool,
        anthropic_client=anthropic_client,
    )