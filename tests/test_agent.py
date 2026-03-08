"""
ChefRAG — Unit tests for app/agent.py.

All external dependencies are mocked:
- ChromaDB client and collections
- DuckDB connection
- HuggingFace embedding model
- Anthropic API client

No network calls, no file I/O, no embedding model required.
"""

from unittest.mock import MagicMock, patch, call
import pytest

from agent import (
    ChefRagAgent,
    MetadataFilterTool,
    RecipeResult,
    RecipeSearchTool,
    SearchFilters,
    build_system_prompt,
    format_recipe_context,
)


# ---------------------------------------------------------------------------
# Helpers — fixture factories
# ---------------------------------------------------------------------------


def make_recipe(
    id: str = "recipe-001",
    name: str = "Test Recipe",
    url: str = "https://umami.recipes/recipe/recipe-001",
    cuisine_tags: list[str] | None = None,
    total_time_minutes: int = 30,
    source_category: str = "favorites",
    ingredients_clean: str = "eggs, cheese, ham",
    score: float = 0.9,
) -> RecipeResult:
    """Create a RecipeResult with sensible defaults for tests."""
    return RecipeResult(
        id=id,
        name=name,
        url=url,
        cuisine_tags=cuisine_tags or ["French", "Main"],
        total_time_minutes=total_time_minutes,
        source_category=source_category,
        ingredients_clean=ingredients_clean,
        score=score,
    )


def make_chroma_response(
    ids: list[str],
    distances: list[float],
    metadatas: list[dict],
) -> dict:
    """Build a mock ChromaDB query response."""
    return {
        "ids": [ids],
        "distances": [distances],
        "metadatas": [metadatas],
    }


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Tests for the system prompt builder."""

    def test_english_prompt_contains_english(self) -> None:
        """English prompt should mention 'English'."""
        prompt = build_system_prompt("en")
        assert "English" in prompt

    def test_french_prompt_contains_french(self) -> None:
        """French prompt should mention 'French'."""
        prompt = build_system_prompt("fr")
        assert "French" in prompt

    def test_prompt_contains_clarifying_questions_instruction(self) -> None:
        """Prompt should instruct Claude to ask clarifying questions."""
        prompt = build_system_prompt("en")
        assert "clarifying questions" in prompt.lower()

    def test_prompt_contains_context_format(self) -> None:
        """Prompt should describe the RECIPE CONTEXT format."""
        prompt = build_system_prompt("en")
        assert "RECIPE CONTEXT" in prompt


# ---------------------------------------------------------------------------
# format_recipe_context
# ---------------------------------------------------------------------------


class TestFormatRecipeContext:
    """Tests for the recipe context formatter."""

    def test_empty_list_returns_no_results_message(self) -> None:
        """Empty recipe list should return a clear 'no results' string."""
        result = format_recipe_context([])
        assert "No matching recipes" in result

    def test_recipe_name_appears_in_context(self) -> None:
        """Recipe name should appear in the formatted context."""
        recipe = make_recipe(name="Omelette du Chef")
        result = format_recipe_context([recipe])
        assert "Omelette du Chef" in result

    def test_recipe_url_appears_in_context(self) -> None:
        """Recipe URL should appear in the formatted context."""
        recipe = make_recipe(url="https://umami.recipes/recipe/abc123")
        result = format_recipe_context([recipe])
        assert "https://umami.recipes/recipe/abc123" in result

    def test_time_formatted_as_minutes(self) -> None:
        """Cook time should appear as 'X min' in the context."""
        recipe = make_recipe(total_time_minutes=45)
        result = format_recipe_context([recipe])
        assert "45 min" in result

    def test_zero_time_shows_not_specified(self) -> None:
        """A zero total time should display as 'time not specified'."""
        recipe = make_recipe(total_time_minutes=0)
        result = format_recipe_context([recipe])
        assert "not specified" in result

    def test_multiple_recipes_numbered(self) -> None:
        """Multiple recipes should be numbered sequentially."""
        recipes = [make_recipe(id=f"r-{i}", name=f"Recipe {i}") for i in range(3)]
        result = format_recipe_context(recipes)
        assert "1." in result
        assert "2." in result
        assert "3." in result

    def test_ingredients_appear_in_context(self) -> None:
        """Ingredient list should appear in the formatted context."""
        recipe = make_recipe(ingredients_clean="quinoa, black beans, lime")
        result = format_recipe_context([recipe])
        assert "quinoa" in result


# ---------------------------------------------------------------------------
# RecipeSearchTool
# ---------------------------------------------------------------------------


class TestRecipeSearchTool:
    """Tests for the ChromaDB semantic search tool."""

    def _make_tool(self, chroma_response: dict) -> tuple[RecipeSearchTool, MagicMock]:
        """Build a RecipeSearchTool with a mocked ChromaDB client.

        Args:
            chroma_response: The dict to return from collection.query().

        Returns:
            Tuple of (tool, mock_collection).
        """
        mock_collection = MagicMock()
        mock_collection.query.return_value = chroma_response

        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value = mock_collection

        mock_embed = MagicMock()
        mock_embed.get_text_embedding.return_value = [0.1] * 384

        tool = RecipeSearchTool(
            chroma_client=mock_chroma,
            embed_model=mock_embed,
            top_k=5,
        )
        return tool, mock_collection

    def test_returns_recipe_results(self) -> None:
        """Search should return a list of RecipeResult."""
        response = make_chroma_response(
            ids=["r1"],
            distances=[0.2],
            metadatas=[{
                "name": "Zha Jiang Mian",
                "url": "https://umami.recipes/r/1",
                "cuisine_tags": "Korean, Asian",
                "total_time_minutes": 0,
                "source_category": "favorites",
                "ingredients_clean": "noodles, pork",
            }],
        )
        tool, _ = self._make_tool(response)
        results = tool.search("noodles with pork")
        assert len(results) == 1
        assert results[0].name == "Zha Jiang Mian"

    def test_score_derived_from_distance(self) -> None:
        """Score should be 1.0 - distance."""
        response = make_chroma_response(
            ids=["r1"],
            distances=[0.3],
            metadatas=[{"name": "R", "url": "", "cuisine_tags": "",
                        "total_time_minutes": 0, "source_category": "new",
                        "ingredients_clean": ""}],
        )
        tool, _ = self._make_tool(response)
        results = tool.search("anything")
        assert abs(results[0].score - 0.7) < 0.001

    def test_missing_collection_is_skipped(self) -> None:
        """If a ChromaDB collection does not exist, it should be skipped silently."""
        mock_chroma = MagicMock()
        mock_chroma.get_collection.side_effect = Exception("Collection not found")

        mock_embed = MagicMock()
        mock_embed.get_text_embedding.return_value = [0.1] * 384

        tool = RecipeSearchTool(chroma_client=mock_chroma, embed_model=mock_embed)
        results = tool.search("pasta")
        assert results == []

    def test_source_category_filter_queries_one_collection(self) -> None:
        """Passing source_category should query only that collection."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = make_chroma_response([], [], [])

        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value = mock_collection

        mock_embed = MagicMock()
        mock_embed.get_text_embedding.return_value = [0.1] * 384

        tool = RecipeSearchTool(chroma_client=mock_chroma, embed_model=mock_embed)
        tool.search("pasta", source_category="favorites")

        # get_collection should be called exactly once with "recipes_favorites"
        mock_chroma.get_collection.assert_called_once_with("recipes_favorites")

    def test_deduplication_keeps_higher_score(self) -> None:
        """If the same recipe appears in both collections, keep the higher score."""
        def get_collection(name: str) -> MagicMock:
            col = MagicMock()
            distance = 0.1 if "favorites" in name else 0.4
            col.query.return_value = make_chroma_response(
                ids=["shared-id"],
                distances=[distance],
                metadatas=[{"name": "Shared", "url": "", "cuisine_tags": "",
                             "total_time_minutes": 0, "source_category": name,
                             "ingredients_clean": ""}],
            )
            return col

        mock_chroma = MagicMock()
        mock_chroma.get_collection.side_effect = get_collection

        mock_embed = MagicMock()
        mock_embed.get_text_embedding.return_value = [0.1] * 384

        tool = RecipeSearchTool(chroma_client=mock_chroma, embed_model=mock_embed)
        results = tool.search("anything")

        assert len(results) == 1
        # Score from favorites: 1.0 - 0.1 = 0.9 (higher — should be kept)
        assert abs(results[0].score - 0.9) < 0.001


# ---------------------------------------------------------------------------
# MetadataFilterTool
# ---------------------------------------------------------------------------


class TestMetadataFilterTool:
    """Tests for the DuckDB structured filter tool."""

    def _make_tool_with_rows(self, rows: list[tuple]) -> MetadataFilterTool:
        """Build a MetadataFilterTool backed by a mocked DuckDB connection.

        Args:
            rows: List of tuples matching the SELECT column order:
                (id, name, url, cuisine_tags, total_time_minutes,
                 source_category, ingredients_clean)

        Returns:
            MetadataFilterTool with mocked connection.
        """
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = rows
        return MetadataFilterTool(duckdb_conn=mock_conn)

    def test_no_filters_returns_results(self) -> None:
        """With no filters, all rows should be returned as RecipeResult."""
        rows = [("r1", "Quinoa", "http://x", "Vegetarian, Main", 35, "new", "quinoa")]
        tool = self._make_tool_with_rows(rows)
        results = tool.filter(SearchFilters())
        assert len(results) == 1
        assert results[0].name == "Quinoa"

    def test_cuisine_tag_appears_in_result(self) -> None:
        """Cuisine tags should be correctly parsed from the comma-separated string."""
        rows = [("r1", "R", "", "Korean, Asian", 0, "favorites", "noodles")]
        tool = self._make_tool_with_rows(rows)
        results = tool.filter(SearchFilters(cuisine_tag="Korean"))
        assert "Korean" in results[0].cuisine_tags

    def test_empty_rows_returns_empty_list(self) -> None:
        """When DuckDB returns no rows, result should be an empty list."""
        tool = self._make_tool_with_rows([])
        results = tool.filter(SearchFilters(max_time_minutes=15))
        assert results == []

    def test_db_exception_returns_empty_list(self) -> None:
        """If the DuckDB query fails, filter should return [] without raising."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("DuckDB error")
        tool = MetadataFilterTool(duckdb_conn=mock_conn)
        results = tool.filter(SearchFilters())
        assert results == []

    def test_none_cuisine_tag_excluded_from_ingredients(self) -> None:
        """Cuisine tags column containing None should produce an empty list."""
        rows = [("r1", "R", "", None, 0, "new", "eggs")]
        tool = self._make_tool_with_rows(rows)
        results = tool.filter(SearchFilters())
        assert results[0].cuisine_tags == []


# ---------------------------------------------------------------------------
# ChefRagAgent
# ---------------------------------------------------------------------------


class TestChefRagAgent:
    """Tests for the main ChefRagAgent orchestrator."""

    def _make_agent(
        self,
        search_results: list[RecipeResult] | None = None,
        filter_results: list[RecipeResult] | None = None,
        claude_reply: str = "Here are some recipes for you!",
    ) -> ChefRagAgent:
        """Build a ChefRagAgent with all dependencies mocked.

        Args:
            search_results: Results returned by RecipeSearchTool.search().
            filter_results: Results returned by MetadataFilterTool.filter().
            claude_reply: Text returned by the mocked Anthropic API.

        Returns:
            A ChefRagAgent with fully mocked dependencies.
        """
        mock_search = MagicMock(spec=RecipeSearchTool)
        mock_search.search.return_value = search_results or []

        mock_filter = MagicMock(spec=MetadataFilterTool)
        mock_filter.filter.return_value = filter_results or []

        mock_content = MagicMock()
        mock_content.text = claude_reply
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_anthropic = MagicMock(spec=["messages"])
        mock_anthropic.messages = MagicMock()
        mock_anthropic.messages.create.return_value = mock_response

        return ChefRagAgent(
            search_tool=mock_search,
            filter_tool=mock_filter,
            anthropic_client=mock_anthropic,
        )

    def test_chat_returns_string(self) -> None:
        """chat() should return the Claude reply as a string."""
        agent = self._make_agent(claude_reply="Bonjour !")
        result = agent.chat([{"role": "user", "content": "I have eggs"}])
        assert result == "Bonjour !"

    def test_chat_calls_anthropic_with_system_prompt(self) -> None:
        """chat() should pass a system prompt to the Anthropic API."""
        agent = self._make_agent()
        agent.chat([{"role": "user", "content": "I have pasta"}], language="en")
        call_kwargs = agent.anthropic_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0

    def test_chat_injects_recipe_context_into_last_message(self) -> None:
        """Recipe context should be appended to the last user message."""
        recipe = make_recipe(name="Special Omelette")
        agent = self._make_agent(search_results=[recipe])
        agent.chat([{"role": "user", "content": "I have eggs and cheese"}])
        call_kwargs = agent.anthropic_client.messages.create.call_args.kwargs
        last_message = call_kwargs["messages"][-1]
        assert "Special Omelette" in last_message["content"]
        assert "RECIPE CONTEXT" in last_message["content"]

    def test_chat_preserves_conversation_history(self) -> None:
        """Prior messages should be preserved in the API call."""
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": "I have eggs"},
            {"role": "assistant", "content": "What cuisine do you prefer?"},
            {"role": "user", "content": "French please"},
        ]
        agent.chat(messages, language="en")
        call_kwargs = agent.anthropic_client.messages.create.call_args.kwargs
        sent_messages = call_kwargs["messages"]
        assert sent_messages[0]["content"] == "I have eggs"
        assert sent_messages[1]["content"] == "What cuisine do you prefer?"

    def test_chat_raises_on_empty_messages(self) -> None:
        """chat() should raise ValueError when messages is empty."""
        agent = self._make_agent()
        with pytest.raises(ValueError, match="must not be empty"):
            agent.chat([])

    def test_chat_raises_if_last_message_not_user(self) -> None:
        """chat() should raise ValueError if the last message role is not 'user'."""
        agent = self._make_agent()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        with pytest.raises(ValueError, match="role='user'"):
            agent.chat(messages)

    def test_unsupported_language_falls_back_to_english(self) -> None:
        """An unsupported language code should silently fall back to 'en'."""
        agent = self._make_agent()
        # Should not raise — just fall back
        agent.chat([{"role": "user", "content": "test"}], language="de")
        call_kwargs = agent.anthropic_client.messages.create.call_args.kwargs
        assert "English" in call_kwargs["system"]

    def test_no_filters_skips_filter_tool(self) -> None:
        """When no filters are passed, MetadataFilterTool should not be called."""
        agent = self._make_agent()
        agent.chat([{"role": "user", "content": "pasta"}], filters=None)
        agent.filter_tool.filter.assert_not_called()

    def test_with_filters_calls_both_tools(self) -> None:
        """When filters are passed, both search and filter tools should be called."""
        recipe = make_recipe()
        agent = self._make_agent(search_results=[recipe], filter_results=[recipe])
        filters = SearchFilters(max_time_minutes=30, cuisine_tag="French")
        agent.chat([{"role": "user", "content": "quick french food"}], filters=filters)
        agent.search_tool.search.assert_called_once()
        agent.filter_tool.filter.assert_called_once_with(filters)

    def test_intersection_returns_common_recipes(self) -> None:
        """Recipes in both semantic and filtered results should be returned."""
        shared = make_recipe(id="shared")
        semantic_only = make_recipe(id="semantic-only")
        agent = self._make_agent(
            search_results=[shared, semantic_only],
            filter_results=[shared],
        )
        filters = SearchFilters(max_time_minutes=60)
        agent.chat([{"role": "user", "content": "food"}], filters=filters)
        call_kwargs = agent.anthropic_client.messages.create.call_args.kwargs
        last_msg = call_kwargs["messages"][-1]["content"]
        # Only shared recipe should appear in context
        assert shared.name in last_msg

    def test_empty_intersection_falls_back_to_semantic(self) -> None:
        """If intersection is empty, fall back to semantic-only results."""
        semantic = make_recipe(id="sem", name="Semantic Recipe")
        filter_result = make_recipe(id="fil", name="Filter Recipe")
        agent = self._make_agent(
            search_results=[semantic],
            filter_results=[filter_result],
        )
        filters = SearchFilters(cuisine_tag="Italian")
        agent.chat([{"role": "user", "content": "italian food"}], filters=filters)
        call_kwargs = agent.anthropic_client.messages.create.call_args.kwargs
        last_msg = call_kwargs["messages"][-1]["content"]
        assert "Semantic Recipe" in last_msg


# ---------------------------------------------------------------------------
# build_agent (environment variable guard)
# ---------------------------------------------------------------------------


class TestBuildAgent:
    """Tests for the build_agent factory function."""

    def test_raises_if_api_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """build_agent should raise EnvironmentError if ANTHROPIC_API_KEY is not set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from agent import build_agent
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            build_agent(
                chroma_host="localhost",
                chroma_port=8000,
                duckdb_path=":memory:",
            )