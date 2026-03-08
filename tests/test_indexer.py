"""
ChefRAG — Unit tests for app/indexer.py.

Fixtures use the real Umami JSON exports provided during development:
- ZHA_JIANG_MIAN__NOODLES_WITH_MEAT_SAUCE_.json  (Favorites, zero times, bullet ingredients)
- One_Pan_Mexican_Quinoa.json                      (New, real times, clean ingredients)

Tests are fully offline — no ChromaDB, no DuckDB, no embedding model required.
ChromaDB and DuckDB interactions are tested with in-memory instances.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from app.indexer import (
    DUCKDB_TABLE_NAME,
    extract_clean_ingredients,
    extract_recipe_id,
    build_recipe_document,
    ensure_duckdb_table,
    index_recipe_in_duckdb,
    parse_cuisine_tags,
    parse_iso_duration,
    parse_recipe_json,
    Recipe,
)

# ---------------------------------------------------------------------------
# Paths to real fixture files
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_ZHA_JIANG = FIXTURES_DIR / "ZHA_JIANG_MIAN__NOODLES_WITH_MEAT_SAUCE_.json"
FIXTURE_QUINOA = FIXTURES_DIR / "One_Pan_Mexican_Quinoa.json"


# ---------------------------------------------------------------------------
# parse_iso_duration
# ---------------------------------------------------------------------------


class TestParseIsoDuration:
    """Tests for the ISO 8601 duration parser."""

    def test_zero_duration_returns_zero(self) -> None:
        """All-zero ISO string should return 0 minutes."""
        assert parse_iso_duration("P0Y0M0DT0H0M0S") == 0

    def test_empty_string_returns_zero(self) -> None:
        """Empty string should return 0 minutes."""
        assert parse_iso_duration("") == 0

    def test_hours_and_minutes(self) -> None:
        """1 hour 30 minutes should return 90."""
        assert parse_iso_duration("P0Y0M0DT1H30M0S") == 90

    def test_minutes_only(self) -> None:
        """25 minutes only."""
        assert parse_iso_duration("P0Y0M0DT0H25M0S") == 25

    def test_days_converted_to_minutes(self) -> None:
        """1 day should return 1440 minutes."""
        assert parse_iso_duration("P0Y0M1DT0H0M0S") == 1440

    def test_combined_days_hours_minutes(self) -> None:
        """1 day + 2 hours + 30 min = 1590."""
        assert parse_iso_duration("P0Y0M1DT2H30M0S") == 1590

    def test_invalid_string_returns_zero(self) -> None:
        """Non-ISO string should return 0 (with a warning logged)."""
        assert parse_iso_duration("not-a-duration") == 0

    def test_quinoa_prep_time(self) -> None:
        """Real prepTime from One Pan Mexican Quinoa: 10 minutes."""
        assert parse_iso_duration("P0Y0M0DT0H10M0S") == 10

    def test_quinoa_cook_time(self) -> None:
        """Real cookTime from One Pan Mexican Quinoa: 25 minutes."""
        assert parse_iso_duration("P0Y0M0DT0H25M0S") == 25

    def test_quinoa_total_time(self) -> None:
        """Real totalTime from One Pan Mexican Quinoa: 35 minutes."""
        assert parse_iso_duration("P0Y0M0DT0H35M0S") == 35


# ---------------------------------------------------------------------------
# extract_clean_ingredients
# ---------------------------------------------------------------------------


class TestExtractCleanIngredients:
    """Tests for the ingredient line filter."""

    def test_bullet_lines_are_kept(self) -> None:
        """Lines starting with • should be kept, bullet stripped."""
        raw = ["Section header", "•1 cup flour", "•2 eggs"]
        result = extract_clean_ingredients(raw)
        assert result == ["1 cup flour", "2 eggs"]

    def test_narrative_lines_are_dropped_when_bullets_present(self) -> None:
        """Non-bullet lines should be dropped when bullets exist in the list."""
        raw = [
            "Ultimate zha jiang sauce",
            "Over the years my mom has incorporated...",
            "•1/3 pound ground turkey",
            "•Cooking oil",
        ]
        result = extract_clean_ingredients(raw)
        assert "Ultimate zha jiang sauce" not in result
        assert "1/3 pound ground turkey" in result

    def test_empty_list_returns_empty(self) -> None:
        """Empty input should return empty list."""
        assert extract_clean_ingredients([]) == []

    def test_no_bullets_fallback_keeps_long_lines(self) -> None:
        """When no bullets are present, keep all lines above min length."""
        raw = ["1 cup quinoa", "2 cloves garlic", "ok"]
        result = extract_clean_ingredients(raw)
        assert "1 cup quinoa" in result
        assert "2 cloves garlic" in result

    def test_blank_lines_are_always_dropped(self) -> None:
        """Blank/whitespace-only lines should never appear in output."""
        raw = ["•1 cup flour", "   ", "", "•2 eggs"]
        result = extract_clean_ingredients(raw)
        assert "" not in result
        assert "   " not in result


# ---------------------------------------------------------------------------
# parse_cuisine_tags
# ---------------------------------------------------------------------------


class TestParseCuisineTags:
    """Tests for the cuisine tag extractor."""

    def test_comma_separated_tags(self) -> None:
        """Standard comma-separated description should split correctly."""
        tags = parse_cuisine_tags("Korean, Chinese, Main, Vegetarian-adaptable, Asian")
        assert tags == ["Korean", "Chinese", "Main", "Vegetarian-adaptable", "Asian"]

    def test_empty_string_returns_empty_list(self) -> None:
        """Empty description should return empty list."""
        assert parse_cuisine_tags("") == []

    def test_single_tag(self) -> None:
        """Single tag without comma should return single-item list."""
        assert parse_cuisine_tags("Vegetarian") == ["Vegetarian"]

    def test_whitespace_is_stripped(self) -> None:
        """Extra whitespace around tags should be removed."""
        tags = parse_cuisine_tags("  Korean ,  Asian  ")
        assert tags == ["Korean", "Asian"]


# ---------------------------------------------------------------------------
# extract_recipe_id
# ---------------------------------------------------------------------------


class TestExtractRecipeId:
    """Tests for the ID extractor."""

    def test_extracts_slug_from_url(self) -> None:
        """Should return the last URL path segment."""
        url = "https://www.umami.recipes/recipe/6GU8VBUycqn2GxzKq223"
        assert extract_recipe_id(url) == "6GU8VBUycqn2GxzKq223"

    def test_trailing_slash_handled(self) -> None:
        """Trailing slash should not break extraction."""
        url = "https://www.umami.recipes/recipe/nqaKsx9aKjdsYLh3PxEk/"
        assert extract_recipe_id(url) == "nqaKsx9aKjdsYLh3PxEk"

    def test_empty_url_returns_uuid(self) -> None:
        """Empty URL should fall back to a UUID string."""
        result = extract_recipe_id("")
        assert len(result) == 36  # UUID format


# ---------------------------------------------------------------------------
# parse_recipe_json — real fixture files
# ---------------------------------------------------------------------------


class TestParseRecipeJson:
    """Integration-level tests using the real Umami JSON fixture files."""

    def test_parse_zha_jiang_mian(self) -> None:
        """Zha Jiang Mian should parse correctly with zero times."""
        recipe = parse_recipe_json(FIXTURE_ZHA_JIANG, "favorites")

        assert recipe.name == "ZHA JIANG MIAN (NOODLES WITH MEAT SAUCE)"
        assert recipe.id == "6GU8VBUycqn2GxzKq223"
        assert recipe.source_category == "favorites"
        assert recipe.recipe_category == "Favorites"
        assert recipe.recipe_cuisine == "Korean"
        assert recipe.total_time_minutes == 0
        assert recipe.prep_time_minutes == 0
        assert recipe.cook_time_minutes == 0

    def test_zha_jiang_cuisine_tags(self) -> None:
        """Cuisine tags should be parsed from the description field."""
        recipe = parse_recipe_json(FIXTURE_ZHA_JIANG, "favorites")
        assert "Korean" in recipe.cuisine_tags
        assert "Asian" in recipe.cuisine_tags
        assert "Main" in recipe.cuisine_tags

    def test_zha_jiang_clean_ingredients_excludes_narrative(self) -> None:
        """Narrative lines (no bullet) should not appear in ingredients_clean."""
        recipe = parse_recipe_json(FIXTURE_ZHA_JIANG, "favorites")
        for ingredient in recipe.ingredients_clean:
            assert not ingredient.startswith("Over the years")
            assert not ingredient.startswith("Ultimate")
            assert not ingredient.startswith("Simplified")
            assert not ingredient.startswith("If you don")

    def test_zha_jiang_instructions_excludes_preparation_header(self) -> None:
        """The 'Preparation' step should be excluded from instructions."""
        recipe = parse_recipe_json(FIXTURE_ZHA_JIANG, "favorites")
        assert "Preparation" not in recipe.instructions

    def test_parse_quinoa(self) -> None:
        """One Pan Mexican Quinoa should parse with correct times."""
        recipe = parse_recipe_json(FIXTURE_QUINOA, "new")

        assert recipe.name == "One Pan Mexican Quinoa"
        assert recipe.id == "nqaKsx9aKjdsYLh3PxEk"
        assert recipe.source_category == "new"
        assert recipe.recipe_category == "New"
        assert recipe.prep_time_minutes == 10
        assert recipe.cook_time_minutes == 25
        assert recipe.total_time_minutes == 35
        assert recipe.recipe_yield == "4 servings"

    def test_quinoa_cuisine_tags(self) -> None:
        """Quinoa cuisine tags from description: Vegetarian, Main."""
        recipe = parse_recipe_json(FIXTURE_QUINOA, "new")
        assert "Vegetarian" in recipe.cuisine_tags
        assert "Main" in recipe.cuisine_tags

    def test_quinoa_ingredients_clean(self) -> None:
        """Quinoa has no bullet format — all non-trivial lines should be kept."""
        recipe = parse_recipe_json(FIXTURE_QUINOA, "new")
        assert len(recipe.ingredients_clean) > 5
        assert any("quinoa" in i.lower() for i in recipe.ingredients_clean)

    def test_invalid_type_raises_value_error(self, tmp_path: Path) -> None:
        """A JSON file with wrong @type should raise ValueError."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps({"@type": "Person", "name": "John"}))
        with pytest.raises(ValueError, match="not a Schema.org Recipe"):
            parse_recipe_json(bad_file, "favorites")

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """A non-existent file path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_recipe_json(tmp_path / "ghost.json", "favorites")

    def test_empty_name_raises_validation_error(self, tmp_path: Path) -> None:
        """A recipe with an empty name should fail Pydantic validation."""
        bad_file = tmp_path / "noname.json"
        bad_file.write_text(json.dumps({"@type": "Recipe", "name": "   "}))
        with pytest.raises(Exception):
            parse_recipe_json(bad_file, "favorites")


# ---------------------------------------------------------------------------
# build_recipe_document
# ---------------------------------------------------------------------------


class TestBuildRecipeDocument:
    """Tests for the embedding document builder."""

    def test_document_contains_recipe_name(self) -> None:
        """The output document should include the recipe name."""
        recipe = parse_recipe_json(FIXTURE_QUINOA, "new")
        doc = build_recipe_document(recipe)
        assert "One Pan Mexican Quinoa" in doc

    def test_document_contains_cuisine_tags(self) -> None:
        """The output document should mention cuisine tags."""
        recipe = parse_recipe_json(FIXTURE_QUINOA, "new")
        doc = build_recipe_document(recipe)
        assert "Vegetarian" in doc

    def test_document_contains_ingredients(self) -> None:
        """The output document should include ingredient text."""
        recipe = parse_recipe_json(FIXTURE_QUINOA, "new")
        doc = build_recipe_document(recipe)
        assert "quinoa" in doc.lower()


# ---------------------------------------------------------------------------
# DuckDB writer (in-memory)
# ---------------------------------------------------------------------------


class TestDuckDBIndexing:
    """Tests for the DuckDB metadata writer using an in-memory database."""

    def get_conn(self) -> duckdb.DuckDBPyConnection:
        """Create a fresh in-memory DuckDB connection for each test."""
        return duckdb.connect(":memory:")

    def test_ensure_table_creates_table(self) -> None:
        """ensure_duckdb_table should create the recipes table."""
        conn = self.get_conn()
        ensure_duckdb_table(conn)
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        assert DUCKDB_TABLE_NAME in table_names

    def test_ensure_table_is_idempotent(self) -> None:
        """Calling ensure_duckdb_table twice should not raise an error."""
        conn = self.get_conn()
        ensure_duckdb_table(conn)
        ensure_duckdb_table(conn)  # Should not raise

    def test_index_recipe_inserts_row(self) -> None:
        """index_recipe_in_duckdb should insert a row for the recipe."""
        conn = self.get_conn()
        recipe = parse_recipe_json(FIXTURE_QUINOA, "new")
        index_recipe_in_duckdb(recipe, conn)

        rows = conn.execute(
            f"SELECT id, name FROM {DUCKDB_TABLE_NAME} WHERE id = ?",
            [recipe.id],
        ).fetchall()

        assert len(rows) == 1
        assert rows[0][0] == recipe.id
        assert rows[0][1] == recipe.name

    def test_index_recipe_upsert_is_idempotent(self) -> None:
        """Indexing the same recipe twice should not create a duplicate row."""
        conn = self.get_conn()
        recipe = parse_recipe_json(FIXTURE_QUINOA, "new")
        index_recipe_in_duckdb(recipe, conn)
        index_recipe_in_duckdb(recipe, conn)

        count = conn.execute(
            f"SELECT COUNT(*) FROM {DUCKDB_TABLE_NAME} WHERE id = ?",
            [recipe.id],
        ).fetchone()[0]

        assert count == 1

    def test_index_stores_correct_times(self) -> None:
        """Cook/prep/total times should be correctly stored in DuckDB."""
        conn = self.get_conn()
        recipe = parse_recipe_json(FIXTURE_QUINOA, "new")
        index_recipe_in_duckdb(recipe, conn)

        row = conn.execute(
            f"""SELECT prep_time_minutes, cook_time_minutes, total_time_minutes
                FROM {DUCKDB_TABLE_NAME} WHERE id = ?""",
            [recipe.id],
        ).fetchone()

        assert row == (10, 25, 35)

    def test_index_zha_jiang_stores_zero_times(self) -> None:
        """Zha Jiang Mian has no times set — DuckDB should store zeros."""
        conn = self.get_conn()
        recipe = parse_recipe_json(FIXTURE_ZHA_JIANG, "favorites")
        index_recipe_in_duckdb(recipe, conn)

        row = conn.execute(
            f"""SELECT prep_time_minutes, cook_time_minutes, total_time_minutes
                FROM {DUCKDB_TABLE_NAME} WHERE id = ?""",
            [recipe.id],
        ).fetchone()

        assert row == (0, 0, 0)