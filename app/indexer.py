"""
ChefRAG — Indexer module.

Responsible for:
- Parsing Umami JSON exports (Schema.org format) into validated Recipe objects
- Generating embeddings and storing them in ChromaDB
- Writing structured metadata to DuckDB
- Running the full indexing pipeline for one recipe category
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import chromadb
import duckdb
from pydantic import BaseModel, Field, field_validator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HuggingFace embedding model — small, fast, multilingual-friendly
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ChromaDB collection name pattern: recipes_{category}
CHROMA_COLLECTION_PREFIX = "recipes"

# DuckDB table name for recipe metadata
DUCKDB_TABLE_NAME = "recipes"

# ISO 8601 duration regex — matches P0Y0M0DT1H30M0S style
ISO_DURATION_PATTERN = re.compile(
    r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?"
    r"T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
)

# Ingredient lines that start with • are actual ingredients (vs narrative text)
INGREDIENT_BULLET = "•"

# Minimum length for a line to be considered a real ingredient
INGREDIENT_MIN_LENGTH = 3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class Recipe(BaseModel):
    """Validated recipe parsed from a Schema.org Umami JSON export.

    Attributes:
        id: Unique identifier derived from the recipe URL slug.
        name: Recipe display name.
        description: Raw description string from Umami (contains cuisine tags).
        url: Original Umami recipe URL.
        prep_time_minutes: Preparation time in minutes (0 if not set).
        cook_time_minutes: Cook time in minutes (0 if not set).
        total_time_minutes: Total time in minutes (0 if not set).
        recipe_yield: Number of servings as a string (may be empty).
        recipe_category: Umami category label ("Favorites" or "New").
        recipe_cuisine: Primary cuisine tag from Umami.
        ingredients_raw: All ingredient lines as exported (including narrative).
        ingredients_clean: Only bullet-point lines — actual ingredients.
        instructions: Step-by-step instruction texts.
        keywords: Comma-separated keyword string from Umami.
        source_category: Folder-derived category slug ("favorites" or "new").
        cuisine_tags: Cuisine/type tags parsed from the description field.
    """

    id: str
    name: str
    description: str = ""
    url: str = ""
    prep_time_minutes: int = Field(default=0, ge=0)
    cook_time_minutes: int = Field(default=0, ge=0)
    total_time_minutes: int = Field(default=0, ge=0)
    recipe_yield: str = ""
    recipe_category: str = ""
    recipe_cuisine: str = ""
    ingredients_raw: list[str] = Field(default_factory=list)
    ingredients_clean: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    keywords: str = ""
    source_category: str = ""
    cuisine_tags: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        """Ensure the recipe name is not blank.

        Args:
            value: The name string to validate.

        Returns:
            The stripped name string.

        Raises:
            ValueError: If the name is empty after stripping.
        """
        if not value.strip():
            raise ValueError("Recipe name must not be empty.")
        return value.strip()


# ---------------------------------------------------------------------------
# Duration parser
# ---------------------------------------------------------------------------


def parse_iso_duration(duration_str: str) -> int:
    """Convert an ISO 8601 duration string to a total number of minutes.

    Handles the Umami export format: P0Y0M0DT1H30M0S.
    Returns 0 for null, empty, or all-zero durations.

    Args:
        duration_str: ISO 8601 duration string (e.g. "P0Y0M0DT1H30M0S").

    Returns:
        Total duration in minutes as an integer.
    """
    if not duration_str:
        return 0

    match = ISO_DURATION_PATTERN.fullmatch(duration_str.strip())
    if not match:
        logger.warning("Could not parse ISO duration: %s", duration_str)
        return 0

    # Groups: years, months, days, hours, minutes, seconds
    _years, _months, days, hours, minutes, _seconds = (
        int(g) if g else 0 for g in match.groups()
    )

    # Years and months are ignored — not meaningful for cook times
    total_minutes = (days * 24 * 60) + (hours * 60) + minutes
    return total_minutes


# ---------------------------------------------------------------------------
# Ingredient cleaner
# ---------------------------------------------------------------------------


def extract_clean_ingredients(raw_ingredients: list[str]) -> list[str]:
    """Filter ingredient lines to keep only real ingredient entries.

    Umami exports mix narrative text (section headers, notes) with actual
    ingredient lines. Real ingredients start with a bullet character (•).
    Lines without a bullet but longer than INGREDIENT_MIN_LENGTH are kept
    as a fallback for exports that don't use bullets.

    Args:
        raw_ingredients: Full list of ingredient strings from the JSON export.

    Returns:
        Filtered list containing only actual ingredient lines, stripped of
        the leading bullet character.
    """
    clean: list[str] = []

    has_bullets = any(line.startswith(INGREDIENT_BULLET) for line in raw_ingredients)

    for line in raw_ingredients:
        stripped = line.strip()
        if not stripped:
            continue

        if has_bullets:
            # Only keep bullet lines — skip narrative/header lines
            if stripped.startswith(INGREDIENT_BULLET):
                clean.append(stripped.lstrip(INGREDIENT_BULLET).strip())
        else:
            # Fallback: keep any non-trivially-short line
            if len(stripped) >= INGREDIENT_MIN_LENGTH:
                clean.append(stripped)

    return clean


# ---------------------------------------------------------------------------
# Tag parser
# ---------------------------------------------------------------------------


def parse_cuisine_tags(description: str) -> list[str]:
    """Extract cuisine/type tags from the Umami description field.

    Umami stores tags as a comma-separated string in the description field,
    e.g. "Korean, Chinese, Main, Vegetarian-adaptable, Asian".

    Args:
        description: Raw description string from the JSON export.

    Returns:
        List of stripped, non-empty tag strings.
    """
    if not description:
        return []
    return [tag.strip() for tag in description.split(",") if tag.strip()]


# ---------------------------------------------------------------------------
# ID generator
# ---------------------------------------------------------------------------


def extract_recipe_id(url: str) -> str:
    """Derive a stable recipe ID from its Umami URL slug.

    Uses the last path segment of the URL. Falls back to a UUID if the URL
    is missing or malformed.

    Args:
        url: Full Umami recipe URL.

    Returns:
        A short string identifier for the recipe.
    """
    if url:
        slug = url.rstrip("/").split("/")[-1]
        if slug:
            return slug
    logger.warning("Could not extract ID from URL '%s', generating UUID.", url)
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------


def parse_recipe_json(filepath: Path, source_category: str) -> Recipe:
    """Parse a single Umami JSON export file into a validated Recipe object.

    Args:
        filepath: Path to the .json file to parse.
        source_category: Category slug derived from the folder name
            ("favorites" or "new").

    Returns:
        A validated Recipe instance.

    Raises:
        ValueError: If the file is not a valid Recipe JSON or fails validation.
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Recipe file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Guard: must be a Schema.org Recipe
    if data.get("@type") != "Recipe":
        raise ValueError(
            f"File {filepath.name} is not a Schema.org Recipe "
            f"(found @type={data.get('@type')!r})."
        )

    raw_ingredients: list[str] = data.get("recipeIngredient", [])
    raw_instructions: list[dict] = data.get("recipeInstructions", [])
    description: str = data.get("description", "")
    url: str = data.get("url", "")

    # Extract instruction texts — skip steps with only "Preparation" label
    instructions: list[str] = [
        step["text"]
        for step in raw_instructions
        if isinstance(step, dict) and step.get("text", "").strip().lower() != "preparation"
    ]

    recipe = Recipe(
        id=extract_recipe_id(url),
        name=data.get("name", ""),
        description=description,
        url=url,
        prep_time_minutes=parse_iso_duration(data.get("prepTime", "")),
        cook_time_minutes=parse_iso_duration(data.get("cookTime", "")),
        total_time_minutes=parse_iso_duration(data.get("totalTime", "")),
        recipe_yield=data.get("recipeYield", ""),
        recipe_category=data.get("recipeCategory", ""),
        recipe_cuisine=data.get("recipeCuisine", ""),
        ingredients_raw=raw_ingredients,
        ingredients_clean=extract_clean_ingredients(raw_ingredients),
        instructions=instructions,
        keywords=data.get("keywords", ""),
        source_category=source_category,
        cuisine_tags=parse_cuisine_tags(description),
    )

    logger.info("Parsed recipe: %s (id=%s)", recipe.name, recipe.id)
    return recipe


# ---------------------------------------------------------------------------
# Embedding builder
# ---------------------------------------------------------------------------


def build_recipe_document(recipe: Recipe) -> str:
    """Build the text document that will be embedded and stored in ChromaDB.

    Combines the most semantically rich fields into a single string so that
    vector search can match on ingredients, cuisine type, and instructions.

    The full ingredients_raw list (including narrative context) is used here
    intentionally — narrative text improves semantic coverage for edge cases
    like "noodles with fermented black bean paste".

    Args:
        recipe: A validated Recipe object.

    Returns:
        A single multi-line string ready for embedding.
    """
    sections = [
        f"Recipe: {recipe.name}",
        f"Cuisine: {', '.join(recipe.cuisine_tags) or recipe.recipe_cuisine}",
        f"Ingredients: {', '.join(recipe.ingredients_raw)}",
        f"Instructions: {' '.join(recipe.instructions)}",
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# ChromaDB writer
# ---------------------------------------------------------------------------


def index_recipe_in_chroma(
    recipe: Recipe,
    chroma_client: chromadb.ClientAPI,
    embed_model: HuggingFaceEmbedding,
) -> None:
    """Embed a recipe and upsert it into the correct ChromaDB collection.

    The collection name is derived from source_category:
    "favorites" → "recipes_favorites", "new" → "recipes_new".

    Upserting is used so re-indexing is idempotent — running the pipeline
    twice on the same file does not create duplicates.

    Args:
        recipe: A validated Recipe object.
        chroma_client: An initialised ChromaDB client.
        embed_model: A LlamaIndex HuggingFaceEmbedding instance.
    """
    collection_name = f"{CHROMA_COLLECTION_PREFIX}_{recipe.source_category}"
    collection = chroma_client.get_or_create_collection(name=collection_name)

    document_text = build_recipe_document(recipe)

    # HuggingFaceEmbedding returns a list of floats
    embedding: list[float] = embed_model.get_text_embedding(document_text)

    # Metadata stored alongside the vector — used for display and filtering
    metadata = {
        "name": recipe.name,
        "url": recipe.url,
        "recipe_category": recipe.recipe_category,
        "source_category": recipe.source_category,
        "recipe_cuisine": recipe.recipe_cuisine,
        "cuisine_tags": ", ".join(recipe.cuisine_tags),
        "total_time_minutes": recipe.total_time_minutes,
        "prep_time_minutes": recipe.prep_time_minutes,
        "cook_time_minutes": recipe.cook_time_minutes,
        "keywords": recipe.keywords,
    }

    collection.upsert(
        ids=[recipe.id],
        embeddings=[embedding],
        documents=[document_text],
        metadatas=[metadata],
    )

    logger.info(
        "Upserted recipe '%s' into ChromaDB collection '%s'.",
        recipe.name,
        collection_name,
    )


# ---------------------------------------------------------------------------
# DuckDB writer
# ---------------------------------------------------------------------------


def ensure_duckdb_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the recipes metadata table in DuckDB if it does not exist.

    Args:
        conn: An open DuckDB connection.
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {DUCKDB_TABLE_NAME} (
            id                  VARCHAR PRIMARY KEY,
            name                VARCHAR NOT NULL,
            url                 VARCHAR,
            source_category     VARCHAR,
            recipe_category     VARCHAR,
            recipe_cuisine      VARCHAR,
            cuisine_tags        VARCHAR,
            prep_time_minutes   INTEGER,
            cook_time_minutes   INTEGER,
            total_time_minutes  INTEGER,
            recipe_yield        VARCHAR,
            keywords            VARCHAR,
            ingredients_clean   VARCHAR,
            description         VARCHAR
        )
    """)


def index_recipe_in_duckdb(
    recipe: Recipe,
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Upsert a recipe's structured metadata into the DuckDB recipes table.

    Uses INSERT OR REPLACE to keep the operation idempotent.

    Args:
        recipe: A validated Recipe object.
        conn: An open DuckDB connection.
    """
    ensure_duckdb_table(conn)

    conn.execute(
        f"""
        INSERT OR REPLACE INTO {DUCKDB_TABLE_NAME} VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            recipe.id,
            recipe.name,
            recipe.url,
            recipe.source_category,
            recipe.recipe_category,
            recipe.recipe_cuisine,
            ", ".join(recipe.cuisine_tags),
            recipe.prep_time_minutes,
            recipe.cook_time_minutes,
            recipe.total_time_minutes,
            recipe.recipe_yield,
            recipe.keywords,
            ", ".join(recipe.ingredients_clean),
            recipe.description,
        ],
    )

    logger.info("Upserted recipe '%s' into DuckDB.", recipe.name)


# ---------------------------------------------------------------------------
# Single recipe indexer
# ---------------------------------------------------------------------------


def index_recipe(
    recipe: Recipe,
    chroma_client: chromadb.ClientAPI,
    duckdb_conn: duckdb.DuckDBPyConnection,
    embed_model: HuggingFaceEmbedding,
) -> None:
    """Index a single recipe into both ChromaDB and DuckDB.

    This is the main integration point called by run_indexing() and by
    the Airflow DAG in Step 6.

    Args:
        recipe: A validated Recipe object.
        chroma_client: An initialised ChromaDB client.
        duckdb_conn: An open DuckDB connection.
        embed_model: A LlamaIndex HuggingFaceEmbedding instance.
    """
    index_recipe_in_chroma(recipe, chroma_client, embed_model)
    index_recipe_in_duckdb(recipe, duckdb_conn)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_indexing(
    data_dir: Path,
    source_category: str,
    chroma_host: str,
    chroma_port: int,
    duckdb_path: Path,
) -> None:
    """Run the full indexing pipeline for one recipe category.

    Scans data_dir for all .json files, parses each one, and indexes it
    into ChromaDB and DuckDB. Skips files that fail validation with a
    warning instead of crashing the whole pipeline.

    Args:
        data_dir: Path to the folder containing JSON exports
            (e.g. data/favorites/).
        source_category: Category slug ("favorites" or "new").
        chroma_host: ChromaDB server hostname.
        chroma_port: ChromaDB server port.
        duckdb_path: Path to the .duckdb file.
    """
    logger.info(
        "Starting indexing pipeline for category '%s' in %s.",
        source_category,
        data_dir,
    )

    # Initialise clients
    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    duckdb_conn = duckdb.connect(str(duckdb_path))
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

    json_files = sorted(data_dir.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s.", data_dir)
        return

    success_count = 0
    error_count = 0

    for filepath in json_files:
        try:
            recipe = parse_recipe_json(filepath, source_category)
            index_recipe(recipe, chroma_client, duckdb_conn, embed_model)
            success_count += 1
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("Skipping %s — parse error: %s", filepath.name, exc)
            error_count += 1

    duckdb_conn.close()

    logger.info(
        "Indexing complete for '%s': %d indexed, %d skipped.",
        source_category,
        success_count,
        error_count,
    )