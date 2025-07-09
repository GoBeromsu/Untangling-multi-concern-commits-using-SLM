from typing import Dict, List, Any

# API Configuration Constants
DEFAULT_LMSTUDIO_URL: str = "http://localhost:1234"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 1000
CONNECTION_TIMEOUT_SECONDS: int = 5

# UI Configuration Constants
CODE_DIFF_INPUT_HEIGHT: int = 300
SYSTEM_PROMPT_INPUT_HEIGHT: int = 200
RECENT_RESULTS_DISPLAY_LIMIT: int = 15

# Commit types for classification
COMMIT_TYPES: List[str] = [
    "docs",
    "test",
    "cicd",
    "build",
    "refactor",
    "feat",
    "fix",
]

# Base response schema for commit classification
RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": COMMIT_TYPES,
            },
        },
        "count": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["types", "count", "reason"],
    "additionalProperties": False,
}


def create_structured_output_format(
    schema_name: str = "commit_classification_response",
    schema: Dict[str, Any] = RESPONSE_SCHEMA,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Create structured output format for API calls.

    Args:
        schema_name: Name for the JSON schema
        schema: The JSON schema definition
        strict: Whether to use strict mode

    Returns:
        Structured output format dict
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": strict,
        },
    }


# Pre-configured structured output formats for different providers
OPENAI_STRUCTURED_OUTPUT_FORMAT: Dict[str, Any] = create_structured_output_format()
LMSTUDIO_STRUCTURED_OUTPUT_FORMAT: Dict[str, Any] = create_structured_output_format()

# Legacy alias for backwards compatibility
STRUCTURED_OUTPUT_FORMAT: Dict[str, Any] = OPENAI_STRUCTURED_OUTPUT_FORMAT
