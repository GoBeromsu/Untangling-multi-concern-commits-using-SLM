from typing import Dict, List, Any

# Commit types for classification
COMMIT_TYPES: List[str] = [
    "docs",
    "test",
    "cicd",
    "build",
    "style",
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
