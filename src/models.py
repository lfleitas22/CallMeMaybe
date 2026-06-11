from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class ParameterDef(BaseModel):
    """Definition of a single parameter within a function."""
    type: str


class ReturnDef(BaseModel):
    """Definition of a function's return type."""
    type: str


class FunctionDefinition(BaseModel):
    """Represents a single available function from functions_definition.json.
    """
    name: str
    description: str
    parameters: Dict[str, ParameterDef] = Field(default_factory=dict)
    returns: Optional[ReturnDef] = None


class TestPrompt(BaseModel):
    """Represents an incoming prompt from function_calling_tests.json."""
    prompt: str


class FunctionCallResult(BaseModel):
    """The required output structure for function_calling_results.json."""
    prompt: str
    name: str
    parameters: Dict[str, Any]
