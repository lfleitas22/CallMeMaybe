"""
Pydantic models for function definitions, test prompts, and output results.
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional


class ParameterDef(BaseModel):
    """Definition of a single function parameter."""
    type: str


class ReturnDef(BaseModel):
    """Definition of a function's return type."""
    type: str


class FunctionDefinition(BaseModel):
    """A function available for calling."""
    name: str
    description: str
    parameters: Dict[str, ParameterDef] = Field(default_factory=dict)
    returns: Optional[ReturnDef] = None


class TestPrompt(BaseModel):
    """A natural language prompt to be converted into a function call."""
    prompt: str
