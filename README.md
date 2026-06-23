*This project has been created as part of the 42 curriculum by lfleitas.*

# Call Me Maybe - Function Calling in LLMs

## Description

This project implements a function calling system that translates natural
language prompts into structured function calls using a small language model
(Qwen3-0.6B). The system uses **constrained decoding** to guarantee 100%
valid JSON output, ensuring near-perfect reliability even with a small 0.6B
parameter model.

Given a natural language prompt like "What is the sum of 40 and 2?", the
system generates:
- Function name: `fn_add_numbers`
- Arguments: `{"a": 40, "b": 2}`

The key innovation is using constrained decoding to guide the model's output
token-by-token, ensuring both syntactic JSON validity and semantic schema
compliance.

## Instructions

### Installation

1. Ensure you have Python 3.13 or later installed
2. Install dependencies using uv:
```bash
make install
```

Or manually:
```bash
uv sync
```

### Running the Program

Run with default paths:
```bash
make run
```

Or with custom paths:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Development Commands

- **Lint code**: `make lint` (runs flake8 and mypy)
- **Strict lint**: `make lint-strict` (runs mypy with --strict)
- **Debug mode**: `make debug` (runs with pdb)
- **Clean cache**: `make clean` (removes __pycache__, .mypy_cache, etc.)

## Algorithm Explanation

### Constrained Decoding Approach

The core algorithm implements constrained decoding through these steps:

1. **Prompt Construction**: Build a system prompt that includes all
   available function definitions and the user's natural language request.

2. **Forced Prefix**: Start generation with `{"name": "` to ensure valid
   JSON structure from the beginning.

3. **Token-by-Token Generation**:
   - Get logits from the LLM for the next token
   - Parse the current partial JSON to determine state
   - Filter valid tokens based on:
     - Current JSON parsing state (expecting key, value, comma, etc.)
     - Schema constraints (parameter types from function definitions)
     - Structural validity (proper JSON syntax)
   - Mask invalid tokens by setting their logits to -∞
   - Select the highest probability valid token (greedy decoding)

4. **State Tracking**: Use an incremental JSON parser that tracks:
   - Current parsing state (in key, in value, expecting colon, etc.)
   - Nested object context (are we inside "parameters"?)
   - Completed key-value pairs
   - Partial key/value being typed

5. **Schema Enforcement**: Once the function name is determined, enforce
   parameter types:
   - For `"number"` type: only allow digit tokens or `-`
   - For `"string"` type: only allow string tokens within quotes
   - For `"boolean"` type: only allow `true` or `false` prefixes

6. **Termination**: Stop when a complete JSON object is formed (matching
   opening and closing braces).

### Key Design Decisions

1. **Vocabulary-Based Validation**: Load the tokenizer's vocabulary file to
   map between token IDs and their string representations. This allows
   precise control over which tokens are valid at each step.

2. **Incremental Parsing**: Implement a custom JSON parser that can handle
   partial JSON strings and determine the current state. This is crucial for
   knowing what tokens are structurally valid.

3. **Two-Level Validation**: Validate tokens at both the structural level
   (valid JSON syntax) and semantic level (matches function schema).

4. **Greedy Decoding**: Use greedy selection (argmax) rather than sampling
   for deterministic, reliable output.

5. **Fractional Digit Limiting**: Prevent infinite loops in number
   generation by limiting fractional digits to 10 places.

## Performance Analysis

### Accuracy
- **Function Selection**: Near 100% accuracy when function descriptions
  clearly match the prompt
- **Parameter Extraction**: High accuracy for simple types (numbers,
  strings)
- **JSON Validity**: 100% - constrained decoding guarantees valid JSON

### Speed
- Processing time depends on:
  - Prompt length and complexity
  - Number of available functions
  - Parameter complexity
- Typical processing: 2-10 seconds per prompt on standard hardware

### Reliability
- **No JSON parsing errors**: Constrained decoding ensures 100% valid JSON
- **Schema compliance**: All outputs match the expected schema
- **Graceful error handling**: Malformed inputs are caught and reported

## Challenges Faced

1. **Token Boundary Issues**: Tokenizers split text in non-obvious ways
   (e.g., including leading spaces). Solution: Clean tokens by replacing
   special characters like `Ġ` (space) and `Ċ` (newline).

2. **State Machine Complexity**: Tracking JSON parsing state while
   considering nested objects required careful state management. Solution:
   Implemented a stack-based parser with explicit state transitions.

3. **Number Generation Loops**: Without limits, the model could generate
   infinite fractional digits. Solution: Limit fractional digits to 10
   places.

4. **Function Name Extraction**: Need to know which function was selected
   before validating parameters. Solution: Track completed key-value pairs
   and extract function name once the "name" field is complete.

5. **Type Validation**: Ensuring parameter types match schema requirements.
   Solution: Look up expected type from function definition and filter
   tokens accordingly.

## Testing Strategy

### Unit Testing
- Test incremental JSON parser with various partial JSON strings
- Test token validation logic for different states and types
- Test vocabulary loading and token cleaning

### Integration Testing
- Test full pipeline with sample prompts
- Verify output format matches requirements
- Test error handling with malformed inputs

### Edge Cases Tested
- Empty strings
- Large numbers
- Special characters in strings
- Functions with multiple parameters
- Ambiguous prompts
- Missing or invalid input files

## Example Usage

### Input Files

**functions_definition.json**:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

**function_calling_tests.json**:
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet John"}
]
```

### Output

**function_calling_results.json**:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Resources

### Documentation
- [Pydantic Documentation](https://docs.pydantic.dev/) - Data validation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
  - Model loading and tokenization
- [JSON Schema](https://json-schema.org/) - Schema validation concepts
- [Constrained Decoding Paper](https://arxiv.org/abs/2307.09702) -
  Theoretical background

### Articles and Tutorials
- "Function Calling in LLMs" - OpenAI Documentation
- "Structured Output Generation" - Anthropic Research
- "Tokenization in NLP" - Hugging Face Course

### AI Usage

AI tools were used in the following ways:

1. **Code Review**: Used AI to review code for potential bugs and suggest
   improvements in error handling.

2. **Documentation**: AI assisted in generating docstrings and comments,
   which were then reviewed and edited for accuracy.

3. **Debugging**: Used AI to help diagnose issues with token validation
   logic and state machine transitions.

4. **Algorithm Design**: Discussed constrained decoding approaches with AI
   to understand best practices, but the final implementation was designed
   and coded independently.

5. **Testing**: AI helped generate edge case scenarios for testing.

All AI-generated content was thoroughly reviewed, tested, and modified to
ensure correctness and understanding. The core algorithm and implementation
logic were developed independently with AI serving as a reference tool.

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── __main__.py      # Entry point and CLI
│   ├── models.py        # Pydantic models for validation
│   ├── parser.py        # Incremental JSON parser
│   ├── decoder.py       # Constrained decoder implementation
│   └── utils.py         # File I/O utilities
├── data/
│   ├── input/           # Input JSON files
│   └── output/          # Generated output files
├── tests/               # Test files
├── llm_sdk/             # LLM SDK package
├── pyproject.toml       # Project dependencies
├── Makefile             # Build automation
└── README.md            # This file
```

## License

This project is part of the 42 school curriculum and follows the school's
academic policies.