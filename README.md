*This project has been created as part of the 42 curriculum by lfleitas.*

# call me maybe — Introduction to function calling in LLMs

## Description

**call me maybe** is a function calling tool that translates natural language
prompts into structured, schema-compliant function calls, without ever
letting the model "guess" its way into invalid JSON.

Given a prompt like *"What is the sum of 40 and 2?"*, the goal is not to let
the LLM answer `42` in plain text. Instead, the program must return the
right tool to solve it:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

To achieve this reliably with a small 0.6B-parameter model
(`Qwen/Qwen3-0.6B`), the project implements **constrained decoding**:
instead of hoping the model outputs valid JSON on its own, the generation
loop inspects the model's raw logits at every step and masks out every
token that would break either the JSON syntax or the expected function
schema. Only tokens that keep the output both syntactically and
semantically valid are allowed to be selected. This guarantees
**100% parseable, schema-compliant JSON**, regardless of how unreliable
the underlying model normally is at structured generation.

## Instructions

### Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package (included in this repository, workspace-linked via
  `uv`)

### Installation

```bash
make install
```

which simply runs:

```bash
uv sync
```

This installs `numpy`, `pydantic`, `flake8`, `mypy`, and the local
`llm_sdk` workspace package.

### Running the program

Run with the default input/output paths (`data/input/` and
`data/output/`):

```bash
make run
```

which runs:

```bash
uv run python -m src
```

Or specify custom paths explicitly:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Other Makefile targets

| Target        | Description                                                    |
|---------------|------------------------------------------------------------------|
| `install`     | Install dependencies via `uv sync`                              |
| `run`         | Run the main script (`uv run python -m src`)                    |
| `debug`       | Run the main script under Python's `pdb` debugger               |
| `lint`        | Run `flake8` and `mypy` with the mandatory flag set              |
| `lint-strict` | Run `flake8` and `mypy --strict`                                |
| `clean`       | Remove `__pycache__`, `.mypy_cache`, `.pytest_cache`, `.venv`   |

## Algorithm Explanation

The implementation lives in `src/decoder.py` (`ConstrainedDecoder`) and
`src/parser.py` (`IncrementalParser`).

1. **Prompt construction**: the available `functions_definition.json`
   entries and the user's prompt are formatted into a ChatML-style prompt
   (`<|im_start|>system ... <|im_start|>user ... <|im_start|>assistant`) so
   the model's instruction-following weights are used as a soft prior —
   but never trusted for the actual structure.

2. **Forced prefix**: generation is seeded with the literal tokens
   `{"name": "` so that every output starts from a guaranteed-valid JSON
   opening, rather than relying on the model to spontaneously emit `{`.

3. **Token-by-token constrained generation**: at each step the loop
   - calls `model.get_logits_from_input_ids(...)` to get the raw logits
     for the next token,
   - asks the `IncrementalParser` what JSON state we are in (expecting a
     key, a colon, a value, a comma/closing brace, etc.),
   - determines the schema context: which function was selected (once the
     `"name"` value is complete) and which of its parameters are still
     missing,
   - filters the vocabulary down to only the tokens that keep the output
     valid for that state *and* that schema (e.g. only tokens that could
     complete a real function name, only digit-shaped tokens for a
     `number` parameter, only `true`/`false` prefixes for a `boolean`),
   - picks the **highest-logit token among the valid ones** (greedy,
     deterministic decoding) — invalid tokens are effectively treated as
     `-inf` since they are excluded from the candidate set entirely.

4. **Vocabulary preprocessing**: `get_path_to_vocab_file()` is used once at
   startup to load the tokenizer's vocabulary file and pre-split it into
   cheap-to-search subsets (numeric-safe tokens, boolean-safe tokens,
   structural tokens, quote/backslash-free "safe" string tokens, and a
   small "unsafe" leftover) so that per-step filtering never has to scan
   the full ~150k-token vocabulary on the hot path.

5. **Termination**: generation stops as soon as the accumulated text is a
   balanced, closed JSON object (`{` count equals `}` count and the text
   ends with `}`), or after a `max_tokens` safety cap.

6. **Final parsing**: the accumulated text is parsed with `json.loads`. If
   the model produced a function name but never actually returns a
   *closable* structure within the token budget, the result is a
   controlled error dictionary rather than a crash.

### Design decisions

- **Greedy decoding**: token selection is always argmax over the valid
  candidates, favouring determinism and reproducibility over diversity.
- **No hardcoded function list**: valid function names and parameter
  schemas are read entirely from `functions_definition.json` at runtime,
  so the decoder generalizes to any function set.
- **Pydantic everywhere**: `FunctionDefinition`, `ParameterDef`,
  `ReturnDef`, `TestPrompt`, and `ConstrainedDecoder` itself are all
  Pydantic models, giving free input validation and clear error messages
  on malformed `functions_definition.json` / `function_calling_tests.json`
  files.
- **No private `llm_sdk` access**: only the public
  `Small_LLM_Model.encode`, `get_logits_from_input_ids`, and
  `get_path_to_vocab_file` methods are used.
- **Fractional digit cap**: number parsing caps the number of digits
  accepted after a `.` to avoid the decoder stalling on pathological
  digit-repetition loops.
- **Graceful failure per prompt**: each prompt is decoded independently
  inside its own `try/except` in `src/__main__.py`; a failure on one
  prompt appends an `"error"`-named fallback entry instead of aborting the
  whole run.

## Performance Analysis

- **JSON validity**: 100% — by construction, every token that would break
  JSON syntax or the target schema is excluded before sampling, so the
  output is always parseable.
- **Function selection accuracy**: depends on prompt/description clarity,
  but is generally very high since the decoder blocks any token sequence
  that isn't a strict prefix of a real function name.
- **Speed**: dominated by the number of forward passes (one per generated
  token) through the 0.6B model; typically a few seconds per prompt on
  standard hardware, and well within the "under 5 minutes for the full
  test set" requirement for reasonably sized test files.
- **Robustness**: missing/invalid input files, malformed JSON, and
  per-prompt decoding failures are all handled without crashing the
  program (see `src/utils.py` and the fallback logic in
  `src/__main__.py`).

## Challenges Faced

- **Token boundary artifacts**: the tokenizer represents leading spaces
  and newlines with special glyphs (`Ġ`, `Ċ`). These are normalized via
  `_clean_token` before any state or schema check is performed.
- **Merged tokens crossing state boundaries**: a single token can contain
  both the end of a value *and* the following structural character (e.g.
  `"}` or `2,`), which can silently violate schema rules (like closing the
  object before all required parameters are filled). This required
  explicit look-ahead logic inside `_is_token_valid` to detect and reject
  these "merged" tokens instead of only checking the state in isolation.
- **Forcing correct comma/brace placement**: naively allowing `,` or `}`
  whenever both are structurally legal let the model close the object
  early or add trailing commas. The decoder tracks `missing_keys` for the
  currently selected function and only allows `}` once every required
  parameter has actually been emitted.
- **Avoiding hallucinated function names**: without a check, the model
  could start spelling any word after `"name": "`. The decoder verifies at
  every character that the partial name is still a valid prefix of at
  least one known function name.
- **Performance of per-token vocabulary scans**: naively iterating the
  full vocabulary at every generation step is expensive. Restricting the
  search space with precomputed subsets (numeric, boolean, structural,
  safe-string tokens) keeps generation fast.

## Testing Strategy

- The project is structured so that the incremental parser
  (`src/parser.py`) and the decoder's token validity logic
  (`src/decoder.py`) can be unit tested independently of the LLM, since
  they operate on plain strings/state rather than model output.
- End-to-end validation is done by running the full pipeline against the
  provided `data/input/functions_definition.json` and
  `data/input/function_calling_tests.json`, then checking that
  `data/output/function_calling_results.json` is valid JSON and that every
  entry's `name` and `parameters` match the declared schema.
- Manual edge-case checks include: prompts with no obviously matching
  function, functions with multiple parameters, numeric parameters
  (integers and floats), string parameters containing special characters,
  and malformed/missing input files.

## Example Usage

**`functions_definition.json`**
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
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {"type": "string"}
    },
    "returns": {"type": "string"}
  }
]
```

**`function_calling_tests.json`**
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

**Command**
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

**`function_calling_results.json`**
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources

- [Pydantic documentation](https://docs.pydantic.dev/) — used for
  validating function definitions, prompts, and internal decoder state.
- [Hugging Face — How to generate text](https://huggingface.co/blog/how-to-generate) —
  background on token-by-token generation and logits.
- [JSON Schema](https://json-schema.org/) — general reference for schema
  validation concepts applied here to function parameters.
- [Guidance / structured generation techniques](https://github.com/guidance-ai/guidance) —
  general prior art on constrained/guided decoding for LLMs.

### AI usage

AI assistance was used during this project in the following ways:

- Discussing and comparing possible approaches to constrained decoding
  (state tracking, logit masking strategies) before writing the actual
  implementation.
- Reviewing draft code for edge cases (merged tokens, escape sequences in
  JSON strings, off-by-one errors in the incremental parser) and
  suggesting fixes that were then verified and adapted by hand.
- Helping draft and refine docstrings and this README, which were then
  checked against the actual code for accuracy.

All AI-assisted code was read, tested, and understood before being kept in
the project; the core state machine and constrained decoding logic were
written and debugged directly against the model's real output.

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── __main__.py      # CLI entry point, orchestration, error handling
│   ├── models.py         # Pydantic models (FunctionDefinition, TestPrompt, ...)
│   ├── parser.py         # Incremental JSON state parser
│   ├── decoder.py         # ConstrainedDecoder: logit masking + generation loop
│   └── utils.py           # JSON file reading/writing helpers
├── data/
│   ├── input/              # functions_definition.json, function_calling_tests.json
│   └── output/              # Generated function_calling_results.json (not committed)
├── llm_sdk/                # Provided LLM SDK package (uv workspace member)
├── pyproject.toml
├── uv.lock
├── Makefile
└── README.md
```