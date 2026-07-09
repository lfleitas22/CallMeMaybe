"""
Constrained decoder that enforces JSON structure and function schema
token by token.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, ClassVar


from pydantic import BaseModel, Field, ConfigDict

from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from src.parser import IncrementalParser, JSONState


class ConstrainedDecoder(BaseModel):
    """
    Generates function call JSON using the LLM under strict schema
    constraints.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True
    )

    # Constant without a type hint is ignored by Pydantic validation
    # (which is what we want)
    MAX_FRACTIONAL_DIGITS: ClassVar[int] = 10

    # Compile regexes once at the class level
    FLOAT_PREFIX_PATTERN: ClassVar[re.Pattern[str]] = (
        re.compile(r'^-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?\d*)?$')
    )
    FLOAT_FINISHED_PATTERN: ClassVar[re.Pattern[str]] = (
        re.compile(r'^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$')
    )
    INT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r'^-?(?:0|[1-9]\d*)$')

    NUMBER_SPLIT_PATTERN: ClassVar[re.Pattern[str]] = (
        re.compile(r'^([-+0-9.eE]+)(.*)$')
    )

    # 2. Pydantic Fields
    model: Small_LLM_Model
    # Alias allows your existing __main__.py to keep using
    # `available_functions=functions`
    functions: List[FunctionDefinition] = Field(alias="available_functions")
    vocab: Dict[str, int] = Field(default_factory=dict)
    selected_function_name: Optional[str] = None
    cleaned_vocab: Dict[int, str] = Field(default_factory=dict)
    static_state_masks: Dict[JSONState, List[int]] = (
        Field(default_factory=dict))
    number_candidates: Dict[int, str] = Field(default_factory=dict)
    boolean_candidates: Dict[int, str] = Field(default_factory=dict)
    structural_candidates: Dict[int, str] = Field(default_factory=dict)
    key_content_candidates: Dict[int, str] = Field(default_factory=dict)
    string_start_candidates: Dict[int, str] = Field(default_factory=dict)
    exact_syntax_ids: Dict[str, int] = Field(default_factory=dict)
    # For strings, we separate tokens that contain special JSON characters
    # from those that are perfectly safe plain text.
    safe_string_ids: List[int] = Field(default_factory=list)
    unsafe_string_vocab: Dict[int, str] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.vocab:
            self.vocab = self._load_vocabulary()

        # Pre-clean all tokens exactly once on startup
        if not self.cleaned_vocab:
            self.cleaned_vocab = {
                tok_id: self._clean_token(tok_str)
                for tok_str, tok_id in self.vocab.items()
            }
        if not self.static_state_masks:
            static_states = [
                JSONState.EXPECT_OBJECT_START,
                JSONState.EXPECT_KEY_QUOTE,
                JSONState.EXPECT_COLON
            ]
            for state in static_states:
                valid_ids = []
                for tok_id, clean_tok in self.cleaned_vocab.items():
                    if self._is_token_valid(clean_tok, state, "", "",
                                            "", False, None):
                        valid_ids.append(tok_id)
                self.static_state_masks[state] = valid_ids
        # 1. Number Candidates: Only tokens strictly containing
        # numeric/structural characters
        if not self.number_candidates:
            allowed_num_chars = set("0123456789.eE+- ,}\n\t\r")
            self.number_candidates = {
                tok_id: tok for tok_id, tok in self.cleaned_vocab.items()
                if set(tok).issubset(allowed_num_chars)
            }

        # 2. Boolean Candidates: Only tokens containing
        # boolean/structural characters
        if not self.boolean_candidates:
            allowed_bool_chars = set("truefalse ,}\n\t\r")
            self.boolean_candidates = {
                tok_id: tok for tok_id, tok in self.cleaned_vocab.items()
                if set(tok).issubset(allowed_bool_chars)
            }

        # 3. String Partitioning: Isolate tokens that contain
        # quotes or backslashes
        if not self.safe_string_ids and not self.unsafe_string_vocab:
            for tok_id, tok in self.cleaned_vocab.items():
                if '"' in tok or '\\' in tok:
                    self.unsafe_string_vocab[tok_id] = tok
                else:
                    self.safe_string_ids.append(tok_id)

        if not self.structural_candidates:
            self.structural_candidates = {
                tok_id: tok for tok_id, tok in self.cleaned_vocab.items()
                if set(tok).issubset(set(",} \n\t\r"))
            }
        # 4. Key Candidates: Only tokens containing letters,
        # underscores, and quotes
        if not self.key_content_candidates:
            # Dynamically gather all possible characters used in
            # your schema keys
            valid_key_chars = set('nameparameters_"')
            for f in self.functions:
                for k in f.parameters.keys():
                    valid_key_chars.update(k)

            self.key_content_candidates = {
                tok_id: tok for tok_id, tok in self.cleaned_vocab.items()
                if set(tok).issubset(valid_key_chars)
            }

        # 5. String Start Candidates: Tokens containing a quote
        if not self.string_start_candidates:
            self.string_start_candidates = {
                tok_id: tok for tok_id, tok in self.cleaned_vocab.items()
                if '"' in tok
            }
        if not self.exact_syntax_ids:
            for tid, t in self.cleaned_vocab.items():
                if t in (":", "{", "}", ","):
                    self.exact_syntax_ids[t] = tid

    def _load_vocabulary(self) -> Dict[str, int]:
        """Load the tokenizer vocabulary from the SDK-provided path."""
        vocab_path = Path(self.model.get_path_to_vocab_file())
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Vocabulary file must contain a dict")
            return {str(k): int(v) for k, v in data.items()}

    def _clean_token(self, raw_token: str) -> str:
        """Normalize token by replacing special whitespace characters."""
        return raw_token.replace("Ġ", " ").replace("Ċ", "\n")

    def _is_token_valid(
        self,
        clean_token: str,
        state: JSONState,
        current_key: str,
        current_key_partial: str,
        current_value_partial: str,
        is_inside_parameters: bool,
        expected_type: Optional[str] = None,
        missing_keys: Optional[List[str]] = None,
    ) -> bool:
        """
        Check whether a cleaned token is valid given the current
        parser state, the partial key/value being built, and the
        schema constraints.
        """
        if not clean_token:
            return True

        # 1. Structural states
        if state == JSONState.EXPECT_OBJECT_START:
            return clean_token.lstrip().startswith('{')

        elif state == JSONState.EXPECT_KEY_QUOTE:
            # Prevent empty parameters dict from appending a comma
            if ('}' in clean_token and ',' in
                    clean_token[clean_token.find('}'):]):
                return False
            return '"' in clean_token or '}' in clean_token

        elif state == JSONState.EXPECT_COLON:
            return ':' in clean_token

        elif state == JSONState.EXPECT_COMMA_OR_END:
            if not is_inside_parameters:
                # Root level boundaries
                if current_key == "name":
                    return ',' in clean_token and '}' not in clean_token
                return '}' in clean_token and ',' not in clean_token
            else:
                all_parameters_filled = (missing_keys is not None
                                         and len(missing_keys) == 0)

                # Prevent the merged token '},' bypass
                if ('}' in clean_token and ',' in
                        clean_token[clean_token.find('}'):]):
                    return False

                # Schema Enforcement: Force JSON to close if no keys remain
                if all_parameters_filled:
                    return '}' in clean_token and ',' not in clean_token

                return ',' in clean_token or '}' in clean_token

        # 2. Schema‑aware states
        elif state == JSONState.IN_KEY:
            proposed = current_key_partial + clean_token.split('"')[0]
            valid_keys: List[str] = []
            if is_inside_parameters:
                # Only allow keys that are still missing!
                if missing_keys is not None:
                    valid_keys = missing_keys
            else:
                valid_keys = ["name", "parameters"]
            return any(k.startswith(proposed) for k in valid_keys)

        elif state == JSONState.IN_STRING_VALUE:
            # 1. Enforce strict JSON escape sequences.
            # Calculate trailing backslashes to see if an escape
            # sequence is active.
            bs_count = (len(current_value_partial) -
                        len(current_value_partial.rstrip('\\')))
            is_active_escape = (bs_count % 2 == 1)

            if is_active_escape:
                # In JSON, a backslash MUST be followed by
                # one of these characters.
                valid_escapes = ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u')
                if not clean_token.startswith(valid_escapes):
                    # This blocks invalid regex tokens like '\s',
                    # forcing the LLM
                    # to backtrack and output another '\' to make
                    # it valid '\\s'.
                    return False

            # 2. Prevent token-merging deadlocks while enforcing JSON schema.
            if '"' in clean_token:
                if clean_token == '"':
                    return True

                # If the tokenizer merges the quote (e.g. 's"'),
                # simulate the addition.
                combined = current_value_partial + clean_token

                # Find the first unescaped quote in the newly combined string
                for i in range(len(current_value_partial), len(combined)):
                    if combined[i] == '"':
                        # Count backslashes before this specific quote
                        b_count = 0
                        curr = i - 1
                        while curr >= 0 and combined[curr] == '\\':
                            b_count += 1
                            curr -= 1

                        if b_count % 2 == 0:
                            # The string has formally closed.
                            rest = combined[i + 1:]
                            if not all(c in ' \n\t\r,}' for c in rest):
                                return False

                            # --- MERGED TOKEN & SCHEMA COMPLETION ENFORCEMENT
                            if is_inside_parameters:
                                all_parameters_filled = (
                                    missing_keys is not None
                                    and len(missing_keys) == 0)

                                if ('}' in rest and ','
                                        in rest[rest.find('}'):]):
                                    return False

                                # Prevent comma if all schema keys are okey
                                if all_parameters_filled and ',' in rest:
                                    return False
                            # ----------------------------------------------------

                            return True

                # All quotes in this token were escaped, so it's valid
                # internal string content.
                return True

            # --- NEW: PREVENT FUNCTION NAME HALLUCINATION ---
            if not is_inside_parameters and current_key == "name":
                proposed = current_value_partial + clean_token.replace('"', '')
                valid_names = [f.name for f in self.functions]
                # If the proposed string isn't spelling a valid function name,
                # kill the token probability
                if not any(name.startswith(proposed) for name in valid_names):
                    return False
            # ------------------------------------------------

            return True

        elif state == JSONState.IN_NUMBER_VALUE:
            combined = current_value_partial + clean_token
            is_integer = (expected_type == "integer")

            # Use compiled split pattern
            match = self.NUMBER_SPLIT_PATTERN.match(combined)
            if not match:
                return False

            num_part, rest = match.groups()

            if not rest:
                if is_integer:
                    if '.' in clean_token or 'e' in clean_token.lower():
                        return False
                    # Use compiled int pattern
                    if not self.INT_PATTERN.match(num_part):
                        return False
                else:
                    # Use compiled float prefix pattern
                    if not self.FLOAT_PREFIX_PATTERN.match(num_part):
                        return False
                # ... [fractional digits check remains the same] ...
                return True
            else:
                if is_integer:
                    # Use compiled int pattern
                    if not self.INT_PATTERN.match(num_part):
                        return False
                else:
                    # Use compiled float finished pattern
                    if not self.FLOAT_FINISHED_PATTERN.match(num_part):
                        return False

                if rest[0] not in (',', '}', ' ', '\n', '\t'):
                    return False

                # --- MERGED TOKEN & SCHEMA COMPLETION ENFORCEMENT ---
                if is_inside_parameters:
                    all_parameters_filled = (missing_keys is not None
                                             and len(missing_keys) == 0)

                    if '}' in rest and ',' in rest[rest.find('}'):]:
                        return False

                    # Prevent comma if all schema keys are satisfied
                    if all_parameters_filled and ',' in rest:
                        return False
                # ----------------------------------------------------

                return True

        elif state == JSONState.IN_BOOLEAN_VALUE:
            combined = current_value_partial + clean_token
            return "true".startswith(combined) or "false".startswith(combined)

        elif state == JSONState.EXPECT_VALUE:
            # Lookups are removed. We just use the passed expected_type.
            if expected_type == "number" or expected_type == "integer":
                stripped = clean_token.lstrip()
                return bool(stripped) and (
                    stripped[0].isdigit() or stripped[0] == '-'
                )
            elif expected_type == "boolean":
                stripped = clean_token.lstrip()
                return stripped.startswith(('t', 'f'))
            elif expected_type == "string":
                return '"' in clean_token
            elif current_key == "name":
                return '"' in clean_token
            return True

        return False

    def _get_best_valid_token(
        self, logits: List[float], valid_ids: List[int]
    ) -> int:
        """
        Finds the ID of the highest probability valid token in O(K) time,
        completely bypassing heavy numpy array allocations and masking.
        """
        best_id = valid_ids[0]
        best_logit = logits[best_id]

        for vid in valid_ids[1:]:
            if logits[vid] > best_logit:
                best_logit = logits[vid]
                best_id = vid

        return best_id

    def generate_function_call(
        self, prompt: str, max_tokens: int = 256, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a function call JSON for the given natural language
        prompt. Returns a dict with keys 'name' and 'parameters'.
        """
        self.selected_function_name = None

        # Build a system prompt that lists available functions
        functions_str = json.dumps(
            [f.model_dump() for f in self.functions],
            separators=(',', ':')
        )

        # 2. Use ChatML formatting to trigger the model's
        # instruction-following weights
        system_prompt = (
            f"<|im_start|>system\n"
            f"Select the correct function and extract parameters as JSON.\n"
            f"Functions:{functions_str}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Tokenize the full prompt
        raw_encoded = self.model.encode(system_prompt).tolist()
        if isinstance(raw_encoded[0], list):
            input_ids = raw_encoded[0]
        else:
            input_ids = list(raw_encoded)

        # Force the JSON opening: {"name": "
        forced_prefix = '{"name": "'
        prefix_raw = self.model.encode(forced_prefix).tolist()
        if isinstance(prefix_raw[0], list):
            prefix_ids = prefix_raw[0]
        else:
            prefix_ids = list(prefix_raw)

        generated_ids: List[int] = list(prefix_ids)
        clean_text = forced_prefix
        parser = IncrementalParser()

        if verbose:
            print(f"\n[Verbose] Prompt: {prompt}")
            print(f"[Verbose] Prefix: {forced_prefix}")

        cached_expected_type: Optional[str] = None
        current_fn_def = None
        for _ in range(max_tokens):
            # 1. Current parser state and context
            state = parser.get_current_state(clean_text)

            # Update selected function name once "name" value is completed
            if (
                self.selected_function_name is None
                and parser.top_level_key_values
                and "name" in parser.top_level_key_values
            ):
                self.selected_function_name = (
                    parser.top_level_key_values["name"]
                )
                # 2. CACHE THE FUNCTION DEFINITION HERE (Runs only once!)
                current_fn_def = next((f for f in self.functions
                                       if f.name ==
                                       self.selected_function_name), None)

            current_key = parser.current_key
            current_key_partial = parser.current_key_partial
            current_value_partial = parser.current_value_content
            is_inside_params = (
                len(parser.context_keys) > 0
                and parser.context_keys[-1] == "parameters"
            )
            # 3. --- FIX THE DEAD VARIABLE ---
            # Update the expected type so the routing logic actually works
            if current_key == "name":
                cached_expected_type = "string"
            elif is_inside_params and current_fn_def is not None:
                # Mypy now knows current_fn_def is 100% safe to use
                if current_key in current_fn_def.parameters:
                    cached_expected_type = (
                        current_fn_def.parameters[current_key].type)
                else:
                    cached_expected_type = None
            else:
                cached_expected_type = None

            # 4. Use the cached current_fn_def instead of searching
            # for it again
            missing_keys: Optional[List[str]] = None
            if (
                is_inside_params
                and current_fn_def is not None
                and state in (JSONState.EXPECT_KEY_QUOTE,
                              JSONState.EXPECT_COMMA_OR_END,
                              JSONState.IN_KEY)
            ):
                expected_keys = list(current_fn_def.parameters.keys())

                compressed_text = clean_text.replace(" ",
                                                     "").replace("\n", "")
                params_marker = '"parameters":{'

                if params_marker in compressed_text:
                    # ISOLATION: Only search for keys AFTER the
                    # parameters dictionary opens
                    params_body = (compressed_text.split(params_marker,
                                                         1)[1])
                    found_keys = set()
                    for k in expected_keys:
                        if f'"{k}":' in params_body:
                            found_keys.add(k)

                    # Always include the key currently being typed
                    if current_key in expected_keys:
                        found_keys.add(current_key)

                    missing_keys = ([k for k in expected_keys
                                     if k not in found_keys])
                else:
                    missing_keys = expected_keys
            # -------------------------

            # 3. Gather valid next tokens
            if state in self.static_state_masks:
                # Instantly retrieve valid tokens, bypassing 150,000 checks
                valid_ids = self.static_state_masks[state]

            elif (state == JSONState.EXPECT_COMMA_OR_END
                  and not is_inside_params):
                # THE RAILROAD: Absolute root-level strictness
                # using text look-behind
                valid_ids = []
                is_after_params = clean_text.rstrip().endswith('}')

                for tid, tok in self.structural_candidates.items():
                    if is_after_params:
                        # Parameters dict closed -> Force root object to close
                        # Require a '}' and strictly forbid ','
                        if '}' in tok and ',' not in tok:
                            valid_ids.append(tid)
                    else:
                        # Name string closed -> Force comma to prepare
                        # for parameters key
                        # Require a ',' and strictly forbid '}'
                        if ',' in tok and '}' not in tok:
                            valid_ids.append(tid)
            else:
                valid_ids = []

                # --- ROUTING LOGIC ---
                search_space = self.cleaned_vocab  # Default fallback

                if state == JSONState.EXPECT_COMMA_OR_END:
                    search_space = self.structural_candidates

                elif (state == JSONState.IN_NUMBER_VALUE or
                        (state == JSONState.EXPECT_VALUE and
                         cached_expected_type in ("number", "integer"))):
                    search_space = self.number_candidates

                elif (state == JSONState.IN_BOOLEAN_VALUE or
                        (state == JSONState.EXPECT_VALUE and
                         cached_expected_type == "boolean")):
                    search_space = self.boolean_candidates

                elif state == JSONState.IN_STRING_VALUE:
                    # If we are in a string and not processing an active
                    # escape sequence (\),
                    # all safe tokens are instantly valid without any regex
                    # evaluation.
                    bs_count = (len(current_value_partial) -
                                len(current_value_partial.rstrip('\\')))
                    if bs_count % 2 == 0:
                        valid_ids.extend(self.safe_string_ids)
                        search_space = self.unsafe_string_vocab
                # -------------------------
                elif state == JSONState.IN_KEY:
                    search_space = self.key_content_candidates

                elif state == JSONState.EXPECT_VALUE and (
                        cached_expected_type == "string"
                        or current_key == "name"):
                    search_space = self.string_start_candidates
                # Only iterate over the aggressively pruned search space
                for token_id, clean_tok in search_space.items():
                    if self._is_token_valid(
                        clean_tok,
                        state,
                        current_key,
                        current_key_partial,
                        current_value_partial,
                        is_inside_params,
                        cached_expected_type,
                        missing_keys,  # <-- ADD THIS ARGUMENT
                    ):
                        valid_ids.append(token_id)
            if valid_ids:
                if state == JSONState.EXPECT_COLON:
                    valid_ids = [self.exact_syntax_ids[":"]]
                elif state == JSONState.EXPECT_OBJECT_START:
                    valid_ids = [self.exact_syntax_ids["{"]]
                elif state == JSONState.EXPECT_COMMA_OR_END:
                    if (is_inside_params and missing_keys is not
                            None and len(missing_keys) == 0):
                        valid_ids = [self.exact_syntax_ids["}"]]
                    elif (not is_inside_params and
                          clean_text.rstrip().endswith('}')):
                        valid_ids = [self.exact_syntax_ids["}"]]
                    elif not is_inside_params:
                        valid_ids = [self.exact_syntax_ids[","]]
                elif state == JSONState.EXPECT_KEY_QUOTE:
                    if (is_inside_params and missing_keys is not
                            None and len(missing_keys) == 0):
                        valid_ids = [self.exact_syntax_ids["}"]]
            if not valid_ids:
                if verbose:
                    print("\n[!] No valid tokens available. Breaking.")
                break

            # --- OPTIMIZATION 1: FAST-FORWARD ---
            if len(valid_ids) == 1:
                # Skip the LLM entirely if the schema strictly
                # forces this token
                next_token_id = valid_ids[0]
            else:
                # --- OPTIMIZATION 2: CONTEXT TRUNCATION ---
                sequence = input_ids + generated_ids
                # Truncate to the last 512 tokens to prevent the
                # model from slowing down
                truncated_sequence = sequence[-512:]

                # 4. Forward pass (Only called when the LLM actually
                # needs to make a choice)
                raw_logits = (
                    self.model.get_logits_from_input_ids(truncated_sequence))
                if isinstance(raw_logits[0], list):
                    logits = raw_logits[0]
                else:
                    logits = list(raw_logits)

                # Apply mask and select next token (greedy)
                next_token_id = self._get_best_valid_token(logits, valid_ids)
            generated_ids.append(next_token_id)

            # 5. Update the cleaned text
            # Instantly fetch the pre-cleaned token in O(1) time
            clean_tok = self.cleaned_vocab.get(next_token_id, "")
            clean_text += clean_tok

            if verbose:
                safe = clean_tok.replace('\n', '\\n')
                print(f"'{safe}' ", end="", flush=True)

            # 6. Stop when a complete JSON object is formed
            if (
                clean_text.strip().endswith("}")
                and clean_text.count("{") == clean_text.count("}")
            ):
                if verbose:
                    print("\n[Verbose] JSON object completed.")
                break

        if verbose:
            print()

        try:
            result = json.loads(clean_text)
            if not isinstance(result, dict) or "name" not in result:
                raise ValueError("Generated JSON missing 'name' key")
            return {
                "name": result["name"],
                "parameters": result.get("parameters", {})
            }
        except (json.JSONDecodeError, ValueError):
            return {"name": "", "parameters": {}}
