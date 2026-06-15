import json
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# Assuming llm_sdk is in your path/virtual env as per your tree
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from src.parser import JSONState, IncrementalParser


class ConstrainedDecoder:
    def __init__(self, model: Small_LLM_Model, available_functions:
                 List[FunctionDefinition]):
        self.model = model
        self.functions = available_functions
        self.vocab = self._load_vocabulary()

    def _load_vocabulary(self) -> Dict[str, int]:
        """
        Loads the tokenizer vocabulary from the path provided by the SDK.
        Returns a dictionary mapping token strings to their IDs.
        """
        vocab_path = Path(self.model.get_path_to_vocab_file())
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_valid_next_tokens(self, generated_text: str) -> List[int]:
        parser = IncrementalParser()
        current_state = parser.get_current_state(generated_text)

        valid_token_ids = []

        for token_str, token_id in self.vocab.items():
            # LLM tokenizers often prefix words with special characters
            # for spaces.
            # E.g., Qwen might use 'Ġ' or similar. You must normalize
            # the token string.
            # Example normalization (adjust based on actual Qwen
            # vocabulary format):
            clean_token = token_str.replace("Ġ", " ").replace("Ċ", "\n")

            if self._is_token_valid_for_state(clean_token, current_state,
                                              parser.current_key,
                                              generated_text):
                valid_token_ids.append(token_id)

        return valid_token_ids

    def _is_token_valid_for_state(self, token: str, state: JSONState,
                                  current_key: str, generated_text:
                                  str) -> bool:
        """
        Determines if a specific token string satisfies the current
        JSON structural expectation
        and the schema constraints.
        """
        # Empty tokens are usually artifacts, allow them or ignore them based
        # on tokenizer behavior
        if not token:
            return True

        # 1. Structural Validation
        if state == JSONState.EXPECT_OBJECT_START:
            return token.lstrip().startswith('{')

        elif state == JSONState.EXPECT_KEY_QUOTE:
            return '"' in token or token.isspace()

        elif state == JSONState.EXPECT_COLON:
            return ':' in token or token.isspace()

        elif state == JSONState.EXPECT_COMMA_OR_END:
            return ',' in token or '}' in token or token.isspace()

        # 2. Semantic/Schema Validation
        elif state == JSONState.IN_NUMBER_VALUE:
            # Only allow digits, decimals, and structural terminators
            return all(c.isdigit() or c in '.-eE, }' for c in token)

        elif state == JSONState.IN_KEY:
            # 1. Gather all legally allowed keys based on our output schema
            valid_keys = ["prompt", "name", "parameters"]

            # 2. Add all parameter names from all available functions
            for fn in self.functions:
                valid_keys.extend(fn.parameters.keys())

            # Remove duplicates to optimize the check
            valid_keys = list(set(valid_keys))

            # 3. Extract what has already been typed for the current key.
            # We look for the last quote in the generated text to find
            # the start of our key.
            last_quote_idx = generated_text.rfind('"')
            if last_quote_idx != -1:
                partial_key = generated_text[last_quote_idx + 1:]
            else:
                partial_key = ""

            # 4. Combine the prefix with the proposed token.
            # We use .split('"')[0] to ignore the closing quote if the
            # token contains one (e.g., 'meters":')
            proposed_string = partial_key + token.split('"')[0]

            # 5. Check if the resulting string matches or is a prefix
            # of any allowed key
            return any(allowed_key.startswith(proposed_string) for
                       allowed_key in valid_keys)

        elif state == JSONState.IN_STRING_VALUE:
            # If the current key is "name", force the token to be
            # a valid function name
            if current_key == "name":
                available_names = [f.name for f in self.functions]
                # Check if the token is part of any available function name
                return any(fn_name.startswith(token.strip('"')) for fn_name
                           in available_names)

            # If it's a normal string, allow mostly anything except
            # unescaped structural quotes
            return True

        elif state == JSONState.EXPECT_VALUE:
            # If we expect a value, ensure the token starts with the correct
            # type based on the schema
            # E.g., if current_key == "name", it must start with '"'
            return True  # Implement strict type enforcement here

        return False

    def _mask_logits(self, logits: List[float],
                     valid_token_ids: List[int]) -> List[float]:
        """
        Sets the probability of all invalid tokens to negative infinity.
        """
        # Convert to numpy array for fast manipulation
        logits_arr = np.array(logits, dtype=np.float32)

        # Create a boolean mask of the same size, default to False
        mask = np.zeros_like(logits_arr, dtype=bool)

        # Set True only for valid token indices
        mask[valid_token_ids] = True

        # Apply constraint: if not in mask, set to -inf
        logits_arr[~mask] = -np.inf

        return logits_arr.tolist()

    def generate_function_call(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """
        The main generation loop executing constrained decoding.
        """
        # Format the prompt
        system_prompt = f"Extract the function call for: {prompt}\n"
        
        # 1. Tokenize the input
        raw_encoded = self.model.encode(system_prompt).tolist()
        
        # PyTorch tensors inject a batch dimension (e.g., [[1, 2, 3]]). 
        # We MUST flatten it so the SDK receives a strict List[int].
        if len(raw_encoded) == 1 and isinstance(raw_encoded[0], list):
            input_ids = raw_encoded[0]
        else:
            input_ids = list(raw_encoded)
            
        generated_ids: List[int] = []
        generated_text = ""
        
        for _ in range(max_tokens):
            # 2. Build the current sequence
            current_sequence = input_ids + generated_ids
            
            # 3. Get logits from the model
            raw_logits = self.model.get_logits_from_input_ids(current_sequence)
            
            # Ensure logits are also flattened to a 1D List[float]
            if len(raw_logits) == 1 and isinstance(raw_logits[0], list):
                logits = raw_logits[0]
            else:
                logits = list(raw_logits)
            
            # 4. Determine which tokens maintain JSON/Schema validity
            valid_token_ids = self._get_valid_next_tokens(generated_text)
            
            # 5. Constrain the decoding
            constrained_logits = self._mask_logits(logits, valid_token_ids)
            
            # 6. Select the next token
            next_token_id = int(np.argmax(constrained_logits))
            generated_ids.append(next_token_id)
            
            # 7. Update the text state
            next_token_str = next((k for k, v in self.vocab.items() if v == next_token_id), "")
            generated_text += next_token_str
            
            # 8. Stop condition: valid JSON object is closed
            if generated_text.endswith("}") and generated_text.count("{") > 0 and generated_text.count("{") == generated_text.count("}"):
                break
                
        # Parse the guaranteed-valid string back into a Python dictionary
        try:
            return json.loads(generated_text)
        except json.JSONDecodeError:
            return {"error": "Failed to generate valid JSON", "raw": generated_text}
        except Exception as e:
            import traceback
            traceback.print_exc()  # <--- Add this temporarily
            print(f"    [!] Failed to parse prompt: {e}", file=sys.stderr)
