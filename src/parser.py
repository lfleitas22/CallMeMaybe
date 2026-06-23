"""
Incremental JSON parser that tracks state, keys, and partial values.
"""
from enum import Enum, auto
from typing import List, Dict
from pydantic import BaseModel, Field


class JSONState(Enum):
    EXPECT_OBJECT_START = auto()
    EXPECT_KEY_QUOTE = auto()
    IN_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    IN_STRING_VALUE = auto()
    IN_NUMBER_VALUE = auto()
    IN_BOOLEAN_VALUE = auto()
    EXPECT_COMMA_OR_END = auto()
    DONE = auto()


class IncrementalParser(BaseModel):
    """Determines the current JSON generation state from a partial string."""

    state_stack: List[JSONState] = Field(
        default_factory=lambda: [JSONState.EXPECT_OBJECT_START]
    )
    current_key: str = ""
    context_keys: List[str] = Field(default_factory=list)
    top_level_key_values: Dict[str, str] = Field(default_factory=dict)
    current_key_partial: str = ""
    current_value_content: str = ""

    def get_current_state(self, partial_json: str) -> JSONState:
        """
        Parse the partial JSON character by character and return the
        final state. Updates internal context and partial tracking
        attributes.
        """
        # Reset state for a fresh parse
        self.state_stack = [JSONState.EXPECT_OBJECT_START]
        self.current_key = ""
        self.context_keys = []
        self.top_level_key_values = {}
        self.current_key_partial = ""
        self.current_value_content = ""

        current_key_partial = ""
        current_value_content = ""
        current_key_value_stack: List[Dict[str, str]] = [{}]

        i = 0
        while i < len(partial_json):
            char = partial_json[i]
            state = self.state_stack[-1]

            # Skip whitespace outside of strings
            if char.isspace() and state not in (JSONState.IN_KEY,
                                                JSONState.IN_STRING_VALUE,
                                                JSONState.IN_NUMBER_VALUE,
                                                JSONState.IN_BOOLEAN_VALUE):
                i += 1
                continue

            # --- Main state machine ---
            if state == JSONState.EXPECT_OBJECT_START:
                if char == '{':
                    self.state_stack[-1] = JSONState.EXPECT_KEY_QUOTE

            elif state == JSONState.EXPECT_KEY_QUOTE:
                if char == '}':
                    # Empty object – pop and handle
                    self.state_stack.pop()
                    if self.context_keys:
                        self.context_keys.pop()
                    if current_key_value_stack:
                        current_key_value_stack.pop()
                    if not self.state_stack:
                        self.state_stack.append(JSONState.DONE)
                        break
                    self.state_stack[-1] = JSONState.EXPECT_COMMA_OR_END
                elif char == '"':
                    self.state_stack[-1] = JSONState.IN_KEY
                    current_key_partial = ""

            elif state == JSONState.IN_KEY:
                if char == '"' and not self._is_escaped(partial_json, i):
                    self.current_key = current_key_partial
                    self.state_stack[-1] = JSONState.EXPECT_COLON
                else:
                    current_key_partial += char

            elif state == JSONState.EXPECT_COLON:
                if char == ':':
                    self.state_stack[-1] = JSONState.EXPECT_VALUE

            elif state == JSONState.EXPECT_VALUE:
                if char == '"':
                    self.state_stack[-1] = JSONState.IN_STRING_VALUE
                    current_value_content = ""
                elif char.isdigit() or char == '-':
                    self.state_stack[-1] = JSONState.IN_NUMBER_VALUE
                    current_value_content = char
                elif char in ('t', 'f'):
                    self.state_stack[-1] = JSONState.IN_BOOLEAN_VALUE
                    current_value_content = char
                elif char == '{':
                    # Nested object (e.g., "parameters")
                    self.state_stack.append(JSONState.EXPECT_KEY_QUOTE)
                    self.context_keys.append(self.current_key)
                    current_key_value_stack.append({})
                else:
                    # Unexpected character – remain in same state
                    pass

            elif state == JSONState.IN_STRING_VALUE:
                if char == '"' and not self._is_escaped(partial_json, i):
                    if current_key_value_stack:
                        kv_dict = current_key_value_stack[-1]
                        kv_dict[self.current_key] = current_value_content
                    self.state_stack[-1] = JSONState.EXPECT_COMMA_OR_END
                else:
                    current_value_content += char

            elif state == JSONState.IN_NUMBER_VALUE:
                if char in (',', '}', ' ', '\n', '\t'):
                    # Number ended – store if we have a key
                    if self.current_key and current_key_value_stack:
                        kv_dict = current_key_value_stack[-1]
                        kv_dict[self.current_key] = current_value_content
                    self.state_stack[-1] = JSONState.EXPECT_COMMA_OR_END
                    i -= 1  # reprocess this separator
                else:
                    current_value_content += char

            elif state == JSONState.IN_BOOLEAN_VALUE:
                if char.isalpha():
                    current_value_content += char
                    if current_value_content in ("true", "false"):
                        if current_key_value_stack:
                            kv_dict = current_key_value_stack[-1]
                            kv_dict[self.current_key] = (
                                current_value_content
                            )
                        self.state_stack[-1] = (
                            JSONState.EXPECT_COMMA_OR_END
                        )
                else:
                    # Invalid boolean character
                    current_value_content += char

            elif state == JSONState.EXPECT_COMMA_OR_END:
                if char == ',':
                    self.state_stack[-1] = JSONState.EXPECT_KEY_QUOTE
                elif char == '}':
                    self.state_stack.pop()
                    if self.context_keys:
                        self.context_keys.pop()
                    if current_key_value_stack:
                        current_key_value_stack.pop()
                        # If we are back to the top level, save values
                        if len(current_key_value_stack) == 1:
                            self.top_level_key_values = (
                                current_key_value_stack[0]
                            )
                    if not self.state_stack:
                        self.state_stack.append(JSONState.DONE)
                        break
                    self.state_stack[-1] = JSONState.EXPECT_COMMA_OR_END
            i += 1

        # After parsing, expose the partial key/value currently being typed
        final_state = self.state_stack[-1]
        if final_state == JSONState.IN_KEY:
            self.current_key_partial = current_key_partial
        elif final_state in (
            JSONState.IN_STRING_VALUE,
            JSONState.IN_NUMBER_VALUE,
            JSONState.IN_BOOLEAN_VALUE
        ):
            self.current_value_content = current_value_content

        # Ensure top_level_key_values is available
        if len(current_key_value_stack) == 1:
            self.top_level_key_values = current_key_value_stack[0]

        return final_state

    def _is_escaped(self, partial_json: str, index: int) -> bool:
        """
        Correctly determines if a character is escaped by counting
        consecutive preceding backslashes.
        """
        bs_count = 0
        curr = index - 1
        while curr >= 0 and partial_json[curr] == '\\':
            bs_count += 1
            curr -= 1
        # If the number of preceding backslashes is odd, the
        # character is escaped.
        return bs_count % 2 == 1
