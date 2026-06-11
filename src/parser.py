from enum import Enum, auto
from typing import List


class JSONState(Enum):
    EXPECT_OBJECT_START = auto()  # Waiting for '{'
    EXPECT_KEY_QUOTE = auto()     # Waiting for '"' to start a key
    IN_KEY = auto()               # Inside a key string
    EXPECT_COLON = auto()         # Waiting for ':'
    EXPECT_VALUE = auto()         # Waiting for string, number, or '{'
    IN_STRING_VALUE = auto()      # Inside a string value
    IN_NUMBER_VALUE = auto()      # Inside a number value
    EXPECT_COMMA_OR_END = auto()  # Waiting for ',' or '}'
    DONE = auto()                 # JSON object is closed


class IncrementalParser:
    def __init__(self):
        # We use a stack to track nested objects (like the "parameters" dict)
        self.state_stack: List[JSONState] = [JSONState.EXPECT_OBJECT_START]
        self.current_key: str = ""

    def get_current_state(self, partial_json: str) -> JSONState:
        """
        Parses the partial JSON string character by character to determine
        the exact state at the end of the string.
        """
        self.state_stack = [JSONState.EXPECT_OBJECT_START]
        self.current_key = ""
        current_string = ""

        i = 0
        while i < len(partial_json):
            char = partial_json[i]
            state = self.state_stack[-1]

            # Skip whitespace unless we are inside a string
            if char.isspace() and state not in (JSONState.IN_KEY,
                                                JSONState.IN_STRING_VALUE):
                i += 1
                continue

            if state == JSONState.EXPECT_OBJECT_START:
                if char == '{':
                    self.state_stack[-1] = JSONState.EXPECT_KEY_QUOTE

            elif state == JSONState.EXPECT_KEY_QUOTE:
                if char == '"':
                    self.state_stack[-1] = JSONState.IN_KEY
                    current_string = ""

            elif state == JSONState.IN_KEY:
                if char == '"' and partial_json[i-1] != '\\':
                    self.current_key = current_string
                    self.state_stack[-1] = JSONState.EXPECT_COLON
                else:
                    current_string += char

            elif state == JSONState.EXPECT_COLON:
                if char == ':':
                    self.state_stack[-1] = JSONState.EXPECT_VALUE

            elif state == JSONState.EXPECT_VALUE:
                if char == '"':
                    self.state_stack[-1] = JSONState.IN_STRING_VALUE
                elif char.isdigit() or char == '-':
                    self.state_stack[-1] = JSONState.IN_NUMBER_VALUE
                elif char == '{':
                    # Nested object (e.g., inside "parameters")
                    self.state_stack.append(JSONState.EXPECT_KEY_QUOTE)

            elif state == JSONState.IN_STRING_VALUE:
                if char == '"' and partial_json[i-1] != '\\':
                    self.state_stack[-1] = JSONState.EXPECT_COMMA_OR_END

            elif state == JSONState.IN_NUMBER_VALUE:
                # If we hit a space, comma, or brace, the number is done
                if char in (',', '}', ' ', '\n'):
                    self.state_stack[-1] = JSONState.EXPECT_COMMA_OR_END
                    i -= 1  # Re-evaluate this character in the new state

            elif state == JSONState.EXPECT_COMMA_OR_END:
                if char == ',':
                    self.state_stack[-1] = JSONState.EXPECT_KEY_QUOTE
                elif char == '}':
                    self.state_stack.pop()
                    if not self.state_stack:
                        return JSONState.DONE
                    else:
                        self.state_stack[-1] = JSONState.EXPECT_COMMA_OR_END
            i += 1

        return self.state_stack[-1]
