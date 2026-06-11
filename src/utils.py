import json
import sys
from pathlib import Path
from typing import Any


def read_json_file(file_path: str | Path) -> Any:
    """
    Reads and parses a JSON file.

    Exits the program gracefully with an error message if the file is missing,
    contains invalid JSON, or cannot be read.
    """
    path = Path(file_path)

    try:
        # Context managers automatically handle file closure
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except FileNotFoundError:
        print(f"Error: The input file '{path}' was not found.",
              file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: The file '{path}' contains invalid JSON. Details: {e}",
              file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied when trying to read '{path}'.",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading '{path}': {e}",
              file=sys.stderr)
        sys.exit(1)


def write_json_file(data: Any, file_path: str | Path) -> None:
    """
    Writes data to a JSON file.

    Creates the necessary parent directories if they do not exist and
    exits gracefully if there are writing permissions or other errors.
    """
    path = Path(file_path)

    try:
        # Ensure the parent directory (e.g., data/output/)
        # exists before writing
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            # Indent of 4 makes the output highly readable
            json.dump(data, f, indent=4)

    except PermissionError:
        print(f"Error: Permission denied when writing to '{path}'.",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while writing to '{path}': {e}",
              file=sys.stderr)
        sys.exit(1)
