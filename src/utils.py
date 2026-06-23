"""
Utility functions for reading and writing JSON files.
"""
import json
import sys
from pathlib import Path
from typing import Any


def read_json_file(file_path: str | Path) -> Any:
    """
    Read and parse a JSON file, exiting gracefully on error.
    """
    path = Path(file_path)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(
            f"Error: The input file '{path}' was not found.",
            file=sys.stderr
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{path}'. {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied reading '{path}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error reading '{path}': {e}", file=sys.stderr)
        sys.exit(1)


def write_json_file(data: Any, file_path: str | Path) -> None:
    """
    Write data to a JSON file, creating parent directories if needed.
    """
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except PermissionError:
        print(
            f"Error: Permission denied writing to '{path}'.",
            file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error writing to '{path}': {e}", file=sys.stderr)
        sys.exit(1)
