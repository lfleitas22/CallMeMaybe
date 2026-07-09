"""
Main entry point for the function calling tool.
Usage: uv run python -m src [--functions_definition ...]
       [--input ...] [--output ...]
"""
import argparse
import sys
import os
from typing import List, Dict, Any

from pydantic import ValidationError

from src.utils import read_json_file, write_json_file
from src.models import FunctionDefinition, TestPrompt
from src.decoder import ConstrainedDecoder
from llm_sdk import Small_LLM_Model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with default paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Translate natural language prompts into structured "
            "function calls."
        )
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help=(
            "Path to the JSON file containing function definitions."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing natural language prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path where the structured JSON output will be saved.",
    )
    return parser.parse_args()


def build_result(prompt: str, call_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the decoder output into the required output format.
    Ensures only 'prompt', 'name', 'parameters' keys are present.
    """
    name = call_dict.get("name", "")
    params = call_dict.get("parameters", {})
    # If parameters is not a dict (should not happen), default to empty dict
    if not isinstance(params, dict):
        params = {}
    return {"prompt": prompt, "name": name, "parameters": params}


def main() -> None:
    """Main execution flow."""
    args = parse_args()
    if not args.functions_definition or not os.path.isfile(
            args.functions_definition):
        print(f"[!] Error: The functions definition file could not "
              f"be found or is invalid: '{args.functions_definition}'",
              file=sys.stderr)
        sys.exit(1)

    # Check input prompts file
    if not args.input or not os.path.isfile(args.input):
        print(f"[!] Error: The input prompts file could not be found "
              f"or is invalid: '{args.input}'", file=sys.stderr)
        sys.exit(1)

    if not args.output or os.path.isdir(args.output):
        print(f"[!] Error: The output path is invalid or is a "
              f"directory: '{args.output}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading function definitions from: {args.functions_definition}")
    print(f"Loading prompts from: {args.input}")

    # 1. Read input files
    raw_functions = read_json_file(args.functions_definition)
    raw_prompts = read_json_file(args.input)

    # 2. Validate with Pydantic models
    try:
        functions = [FunctionDefinition(**f) for f in raw_functions]
        prompts = [TestPrompt(**p) for p in raw_prompts]
    except ValidationError as e:
        print(
            f"Error: Input files do not match expected schema.\n{e}",
            file=sys.stderr
        )
        sys.exit(1)
    except TypeError:
        print(
            "Error: Input files must contain JSON arrays.",
            file=sys.stderr
        )
        sys.exit(1)

    print("Initializing LLM and Constrained Decoder...")
    try:
        model = Small_LLM_Model()
        decoder = ConstrainedDecoder(
            model=model, available_functions=functions
        )
    except Exception as e:
        print(f"Error initializing model/decoder: {e}", file=sys.stderr)
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    print(f"Processing {len(prompts)} prompts...")

    for test_prompt in prompts:
        print(f" -> Analyzing: '{test_prompt.prompt}'")
        try:
            # Verbose show the token process
            call_dict = decoder.generate_function_call(
                test_prompt.prompt, verbose=True
            )
            results.append(build_result(test_prompt.prompt, call_dict))
        except Exception as e:
            # 1. Provide a clear, traceable error message to
            # the user without crashing
            print(f"    [!] Error processing prompt '{test_prompt.prompt}': "
                  f"{e}", file=sys.stderr)
            # 2. Insert a safe fallback entry that does not
            # invent fake schema keys
            results.append({
                "prompt": test_prompt.prompt,
                "name": "",
                "parameters": {}
            })

    print(f"Writing results to: {args.output}")
    write_json_file(results, args.output)
    print("Execution completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Execution interrupted by user. "
              "Exiting gracefully...", file=sys.stderr)
        sys.exit(130)
