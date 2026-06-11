import argparse
import sys
from pydantic import ValidationError

# Import our custom modules
from src.utils import read_json_file, write_json_file
from src.models import FunctionDefinition, TestPrompt
from src.decoder import ConstrainedDecoder

# Import the provided LLM SDK
from llm_sdk.llm_sdk import Small_LLM_Model


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments according to project requirements."""
    parser = argparse.ArgumentParser(
        description="LLM Function Calling Tool: Translates natural language "
                    "into structured function calls."
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing natural language prompts."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path where the structured JSON output will be saved."
    )

    return parser.parse_args()


def main() -> None:
    """Main execution flow."""
    args = parse_args()

    print(f"Loading function definitions from: {args.functions_definition}")
    print(f"Loading prompts from: {args.input}")

    # 1. Read input files safely
    raw_functions = read_json_file(args.functions_definition)
    raw_prompts = read_json_file(args.input)

    # 2. Validate input data using Pydantic models
    # This guarantees the data structures match what the decoder expects
    try:
        functions = [FunctionDefinition(**f) for f in raw_functions]
        prompts = [TestPrompt(**p) for p in raw_prompts]
    except ValidationError as e:
        print(f"Error: Input files do not match expected schema.\n{e}",
              file=sys.stderr)
        sys.exit(1)
    except TypeError:
        print("Error: Input files must contain JSON arrays "
              "(lists of objects).", file=sys.stderr)
        sys.exit(1)

    print("Initializing LLM and Constrained Decoder...")

    # 3. Initialize the model and decoder
    try:
        model = Small_LLM_Model()
        decoder = ConstrainedDecoder(model=model,
                                     available_functions=functions)
    except Exception as e:
        print(f"Error initializing the model or decoder: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Process each prompt through the decoder
    results = []
    print(f"Processing {len(prompts)} prompts...")

    for test_prompt in prompts:
        print(f" -> Analyzing: '{test_prompt.prompt}'")
        try:
            # Generate the structured dictionary using our state machine
            call_dict = decoder.generate_function_call(test_prompt.prompt)
            results.append(call_dict)
        except Exception as e:
            # Graceful error handling for individual prompt failures
            print(f"    [!] Failed to parse prompt: {e}", file=sys.stderr)
            # Append a safe error object to maintain output file integrity
            results.append({
                "prompt": test_prompt.prompt,
                "name": "error",
                "parameters": {"details": str(e)}
            })

    # 5. Write the final results to disk
    print(f"Writing results to: {args.output}")
    write_json_file(results, args.output)

    print("Execution completed successfully.")


if __name__ == "__main__":
    main()
