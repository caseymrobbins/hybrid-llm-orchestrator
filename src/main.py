# src/main.py

import argparse
import asyncio
from orchestrator import Orchestrator

async def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Hybrid LLM Orchestrator")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The user query to process through the workflow."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs",
        help="Path to the configuration directory."
    )
    args = parser.parse_args()

    try:
        orchestrator = Orchestrator(config_path=args.config_path)
        final_response = await orchestrator.execute_workflow(query=args.query)
        
        print("\n" + "="*20 + " FINAL RESPONSE " + "="*20)
        print(final_response)
        print("="*58)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # In a production app, log the full traceback here

if __name__ == "__main__":
    asyncio.run(main())


