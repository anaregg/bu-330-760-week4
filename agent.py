"""Math agent that solves questions using tools in a ReAct loop."""

import json
import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from calculator import calculate


def configure_api_env() -> None:
    """Prefer existing OS environment variables.
    Only load .env if no supported API key is already set.
    Also map GEMINI_API_KEY -> GOOGLE_API_KEY for compatibility.
    """
    existing_keys = (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    )

    # First, check the system environment variables for an existing key; if none is found, then read the .env file.
    if not any(os.getenv(key) for key in existing_keys):
        load_dotenv()

    # Compatible with GEMINI_API_KEY
    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


configure_api_env()

# Configure your model below. Examples:
#   "google-gla:gemini-2.5-flash"       (needs GOOGLE_API_KEY)
#   "openai:gpt-4o-mini"                (needs OPENAI_API_KEY)
#   "anthropic:claude-sonnet-4-6"    (needs ANTHROPIC_API_KEY)
MODEL = "google-gla:gemini-2.5-flash"

agent = Agent(
    MODEL,
    system_prompt=(
        "You are a helpful assistant. Solve each question step by step. "
        "Use the calculator tool for arithmetic. "
        "Use the product_lookup tool when a question mentions products from the catalog. "
        "If a question cannot be answered with the information given, say so."
    ),
)


@agent.tool_plain
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Examples: "847 * 293", "10000 * (1.07 ** 5)", "23 % 4"
    """
    return calculate(expression)


# TODO: Implement this tool by uncommenting the code below and replacing
# the ... with your implementation. The tool should:
#   1. Read products.json using json.load() (json is already imported above)
#   2. If the product_name is in the catalog, return its price as a string
#   3. If not found, return the list of available product names so the agent
#      can try again with the correct name
#
@agent.tool_plain
def product_lookup(product_name: str) -> str:
    """Look up the price of a product by name.
    Use this when a question asks about product prices from the catalog.
    """
    with open("products.json") as f:
        products = json.load(f)

    if product_name in products:
        return str(products[product_name])

    available_products = ", ".join(products.keys())
    return f"Product not found. Available products: {available_products}"


def load_questions(path: str = "math_questions.md") -> list[str]:
    """Load numbered questions from the markdown file."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and ". " in line[:4]:
                questions.append(line.split(". ", 1)[1])
    return questions


def main():
    questions = load_questions()
    for i, question in enumerate(questions, 1):
        print(f"## Question {i}")
        print(f"> {question}\n")

        result = agent.run_sync(question)

        print("### Trace")
        for message in result.all_messages():
            for part in message.parts:
                kind = part.part_kind
                if kind in ("user-prompt", "system-prompt"):
                    continue
                elif kind == "text":
                    print(f"- **Reason:** {part.content}")
                elif kind == "tool-call":
                    print(f"- **Act:** `{part.tool_name}({part.args})`")
                elif kind == "tool-return":
                    print(f"- **Result:** `{part.content}`")

        print(f"\n**Answer:** {result.output}\n")
        print("---\n")


if __name__ == "__main__":
    main()
