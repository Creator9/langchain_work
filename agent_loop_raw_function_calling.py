import json
from dotenv import load_dotenv
import ollama
from langsmith import traceable, trace  # Import 'trace' context manager

load_dotenv()

MAX_INTERATIONS = 10
MODEL = "qwen3:1.7b"

# ---- Tools ----

@traceable(name="Tool: Get Product Price")
def get_product_price(product: str) -> float:
    print(f" >> Executing tool: get_product_price for product: {product}")
    prices = {
        "laptop": 999.00,
        "keyboard": 699.00,
        "headphones": 199.00
    }
    return prices.get(product.lower(), 0.0)

@traceable(name="Tool: Apply Discount")
def apply_discount(price: float, discount: str) -> float:
    print(f" >> Executing tool: apply_discount for price: {price} with discount: {discount}")
    discount_values = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount_percentage = discount_values.get(discount.lower(), 0)
    return round(price * (1 - discount_percentage / 100), 2)

tools_dict = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

ollama_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "The name of the product"}
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The base price of the item"},
                    "discount": {"type": "string", "description": "Available tiers: bronze, silver, gold"}
                },
                "required": ["price", "discount"]
            }
        }
    }
]

# ---- Agent Loop ----

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. You have access to a product catalog tool "
                "and a discount application tool. \n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. You MUST call get_product_price to get the real price.\n"
                "2. Only call apply_discount AFTER you have received a price from get_product_price. "
                "Pass the exact price returned by get_product_price - do not pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.\n"
                "4. NEVER make up a discount tier. Only use bronze, silver, or gold. Ask them which tier to use - do not assume one."
            )
        },
        {"role": "user", "content": question}
    ]
    
    for iteration in range(1, MAX_INTERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # MANUAL APPROACH: Wrap the raw call inside a `trace()` context block
        # specifying the run type as "llm" so LangSmith treats it properly.
        with trace(
            name=f"Ollama Chat Generation (Iteration {iteration})", 
            run_type="llm",
            inputs={"messages": messages, "model": MODEL}
        ) as run:
            
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=ollama_tools,
                options={"temperature": 0.0}
            )
            
            # Manually pass the output payload back to the LangSmith block
            run.end(outputs={"response": response})
        
        # Append the assistant's message to history
        messages.append(response["message"])
        
        tool_calls = response["message"].get("tool_calls", [])

        if not tool_calls:
            print("No tool calls detected. Final response:")
            print(response["message"]["content"])
            return response["message"]["content"]
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]
            
            print(f"Agent requested tool: {tool_name} with args: {tool_args}")
            
            if tool_name in tools_dict:
                tool_output = tools_dict[tool_name](**tool_args)
                messages.append({
                    "role": "tool",
                    "content": str(tool_output),
                    "name": tool_name
                })
            else:
                error_msg = f"Error: Tool {tool_name} not found."
                print(error_msg)
                messages.append({
                    "role": "tool",
                    "content": error_msg,
                    "name": tool_name
                })

    print("Reached max iterations without a final answer.")
    return "Agent timed out."

if __name__ == "__main__":
    print("Welcome to the Ollama Native Agent Loop!")
    result = run_agent("What is the price of a laptop and apply a silver discount?")
    print(f"\nFinal Result: {result}")
