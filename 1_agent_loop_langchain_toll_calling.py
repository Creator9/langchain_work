from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_INTERATIONS = 10
MODEL = "qwen3:1.7b"

# ---- Tools (Langchain @tool decorator) ----

@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.
    """
    print(f" >> Executing tool: get_product_price for product: {product}")
    # FIXED: Changed values from strings (like "$999") to floats so math functions don't crash later
    prices = {
        "laptop": 999.00,
        "keyboard": 699.00,
        "headphones": 199.00
    }
    return prices.get(product.lower(), 0.0)

@tool
def apply_discount(price: float, discount: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """
    print(f" >> Executing tool: apply_discount for price: {price} with discount: {discount}")
    discount_values = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount_percentage = discount_values.get(discount.lower(), 0)
    return round(price * (1 - discount_percentage / 100), 2)

# ---- Agent Loop ----

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    
    llm = init_chat_model(
        model=MODEL,
        model_provider="ollama",
        temperature=0.0,
    )

    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 60)
    
    messages = [
        SystemMessage(content="You are a helpful assistant. "
                      "You have access to a product catalog tool "
                      "and a discount application tool. \n\n"
                      "STRICT RULES - you must follow these exactly:\n"
                      "1. NEVER guess or assume any product price."
                      "You MUST call get_product_price to get the real price.\n"
                      "2. Only call apply_discount AFTER you have received "
                      " a price from get_product_price. Pass the exact price "
                      "returned by get_product_price - do not pass a made-up number.\n "
                      "3. NEVER calculate discounts yourself using math."
                      "Always use the apply_discount tool.\n"
                      "4. NEVER make up a discount tier. Only use bronze, silver, or gold.\n"
                      "ask them which tier to use - do not assume one."),
        HumanMessage(content=question),
    ]
    
    for iteration in range(1, MAX_INTERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Invoke the LLM with the current chat history
        ai_message = llm_with_tools.invoke(messages)
        
        # KEY FIX 1: Append the AI's response to keep track of its tool request intents
        messages.append(ai_message)
        
        tool_calls = ai_message.tool_calls

        # If the LLM has finished calling tools, return the final written text response
        if not tool_calls:
            print("No tool calls detected. Final response:")
            print(ai_message.content)
            return ai_message.content
        
        # KEY FIX 2: Loop through and execute any tools the model requested
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            print(f"Agent requested tool: {tool_name} with args: {tool_args}")
            
            if tool_name in tools_dict:
                # Call the local Python tool function using LangChain's `.invoke()`
                tool_output = tools_dict[tool_name].invoke(tool_args)
                
                # Append the result as a ToolMessage tied back to the original tool ID
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))
            else:
                error_msg = f"Error: Tool {tool_name} not found."
                print(error_msg)
                messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))

    print("Reached max iterations without a final answer.")
    return "Agent timed out."

if __name__ == "__main__":
    print("Welcome to the LangChain Agent Loop Example!")
    result = run_agent("What is the price of a laptop and apply a silver discount?")
    print(f"\nFinal Result: {result}")
