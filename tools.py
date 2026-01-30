import inspect
import requests
import os
import dotenv

dotenv.load_dotenv()

# --- CONFIGURATION ---
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, func):
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "No description."
        
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        for name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation == int: param_type = "integer"
            
            schema["function"]["parameters"]["properties"][name] = {
                "type": param_type,
                "description": f"The {name} argument"
            }
            schema["function"]["parameters"]["required"].append(name)

        self._tools[func.__name__] = {
            "func": func,
            "schema": schema
        }
        return func

    @property
    def schemas(self):
        return [t["schema"] for t in self._tools.values()]

    def execute(self, name, args):
        if name in self._tools:
            return self._tools[name]["func"](**args)
        return f"Error: Tool '{name}' not found."

registry = ToolRegistry()
tool = registry.register

# --- TOOLS ---

@tool
def brave_search(query: str) -> str:
    """
    Use this tool ONLY for retrieving:
    1. Real-time news (events happening now or recently).
    2. Specific technical specifications.
    3. Facts that likely changed after 2023.
    """
    print(f"   [DEBUG] 🔎 Searching: '{query}'")
    
    # 1. Try Brave API
    if BRAVE_API_KEY:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
        try:
            response = requests.get(url, headers=headers, params={"q": query, "count": 3}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = data.get("web", {}).get("results", [])
                if results:
                    return "\n".join([f"- {r.get('title')}: {r.get('description')}" for r in results])
            else:
                body_preview = (response.text or "").strip().replace("\n", " ")[:200]
                print(f"   [WARN] Brave HTTP {response.status_code}: {body_preview}")
        except Exception as e:
            print(f"   [WARN] Brave failed: {e}")

    # 2. Fallback to DuckDuckGo (Free)
    try:
        # DuckDuckGo search lib was renamed from duckduckgo_search -> ddgs.
        # Import lazily so this file still works without that dependency.
        try:
            from ddgs import DDGS  # type: ignore
        except Exception:
            from duckduckgo_search import DDGS  # type: ignore

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([f"- {r.get('title')}: {r.get('body')}" for r in results])
    except Exception:
        pass

    return "Observation: No results found."

@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.
    Input must be a valid Python math expression (e.g., "125 * 4.5", "2**10", "sqrt(144)").
    Use this for any math more complex than basic arithmetic.
    """
    import math
    print(f"   [DEBUG] 🔢 Calculating: {expression}")
    try:
        # Provide math functions in the eval context
        allowed = {"__builtins__": {}, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, 
                   "tan": math.tan, "pi": math.pi, "e": math.e, "log": math.log, "abs": abs,
                   "pow": pow, "round": round, "min": min, "max": max}
        result = eval(expression, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation Error: {e}"

@tool  
def get_current_datetime() -> str:
    """
    Get the current date and time. Use this when user asks about today's date, 
    current time, or needs to know what day it is.
    """
    from datetime import datetime
    now = datetime.now()
    return f"Current datetime: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"

# --- OPTIONAL: Project Tracking Integration ---
# Uncomment and configure if you use Jira, GitHub Issues, Linear, etc.
#
# @tool
# def check_jira_ticket(ticket_id: str) -> str:
#     """Check status of a Jira ticket."""
#     # Your Jira integration code here
#     pass