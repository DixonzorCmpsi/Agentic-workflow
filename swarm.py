import json
import datetime
import sys
import os
import time
from dotenv import load_dotenv
from colorama import Fore, Style, init
from tools import registry
from llm_client import LLMClient
from memory import VectorMemory

init(autoreset=True)
load_dotenv() # <--- Load the .env file immediately

# --- DYNAMIC CONFIGURATION ---
# We now pull these from the environment, defaulting to local if missing
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2")

print(f"{Fore.CYAN}[SYSTEM] Connecting to {PROVIDER.upper()} ({MODEL_NAME})...{Style.RESET_ALL}")
llm = LLMClient(provider=PROVIDER, model_name=MODEL_NAME)

# --- PERSISTENT MEMORY ---
print(f"{Fore.CYAN}[SYSTEM] Initializing Vector Memory...{Style.RESET_ALL}")
memory = VectorMemory()

def get_system_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")

# --- RLM EVALUATOR (Recursive Language Memory) ---
def rlm_evaluator(original_query, context):
    """
    Metacognitive check: Ask the LLM if the retrieved memory context is sufficient.
    Returns (is_sufficient: bool, refined_query: str or None)
    """
    prompt = f"""You are evaluating memory search results for relevance.

ORIGINAL QUERY: {original_query}

RETRIEVED CONTEXT:
{context}

TASK: Determine if this context contains enough information to answer the query.

RESPOND WITH JSON ONLY:
{{"sufficient": true/false, "refined_query": "new search query if needed, else null"}}"""

    try:
        response = llm.chat([{"role": "user", "content": prompt}], format="json")
        content = response.get('content', '').strip()
        if "```" in content:
            content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return data.get('sufficient', False), data.get('refined_query')
    except:
        return True, None  # Default to sufficient if evaluation fails

# --- ROBUST ARGUMENT CLEANER ---
def nuclear_argument_cleaner(args):
    """Recursively hunts for the actual string value inside nested dictionaries."""
    if isinstance(args, str):
        try: args = json.loads(args)
        except: return args 
    if isinstance(args, dict):
        cleaned = {}
        for key, value in args.items():
            cleaned[key] = _extract_string(value)
        return cleaned
    return args

def _extract_string(data):
    if isinstance(data, str): return data
    if isinstance(data, list): return _extract_string(data[0]) if data else ""
    if isinstance(data, dict):
        for k in ['content', 'value', 'query', 'argument', 'expression']:
            if k in data: return _extract_string(data[k])
        for v in data.values():
            found = _extract_string(v)
            if found: return found
    return str(data)

class SwarmWorker:
    def __init__(self, name, role_prompt, tools_enabled=False):
        self.name = name
        self.base_prompt = role_prompt
        self.tools = registry.schemas if tools_enabled else None

    def run(self, task_description, is_system_task=False):
        current_date = get_system_date()
        system_prompt = f"You are {self.name}. CURRENT DATE: {current_date}. {self.base_prompt}"
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': task_description}
        ]
        
        # 1. First Call
        msg = llm.chat(messages=messages, tools=self.tools)
        
        # 2. Tool Execution Loop
        if msg.get('tool_calls'):
            print(f"{Fore.YELLOW}[{self.name}] Tool Triggered.{Style.RESET_ALL}")
            tool_outputs = []
            
            # Append the Assistant's "I want to call a tool" message
            messages.append(msg)

            for tool in msg['tool_calls']:
                fn_name = tool['function']['name']
                args = tool['function']['arguments']
                clean_args = nuclear_argument_cleaner(args)

                print(f"{Fore.CYAN}[TOOL] {fn_name}({clean_args}){Style.RESET_ALL}")
                try:
                    result = registry.execute(fn_name, clean_args)
                    result_str = str(result)
                    tool_outputs.append(result_str)
                    
                    # Gemini/OpenAI expect 'tool' role for results
                    messages.append({
                        'role': 'tool', 
                        'name': fn_name, 
                        'content': result_str,
                        'tool_call_id': tool.get('id') # Important for OpenAI, ignored by others
                    })
                except Exception as e:
                    error_msg = f"Error: {e}"
                    tool_outputs.append(error_msg)
                    messages.append({
                        'role': 'tool', 
                        'name': fn_name, 
                        'content': error_msg
                    })
            
            if is_system_task:
                return "\n".join(tool_outputs)

            # 3. Final Synthesis
            final_res = llm.chat(messages=messages)
            return final_res['content']
        
        return msg['content']

class SwarmManager:
    def __init__(self):
        self.workers = {}
        # Will be set by bootstrap - no hardcoded default
        self.knowledge_cutoff = None
        # Track conversation for memory storage
        self.conversation_history = []

    def add_worker(self, worker):
        self.workers[worker.name] = worker

    def save_interaction(self, user_query, response):
        """Store the interaction in vector memory for future recall."""
        interaction = f"User: {user_query}\nAssistant: {response}"
        metadata = {
            "type": "conversation",
            "timestamp": get_system_date(),
            "query_preview": user_query[:100]
        }
        memory.add(interaction, metadata)
        self.conversation_history.append({"user": user_query, "assistant": response})

    def add_worker(self, worker):
        self.workers[worker.name] = worker

    def bootstrap_knowledge_cutoff(self):
        """
        LLM-as-Parser approach: Let the LLM semantically understand search results
        and extract the correct cutoff date. No regex needed.
        """
        actual_model = llm.model_name
        print(f"{Fore.BLUE}[SYSTEM] Bootstrapping: Detecting Knowledge Cutoff for {actual_model}...{Style.RESET_ALL}")

        # Step 1: Search for cutoff information
        search_query = f"{actual_model} official knowledge cutoff date training data"
        print(f"{Fore.CYAN}[SEARCH] {search_query}{Style.RESET_ALL}")
        
        try:
            raw_search_results = str(registry.execute("brave_search", {"query": search_query}))
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Search failed: {e}{Style.RESET_ALL}")
            raw_search_results = ""
        
        # If first search didn't find much, try a backup query
        if not raw_search_results or "No results" in raw_search_results:
            backup_query = f"Google {actual_model} model knowledge cutoff"
            print(f"{Fore.CYAN}[SEARCH] Backup: {backup_query}{Style.RESET_ALL}")
            try:
                time.sleep(1)
                raw_search_results = str(registry.execute("brave_search", {"query": backup_query}))
            except:
                pass
        
        print(f"{Fore.CYAN}[DEBUG] Raw search results:\n{raw_search_results[:600]}...{Style.RESET_ALL}")
        
        # Step 2: LLM-as-Parser - Ask the LLM to semantically extract the date
        extraction_prompt = f"""You are a precise data extraction assistant.

TASK: Analyze the search results below and identify the official knowledge cutoff date for the AI model "{actual_model}".

SEARCH RESULTS:
{raw_search_results}

INSTRUCTIONS:
1. Look for phrases like "knowledge cutoff", "training data cutoff", "trained on data up to", "cutoff date"
2. The cutoff date is when the model's training data ends - NOT release dates or deprecation dates
3. If multiple dates appear, choose the one that specifically refers to knowledge/training cutoff
4. If the search mentions "August 2024" as a cutoff, that means 2024-08-01

RESPOND WITH ONLY A JSON OBJECT:
{{"knowledge_cutoff": "YYYY-MM-DD", "confidence": "high/medium/low", "source": "brief quote from results"}}

If you cannot find a clear cutoff date, respond:
{{"knowledge_cutoff": null, "confidence": "none", "source": "no cutoff date found"}}"""

        # Step 3: Get LLM's semantic analysis
        for attempt in range(3):
            try:
                response = llm.chat(
                    messages=[{'role': 'user', 'content': extraction_prompt}],
                    format="json"
                )
                content = response.get('content', '').strip()
                
                # Clean markdown if present
                if "```" in content:
                    content = content.replace("```json", "").replace("```", "").strip()
                
                print(f"{Fore.CYAN}[LLM RESPONSE] {content}{Style.RESET_ALL}")
                
                # Step 4: Parse the structured JSON response
                data = json.loads(content)
                extracted_date = data.get('knowledge_cutoff')
                confidence = data.get('confidence', 'unknown')
                source = data.get('source', '')
                
                if extracted_date and extracted_date != "null":
                    self.knowledge_cutoff = extracted_date
                    print(f"{Fore.GREEN}[SYSTEM] ✅ Cutoff Calibration Success!{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}   Date: {self.knowledge_cutoff}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}   Confidence: {confidence}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}   Source: {source[:100]}...{Style.RESET_ALL}")
                    return
                else:
                    print(f"{Fore.YELLOW}[SYSTEM] LLM could not find cutoff date in results{Style.RESET_ALL}")
                    break
                    
            except json.JSONDecodeError as e:
                print(f"{Fore.YELLOW}[SYSTEM] JSON parse error: {e}. Retrying...{Style.RESET_ALL}")
                continue
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"{Fore.YELLOW}[SYSTEM] Rate limited, waiting 5s...{Style.RESET_ALL}")
                    time.sleep(5)
                else:
                    print(f"{Fore.RED}[ERROR] {e}{Style.RESET_ALL}")
                    break
        
        # Step 5: Fallback - estimate based on current date (6 months ago)
        from datetime import datetime, timedelta
        fallback_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-01")
        self.knowledge_cutoff = fallback_date
        print(f"{Fore.YELLOW}[SYSTEM] ⚠️ Could not determine cutoff from search.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[SYSTEM] Using estimated fallback: {self.knowledge_cutoff}{Style.RESET_ALL}")

    def delegate(self, user_query):
        print(f"{Fore.MAGENTA}[MANAGER] classifying query...{Style.RESET_ALL}")
        worker_list = ", ".join(self.workers.keys())
        
        system_prompt = f"""
        You are the Swarm Manager. Current Date: {get_system_date()}. 
        Model Knowledge Cutoff: {self.knowledge_cutoff}.
        
        ROUTING LOGIC (choose the BEST worker):
        1. Question about previous conversations, user preferences, "remember", "what did I say"? -> "Memory"
        2. Complex multi-step task needing breakdown? -> "Planner"
        3. Task mentions dependencies, blockers, prerequisites? -> "DependencyAnalyst"  
        4. Need real-time info AFTER {self.knowledge_cutoff}? -> "Researcher"
        5. Math, coding, logic, or pre-cutoff knowledge? -> "Analyst"
        
        AVAILABLE WORKERS: {worker_list}
        OUTPUT FORMAT (JSON ONLY):
        {{"reasoning": "why this worker", "worker": "Worker_Name"}}
        """

        try:
            fmt = "json" if PROVIDER in ["openai", "gemini"] else None
            
            response = llm.chat(messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_query}], format=fmt)
            content = response['content']
            
            if "```" in content:
                content = content.replace("```json", "").replace("```", "").strip()
                
            data = json.loads(content)
            worker_name = data.get('worker')
            print(f"{Fore.MAGENTA}[LOGIC] {data.get('reasoning')}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}[ROUTE] {worker_name}{Style.RESET_ALL}")

            if worker_name in self.workers:
                result = self.workers[worker_name].run(user_query)
                # Save interaction to memory (skip for memory queries to avoid recursion)
                if worker_name != "Memory":
                    self.save_interaction(user_query, result)
                return result
            return f"Error: Manager selected unknown worker '{worker_name}'."
        except Exception as e:
            return f"Routing Failure: {e}"
    
    def execute_plan(self, plan_steps):
        """Execute a multi-step plan created by the Planner."""
        results = []
        for i, step in enumerate(plan_steps):
            print(f"{Fore.BLUE}[PLAN] Step {i+1}: {step}{Style.RESET_ALL}")
            result = self.delegate(step)
            results.append(f"Step {i+1}: {result}")
        return "\n\n".join(results)

# --- WORKER DEFINITIONS ---
# Memory Agent: Handles recall of past conversations using RLM
MEMORY_PROMPT = """You are a Memory Agent with access to the user's conversation history.
You use semantic search over stored memories to answer questions about past interactions.

When responding:
1. Only cite information you found in the provided memory context
2. If no relevant memories exist, say "I don't have any memories about that"
3. Be specific about what you remember vs what you're inferring"""

class MemoryWorker(SwarmWorker):
    """Special worker that uses RLM (Recursive Language Memory) for recall."""
    
    def __init__(self, name, role_prompt):
        super().__init__(name, role_prompt, tools_enabled=False)
    
    def run(self, task_description, is_system_task=False):
        print(f"{Fore.CYAN}[MEMORY] Searching vector database...{Style.RESET_ALL}")
        
        # Use RLM for recursive memory search
        memories = memory.recursive_recall(
            query=task_description,
            llm_check_fn=rlm_evaluator,
            max_depth=2
        )
        
        if not memories:
            return "I don't have any stored memories related to that query."
        
        # Format memories for context
        memory_context = "\n---\n".join(memories)
        print(f"{Fore.CYAN}[MEMORY] Found {len(memories)} relevant memories{Style.RESET_ALL}")
        
        # Ask LLM to synthesize an answer from memories
        synthesis_prompt = f"""You are answering based on stored conversation memories.

RETRIEVED MEMORIES:
{memory_context}

USER QUESTION: {task_description}

Based ONLY on the memories above, provide a helpful answer. If the memories don't contain 
relevant information, say so. Don't make up information not present in the memories."""

        response = llm.chat([
            {"role": "system", "content": self.base_prompt},
            {"role": "user", "content": synthesis_prompt}
        ])
        return response['content']

# Planner: Breaks down complex tasks into actionable steps
PLANNER_PROMPT = """You are a Planning Agent. Your job is to break down complex tasks into clear, actionable steps.

When given a task:
1. Identify all sub-tasks needed
2. Order them logically (dependencies first)
3. Output a numbered plan

FORMAT:
PLAN:
1. [First step]
2. [Second step]
...

Keep steps atomic and clear. Each step should be executable by another agent."""

# Dependency Analyst: Analyzes blockers, prerequisites, and dependencies
DEPENDENCY_PROMPT = """You are a Dependency Analysis Agent. You analyze tasks for:
- Prerequisites (what must be done first)
- Blockers (what's preventing progress)  
- Dependencies (external factors needed)
- Risks (what could go wrong)

Use the brave_search tool to check current status of external dependencies or verify information.

OUTPUT FORMAT:
PREREQUISITES:
- [List what must be done first]

BLOCKERS:
- [List any blockers found]

DEPENDENCIES:
- [List external dependencies]

RISKS:
- [List potential risks]

RECOMMENDATION:
[Your advice on how to proceed]"""

# Researcher: Real-time information gathering
RESEARCHER_PROMPT = """You are a Research Agent. You MUST use the brave_search tool to find real-time information.
NEVER make up facts. If you can't find information, say so.
Always cite what you found in search results."""

# Analyst: Logic, math, coding, general knowledge
ANALYST_PROMPT = """You are a Logic & Analysis Engine. You handle:
- Mathematical calculations
- Code review and generation
- Logical reasoning
- Historical/factual questions from your training data

Be precise and show your work."""

# --- INITIALIZATION ---
boss = SwarmManager()
boss.add_worker(MemoryWorker("Memory", MEMORY_PROMPT))  # Memory agent with RLM
boss.add_worker(SwarmWorker("Planner", PLANNER_PROMPT, tools_enabled=False))
boss.add_worker(SwarmWorker("DependencyAnalyst", DEPENDENCY_PROMPT, tools_enabled=True))
boss.add_worker(SwarmWorker("Researcher", RESEARCHER_PROMPT, tools_enabled=True))
boss.add_worker(SwarmWorker("Analyst", ANALYST_PROMPT, tools_enabled=False))

if __name__ == "__main__":
    print(f"{Fore.GREEN}--- MULTI-AGENT SWARM v2.0 (with Memory) ---{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Workers: Memory | Planner | DependencyAnalyst | Researcher | Analyst{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Memory: {memory.collection.count()} stored interactions{Style.RESET_ALL}")
    boss.bootstrap_knowledge_cutoff()
    
    while True:
        try:
            q = input(f"\n{Fore.WHITE}Task: {Style.RESET_ALL}")
            if q.lower() == 'q': break
            print(f"\n{Fore.GREEN}>>> REPORT:\n{boss.delegate(q)}{Style.RESET_ALL}")
        except KeyboardInterrupt: break