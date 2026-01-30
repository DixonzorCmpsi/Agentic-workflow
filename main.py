import ollama
import json
from colorama import Fore, Style, init
from tools import registry
from memory import memory  # Import our new Vector Brain

init(autoreset=True)

MODEL = "llama3.2"

class Agent:
    def __init__(self):
        self.history = [] # Short-term working memory (RAM)

    def _sanitize_message(self, msg):
        if isinstance(msg, dict): return msg
        if hasattr(msg, 'model_dump'): return msg.model_dump()
        return dict(msg)

    def save_interaction(self, user, assistant):
        """Saves the Q&A pair to the Vector DB."""
        # We store the pair so the context is preserved
        blob = f"User: {user}\nAssistant: {assistant}"
        memory.add(blob, metadata={"type": "chat_history"})

    def rlm_evaluator(self, original_query, retrieved_context):
        """
        The Metacognitive Function for RLM.
        Returns: (bool is_sufficient, str new_query)
        """
        eval_prompt = f"""
        Analyze if the retrieved context answers the query.
        Query: "{original_query}"
        Context: "{retrieved_context}"
        
        OUTPUT FORMAT:
        SUFFICIENT: [YES/NO]
        REFINEMENT: [New Search Query if NO, otherwise "None"]
        """
        
        response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': eval_prompt}])
        content = response['message']['content']
        
        if "SUFFICIENT: YES" in content.upper():
            return True, None
        else:
            # Extract refinement
            try:
                new_q = content.split("REFINEMENT:")[1].strip()
                return False, new_q
            except:
                return True, None # Fallback

    def route_query(self, user_input):
        routing_prompt = [
            {'role': 'system', 'content': """
            Analyze the input.
            - If it asks for personal info ("who am I", "my job") or past chat details -> output "MEMORY"
            - If it asks for external facts (news, code, math) -> output "TOOL"
            - If it is just "hello" or "thanks" -> output "CHAT"
            OUTPUT ONLY ONE WORD.
            """},
            {'role': 'user', 'content': user_input}
        ]
        try:
            res = ollama.chat(model=MODEL, messages=routing_prompt)
            decision = res['message']['content'].strip().upper()
            if "MEMORY" in decision: return "MEMORY"
            if "TOOL" in decision: return "TOOL"
            return "CHAT"
        except:
            return "CHAT"

    def chat(self, user_input):
        print(f"{Fore.MAGENTA}Thinking (Routing)...{Style.RESET_ALL}")
        mode = self.route_query(user_input)
        
        context_str = ""
        
        # --- RLM MEMORY RETRIEVAL ---
        if mode == "MEMORY":
            print(f"{Fore.BLUE}[DECISION] RLM Vector Search Enabled.{Style.RESET_ALL}")
            # We pass our evaluator function to the memory module
            results = memory.recursive_recall(
                user_input, 
                llm_check_fn=self.rlm_evaluator,
                max_depth=2
            )
            context_str = "\n".join(results)
            print(f"{Fore.CYAN}[RLM] Retrieved {len(results)} relevant memories.{Style.RESET_ALL}")

        # Construct System Prompt based on Mode
        system_prompt = "You are a helpful AI."
        if mode == "MEMORY":
            system_prompt = f"Answer using this retrieved memory:\n{context_str}\nIf the memory is empty, say you don't know."
        elif mode == "TOOL":
            system_prompt = "You are a researcher. Use tools to find answers."

        # Prepare Run Messages
        run_messages = [{'role': 'system', 'content': system_prompt}] + self.history + [{'role': 'user', 'content': user_input}]

        if mode == "TOOL":
            print(f"{Fore.YELLOW}[DECISION] Tool Mode.{Style.RESET_ALL}")
            response = ollama.chat(model=MODEL, messages=run_messages, tools=registry.schemas)
            message = response['message']
            
            # Tool Loop (Same as before)
            if message.get('tool_calls'):
                for tool in message['tool_calls']:
                    fn_name = tool['function']['name']
                    args = tool['function']['arguments']
                    clean_args = {} # (Insert your Argument Cleaner logic here if desired)
                    # Simplified for brevity:
                    for k,v in args.items(): clean_args[k] = v 
                    
                    print(f"{Fore.YELLOW}[ACTION] {fn_name}({clean_args}){Style.RESET_ALL}")
                    result = registry.execute(fn_name, clean_args)
                    run_messages.append(message)
                    run_messages.append({'role': 'tool', 'content': str(result)})
                
                final = ollama.chat(model=MODEL, messages=run_messages)
                content = final['message']['content']
            else:
                content = message['content']
        else:
            # Direct Chat or Memory Response
            response = ollama.chat(model=MODEL, messages=run_messages)
            content = response['message']['content']

        # --- SAVE TO LONG TERM MEMORY ---
        self.save_interaction(user_input, content)
        
        # Update Short Term Memory (keep last 4 turns)
        self.history.append({'role': 'user', 'content': user_input})
        self.history.append({'role': 'assistant', 'content': content})
        if len(self.history) > 4: self.history = self.history[-4:]
        
        return content

if __name__ == "__main__":
    print(f"{Fore.GREEN}--- AGENT (Vector RLM) INITIALIZED ---{Style.RESET_ALL}")
    agent = Agent()
    while True:
        try:
            q = input(f"\n{Fore.WHITE}Query (or 'q'): {Style.RESET_ALL}")
            if q.lower() == 'q': break
            print(f"\n{Fore.GREEN}>>> {agent.chat(q)}{Style.RESET_ALL}")
        except KeyboardInterrupt: break