import os
import json
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, provider="ollama", model_name="llama3.2"):
        self.provider = provider.lower()
        self.model_name = model_name
        self.client = self._initialize_client()

        # Normalize model names for providers that require specific naming.
        # This prevents common 404s like "model not found" when switching providers
        # but leaving the model set to a local Ollama model.
        self.model_name = self._normalize_model_name(self.model_name)

    def _normalize_model_name(self, model_name: str) -> str:
        if not model_name:
            model_name = ""

        # Gemini model names are not compatible with Ollama model names.
        # If the user forgot to set LLM_MODEL when switching to Gemini,
        # default to a commonly available Gemini model.
        if self.provider == "gemini":
            candidate = str(model_name).strip()
            if not candidate or candidate.lower().startswith("llama") or candidate.lower().startswith("mistral"):
                # "gemini-pro" remains one of the most widely available defaults.
                return "gemini-pro"
            return candidate

        return model_name

    def _initialize_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == "gemini":
            # --- NEW SDK MIGRATION (Fixes v1beta 404 error) ---
            try:
                from google import genai
            except Exception as e:
                raise ImportError(
                    "Gemini provider selected, but the 'google-genai' SDK is not available in this Python environment. "
                    "Install it with: pip install google-genai"
                ) from e
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
            return genai.Client(api_key=api_key)
        else:
            import ollama
            return ollama

    def chat(self, messages, tools=None, format=None):
        """Universal Chat Wrapper."""
        try:
            # --- OPENAI ---
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    response_format={"type": "json_object"} if format == "json" else None
                )
                msg = response.choices[0].message
                return {
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "function": {
                                "name": tc.function.name,
                                "arguments": json.loads(tc.function.arguments)
                            }
                        } for tc in msg.tool_calls
                    ] if msg.tool_calls else None
                }
            
            # --- GEMINI (Google GenAI V1) ---
            elif self.provider == "gemini":
                from google.genai import types
                
                # 1. Separate System Prompt from History
                system_instruction = None
                contents = []
                
                for m in messages:
                    if m.get('role') == 'system':
                        system_instruction = m.get('content', '')
                    elif m.get('role') == 'tool':
                        # Gemini expects function response parts
                        contents.append(types.Content(
                            role="user",
                            parts=[types.Part.from_function_response(
                                name=m.get('name', 'tool'),
                                response={"result": m.get('content', '')}
                            )]
                        ))
                    elif m.get('tool_calls'):
                        # This is an assistant message requesting tool calls - add as model turn
                        # Gemini needs to see the function call it made
                        func_call_parts = []
                        for tc in m['tool_calls']:
                            func_call_parts.append(types.Part.from_function_call(
                                name=tc['function']['name'],
                                args=tc['function'].get('arguments', {})
                            ))
                        contents.append(types.Content(role="model", parts=func_call_parts))
                    elif m.get('role') in ['user', 'assistant']:
                        # Map 'assistant' -> 'model'
                        role = "model" if m.get('role') == 'assistant' else "user"
                        content_text = m.get('content', '')
                        if content_text:
                            contents.append(types.Content(
                                role=role,
                                parts=[types.Part.from_text(text=str(content_text))]
                            ))
                    # Skip any malformed messages

                # 2. Configure Generation
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0
                )
                
                # 3. Handle JSON Mode
                if format == "json":
                    config.response_mime_type = "application/json"
                
                # 4. Convert tools to Gemini format
                if tools:
                    gemini_tools = []
                    for tool in tools:
                        func = tool.get('function', {})
                        params = func.get('parameters', {})
                        properties = params.get('properties', {})
                        required = params.get('required', [])
                        
                        # Build Gemini-compatible schema
                        gemini_properties = {}
                        for prop_name, prop_def in properties.items():
                            gemini_properties[prop_name] = types.Schema(
                                type=types.Type.STRING if prop_def.get('type') == 'string' else types.Type.NUMBER,
                                description=prop_def.get('description', '')
                            )
                        
                        gemini_func = types.FunctionDeclaration(
                            name=func.get('name'),
                            description=func.get('description', ''),
                            parameters=types.Schema(
                                type=types.Type.OBJECT,
                                properties=gemini_properties,
                                required=required
                            )
                        )
                        gemini_tools.append(gemini_func)
                    
                    config.tools = [types.Tool(function_declarations=gemini_tools)]

                # 5. Generate with fallback
                def _is_gemini_not_found(err: Exception) -> bool:
                    t = str(err)
                    return (
                        "404" in t
                        or "NOT_FOUND" in t
                        or "is not found for API version" in t
                        or "not found" in t.lower()
                    )

                def _generate_with_model(model: str):
                    return self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )

                try:
                    response = _generate_with_model(self.model_name)
                except Exception as e:
                    # Common failure when users switch providers but the chosen model isn't available
                    if _is_gemini_not_found(e):
                        fallback_models = [
                            "gemini-2.0-flash",
                            "gemini-1.5-flash",
                            "gemini-1.5-pro",
                            "gemini-pro",
                        ]
                        tried = {self.model_name}
                        last_err = e

                        for candidate in fallback_models:
                            if candidate in tried:
                                continue
                            tried.add(candidate)
                            try:
                                response = _generate_with_model(candidate)
                                print(f"[LLM] Gemini model '{self.model_name}' unavailable; switching to '{candidate}'.")
                                self.model_name = candidate
                                break
                            except Exception as e2:
                                last_err = e2
                                if not _is_gemini_not_found(e2):
                                    raise
                        else:
                            raise last_err
                    else:
                        raise
                
                # 6. Extract tool calls from response
                tool_calls = None
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            if tool_calls is None:
                                tool_calls = []
                            tool_calls.append({
                                "id": f"call_{fc.name}",
                                "function": {
                                    "name": fc.name,
                                    "arguments": dict(fc.args) if fc.args else {}
                                }
                            })
                
                # 7. Return in standardized format
                return {
                    "content": response.text if not tool_calls else None,
                    "tool_calls": tool_calls
                }

            # --- OLLAMA (Local) ---
            else:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    format=format
                )
                return response['message']

        except Exception as e:
            err_text = str(e)

            # Make the most common 404 actionable.
            if self.provider == "gemini" and ("404" in err_text or "NOT_FOUND" in err_text or "not found" in err_text.lower()):
                err_text = (
                    f"{err_text}\n"
                    "Gemini returned 404 (model not found). Check that LLM_MODEL is a Gemini model, e.g. "
                    "'gemini-1.5-flash' or 'gemini-1.5-pro'. If you switched providers from Ollama, make sure you didn't leave LLM_MODEL='llama3.2'."
                )

            print(f"LLM Error: {err_text}")
            # Return a safe error so the pipeline doesn't crash
            return {"content": json.dumps({"error": err_text}), "tool_calls": None}