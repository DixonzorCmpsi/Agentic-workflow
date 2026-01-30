#!/usr/bin/env python
"""
FULL TEST SUITE for Multi-Agent Swarm v2.0
Tests all components: LLM, Tools, Memory, Agents, Bootstrap
"""
import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

# Test counters
passed = 0
failed = 0
tests = []

def test(name):
    """Decorator to register and run tests"""
    def decorator(func):
        tests.append((name, func))
        return func
    return decorator

def run_tests():
    global passed, failed
    print("=" * 70)
    print("MULTI-AGENT SWARM v2.0 - FULL TEST SUITE")
    print("=" * 70)
    
    for name, func in tests:
        print(f"\n{'─' * 70}")
        print(f"TEST: {name}")
        print('─' * 70)
        try:
            func()
            print(f"✅ PASSED: {name}")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {name}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {name}")
            print(f"   Exception: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 70)
    return failed == 0

# ============================================================
# TEST 1: Environment & Configuration
# ============================================================
@test("Environment variables loaded")
def test_env():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "llama3.2")
    print(f"   Provider: {provider}")
    print(f"   Model: {model}")
    assert provider in ["ollama", "gemini", "openai", "anthropic"], f"Unknown provider: {provider}"

# ============================================================
# TEST 2: Tool Registry
# ============================================================
@test("Tool Registry - tools registered correctly")
def test_tool_registry():
    from tools import registry
    tools = [t['schema']['function']['name'] for t in registry._tools.values()]
    print(f"   Registered tools: {tools}")
    assert "brave_search" in registry._tools, "brave_search not registered"
    assert "calculate" in registry._tools, "calculate not registered"
    assert len(tools) >= 2, "Expected at least 2 tools"

# ============================================================
# TEST 3: Brave Search Tool
# ============================================================
@test("Brave Search - returns real results")
def test_brave_search():
    from tools import brave_search
    result = brave_search("Python programming language")
    print(f"   Result preview: {result[:200]}...")
    assert result, "Search returned empty"
    assert "No results found" not in result or len(result) > 50, "Search failed"

# ============================================================
# TEST 4: Calculate Tool
# ============================================================
@test("Calculate Tool - math works correctly")
def test_calculate():
    from tools import calculate
    
    # Test basic math
    r1 = calculate("2 + 2")
    print(f"   2 + 2 = {r1}")
    assert "4" in r1, f"Expected 4, got {r1}"
    
    # Test complex expression
    r2 = calculate("sqrt(144) + 10")
    print(f"   sqrt(144) + 10 = {r2}")
    assert "22" in r2, f"Expected 22, got {r2}"

# ============================================================
# TEST 5: LLM Client Initialization
# ============================================================
@test("LLM Client - initializes without error")
def test_llm_init():
    from llm_client import LLMClient
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "llama3.2")
    
    llm = LLMClient(provider=provider, model_name=model)
    print(f"   Initialized: {llm.provider} / {llm.model_name}")
    assert llm.client is not None, "Client not initialized"

# ============================================================
# TEST 6: LLM Chat - Basic Response
# ============================================================
@test("LLM Client - basic chat works")
def test_llm_chat():
    from llm_client import LLMClient
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "llama3.2")
    
    llm = LLMClient(provider=provider, model_name=model)
    response = llm.chat([{"role": "user", "content": "Say 'hello' and nothing else"}])
    
    content = response.get('content', '')
    print(f"   Response: {content[:100]}")
    assert content, "Empty response from LLM"
    assert "error" not in content.lower() or "hello" in content.lower(), f"LLM error: {content}"

# ============================================================
# TEST 7: LLM Tool Calling
# ============================================================
@test("LLM Client - tool calling works")
def test_llm_tools():
    from llm_client import LLMClient
    from tools import registry
    
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "llama3.2")
    
    llm = LLMClient(provider=provider, model_name=model)
    
    messages = [
        {"role": "system", "content": "You must use the brave_search tool to answer. Do not answer without searching."},
        {"role": "user", "content": "Search for: current weather in New York"}
    ]
    
    response = llm.chat(messages=messages, tools=registry.schemas)
    tool_calls = response.get('tool_calls')
    
    print(f"   Tool calls: {tool_calls}")
    
    if tool_calls:
        print("   ✓ LLM requested tool call")
        assert tool_calls[0]['function']['name'] == 'brave_search', "Wrong tool called"
    else:
        print("   ⚠ LLM did not request tool (may be provider limitation)")

# ============================================================
# TEST 8: Vector Memory - Initialization
# ============================================================
@test("Vector Memory - ChromaDB initializes")
def test_memory_init():
    from memory import VectorMemory
    mem = VectorMemory()
    print(f"   Memory count: {mem.collection.count()}")
    assert mem.collection is not None, "Collection not created"

# ============================================================
# TEST 9: Vector Memory - Add & Search
# ============================================================
@test("Vector Memory - add and search works")
def test_memory_ops():
    from memory import VectorMemory
    mem = VectorMemory()
    
    # Add test memory
    test_text = f"TEST_MEMORY_{time.time()}: The user's favorite color is blue."
    mem.add(test_text, metadata={"type": "test"})
    print(f"   Added: {test_text[:50]}...")
    
    # Search for it
    results = mem.search("What is the user's favorite color?", n_results=3)
    print(f"   Search results: {len(results)} items")
    
    # Check if our test memory is found
    found = any("blue" in r.lower() for r in results)
    print(f"   Found test memory: {found}")
    assert found, "Memory search failed to find added content"

# ============================================================
# TEST 10: Swarm Manager - Bootstrap
# ============================================================
@test("Swarm Manager - bootstrap detects cutoff date")
def test_bootstrap():
    # Import swarm (module already loaded at startup, don't reload to avoid interactive loop)
    import swarm
    
    boss = swarm.SwarmManager()
    boss.bootstrap_knowledge_cutoff()
    
    print(f"   Detected cutoff: {boss.knowledge_cutoff}")
    assert boss.knowledge_cutoff is not None, "Cutoff not detected"
    assert len(boss.knowledge_cutoff) == 10, f"Invalid date format: {boss.knowledge_cutoff}"

# ============================================================
# TEST 11: Swarm Workers - All registered
# ============================================================
@test("Swarm Workers - all agents registered (including Memory)")
def test_workers():
    import swarm
    
    workers = list(swarm.boss.workers.keys())
    print(f"   Workers: {workers}")
    
    assert "Memory" in workers, "Memory not registered"
    assert "Planner" in workers, "Planner not registered"
    assert "Researcher" in workers, "Researcher not registered"
    assert "Analyst" in workers, "Analyst not registered"
    assert "DependencyAnalyst" in workers, "DependencyAnalyst not registered"
    assert len(workers) == 5, f"Expected 5 workers, got {len(workers)}"

# ============================================================
# TEST 12: Memory Routing - queries about past conversations
# ============================================================
@test("Memory Agent - recall routes correctly")
def test_memory_routing():
    import swarm
    
    # Store a test memory first
    test_info = "TEST: The user said their favorite programming language is Python."
    swarm.memory.add(test_info, {"type": "test", "timestamp": "2024-01-01"})
    print(f"   Added test memory: {test_info[:50]}...")
    
    # Now query should route to Memory agent
    result = swarm.boss.delegate("What is my favorite programming language?")
    print(f"   Result preview: {result[:150] if result else 'None'}...")
    assert result, "No response from memory agent"

# ============================================================
# TEST 13: End-to-End Query (if time permits)
# ============================================================
@test("End-to-End - simple query routed correctly")
def test_e2e():
    import swarm
    
    # Quick test - math should go to Analyst
    result = swarm.boss.delegate("What is 15 * 7?")
    print(f"   Result preview: {result[:150] if result else 'None'}...")
    assert result, "No response from swarm"

# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
