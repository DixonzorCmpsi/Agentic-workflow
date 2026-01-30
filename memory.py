import chromadb
from chromadb.utils import embedding_functions
import uuid
import json

# --- CONFIGURATION ---
# We use a persistent local database stored in the 'memory_db' folder
DB_PATH = "./memory_db"
COLLECTION_NAME = "agent_history"

class VectorMemory:
    def __init__(self):
        print("    [SYSTEM] Initializing Vector Database (ChromaDB)...")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        
        # We use a free, local embedding model (all-MiniLM-L6-v2)
        # This converts text into a list of 384 numbers (vectors)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn
        )
        print(f"    [SYSTEM] Memory Layer Ready. Count: {self.collection.count()} memories.")

    def add(self, text, metadata=None):
        """Stores a piece of information into long-term memory."""
        if not metadata:
            metadata = {"source": "conversation"}
            
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    def search(self, query, n_results=3):
        """Standard Semantic Search (Linear)."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        # Flatten the list of lists
        return results['documents'][0] if results['documents'] else []

    def recursive_recall(self, query, llm_check_fn=None, depth=1, max_depth=2):
        """
        RLM Implementation: Recursive Context Search.
        If the standard search yields vague results, we decompose and search again.
        """
        print(f"    [RLM] Depth {depth}: Searching for '{query}'...")
        
        # 1. Standard Search
        results = self.search(query, n_results=5)
        
        # Base Case: We hit max depth or found ample results
        if depth >= max_depth or not llm_check_fn:
            return results

        # 2. Evaluation Step (Metacognition)
        # We ask the Agent (passed in via llm_check_fn) if these results are enough.
        # This is the "Recursive" part: The Agent evaluates its own memory retrieval.
        combined_context = "\n".join(results)
        is_sufficient, new_query = llm_check_fn(query, combined_context)
        
        if is_sufficient:
            return results
        else:
            print(f"    [RLM] Context insufficient. Refining query to: '{new_query}'")
            # RECURSIVE CALL with the new, refined query
            additional_results = self.recursive_recall(new_query, llm_check_fn, depth + 1, max_depth)
            return list(set(results + additional_results)) # Merge unique results

# Initialize Global Memory Instance
memory = VectorMemory()