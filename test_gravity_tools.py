import os
import json
import logging
from unittest.mock import MagicMock, patch

# Setup mock environment
os.environ["SUPABASE_URL"] = "https://example.supabase.co"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "fake-key"
os.environ["PINECONE_API_KEY"] = "fake-key"
os.environ["PINECONE_INDEX_NAME"] = "fake-index"
os.environ["GEMINI_API_KEY"] = "fake-key"

from plugins.memory.gravity import GravityMemoryProvider

def test_categorization():
    print("Testing GravityMemoryProvider categorization...")
    with patch('supabase.create_client'), patch('pinecone.Pinecone'), patch('google.generativeai.configure'):
        provider = GravityMemoryProvider()
        provider._initialized = True
        provider._supabase = MagicMock()
        provider._pc_index = MagicMock()
        
        print("Testing handle_manage_memory 'add' with category...")
        res = provider.handle_manage_memory(
            action="add", 
            content="User prefers dark mode.", 
            metadata={"sub_category": "user_preference"}
        )
        print(f"Result: {res}")
        assert "success" in res

        # Verify that prefetch would show the category
        # Mocking metadata in Pinecone match
        match = {
            "metadata": {
                "text": "User prefers dark mode.",
                "category": "fact",
                "sub_category": "user_preference"
            },
            "score": 0.95
        }
        
        provider._pc_index.query.return_value = {"matches": [match]}
        with patch('google.generativeai.embed_content', return_value={'embedding': [0.1]*768}):
            context = provider.prefetch("interface preferences")
            print(f"Context snippet: {context[:100]}...")
            assert "HECHO:user_preference" in context

    print("Categorization tests passed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_categorization()
