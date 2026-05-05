import time
from typing import List
from ..utils.pinecone_client import GravityPineconeClient
from ..schema import SemanticRecord

class Tier3Semantic:
    def __init__(self, pinecone: GravityPineconeClient):
        self.pc = pinecone

    def index_interaction(self, text: str, category: str = "conversation", metadata: dict = None):
        """Indexes an interaction for semantic retrieval."""
        meta = metadata or {}
        meta["category"] = category
        
        record = SemanticRecord(
            id=f"sem-{int(time.time() * 1000)}",
            text=text,
            metadata=meta
        )
        return self.pc.upsert_semantic(record)

    def search_relevant_context(self, query: str) -> str:
        """Searches for relevant historical context."""
        records = self.pc.search_semantic(query, top_k=5)
        if not records:
            return ""
        
        results = [f"- {r.text} (Rel: {r.score:.2f})" for r in records]
        return "\n".join(results)
