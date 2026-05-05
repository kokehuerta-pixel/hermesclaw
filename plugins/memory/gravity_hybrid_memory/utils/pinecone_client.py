import logging
from typing import List
from pinecone import Pinecone
from ..schema import SemanticRecord, MemoryConfig

logger = logging.getLogger(__name__)

class GravityPineconeClient:
    def __init__(self, config: MemoryConfig):
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index(config.pinecone_index)
        self.user_id = config.user_id

    def upsert_semantic(self, record: SemanticRecord, namespace: str = "gravity") -> bool:
        try:
            # Matt Pocock Style: Explicitly flatten metadata for Pinecone alignment
            record_data = {
                "id": record.id,
                "text": record.text,
                "user_id": self.user_id,
                **record.metadata
            }
            self.index.upsert_records(
                namespace=namespace,
                records=[record_data]
            )
            return True
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {e}")
            return False

    def search_semantic(self, query: str, top_k: int = 5, namespace: str = "gravity") -> List[SemanticRecord]:
        try:
            res = self.index.search(
                namespace=namespace,
                inputs={"text": query},
                top_k=top_k,
                fields=["text", "metadata"]
            )
            
            hits = getattr(res.result, "hits", []) if hasattr(res, "result") else []
            
            return [
                SemanticRecord(
                    id=hit.id,
                    text=hit.fields.get("text", ""),
                    score=hit.score,
                    metadata=hit.fields.get("metadata", {})
                ) for hit in hits
            ]
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
