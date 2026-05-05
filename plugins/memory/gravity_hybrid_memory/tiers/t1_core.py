from typing import List, Optional
from ..utils.supabase_client import GravitySupabaseClient
from ..schema import CoreFact

class Tier1Core:
    def __init__(self, supabase: GravitySupabaseClient):
        self.db = supabase

    def learn_fact(self, key: str, value: str, category: str = "general"):
        """Saves a permanent fact to Tier 1."""
        fact = CoreFact(key=key, value=value, category=category)
        return self.db.upsert_fact(fact)

    def recall_facts(self, limit: int = 10) -> str:
        """Retrieves T1 facts as a formatted block for the prompt."""
        facts = self.db.get_facts(limit)
        if not facts:
            return "No permanent facts known yet."
        
        lines = [f"- [{f.category}] {f.key}: {f.value}" for f in facts]
        return "\n".join(lines)
