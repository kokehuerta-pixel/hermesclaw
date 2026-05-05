import logging
from typing import List, Optional
from supabase import create_client, Client
from ..schema import CoreFact, Message, MemoryConfig

logger = logging.getLogger(__name__)

class GravitySupabaseClient:
    def __init__(self, config: MemoryConfig):
        self.client: Client = create_client(config.supabase_url, config.supabase_key)
        self.user_id = config.user_id

    def upsert_fact(self, fact: CoreFact) -> bool:
        try:
            data = fact.dict(exclude={'updated_at'})
            data['user_id'] = self.user_id
            self.client.schema("gravity").table("core_memory").upsert(data, on_conflict="user_id,key").execute()
            return True
        except Exception as e:
            logger.error(f"Error upserting fact to Supabase: {e}")
            return False

    def get_facts(self, limit: int = 20) -> List[CoreFact]:
        try:
            res = self.client.schema("gravity").table("core_memory")\
                .select("*")\
                .eq("user_id", self.user_id)\
                .order("updated_at", ascending=False)\
                .limit(limit)\
                .execute()
            return [CoreFact(**item) for item in res.data]
        except Exception as e:
            logger.error(f"Error fetching facts from Supabase: {e}")
            return []

    def save_message(self, msg: Message) -> bool:
        try:
            self.client.schema("gravity").table("messages").insert(msg.dict(exclude={'created_at'})).execute()
            return True
        except Exception as e:
            logger.error(f"Error saving message to Supabase: {e}")
            return False
