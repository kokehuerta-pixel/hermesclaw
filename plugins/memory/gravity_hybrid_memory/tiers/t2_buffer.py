from ..utils.supabase_client import GravitySupabaseClient
from ..schema import Message

class Tier2Buffer:
    def __init__(self, supabase: GravitySupabaseClient):
        self.db = supabase

    def store_interaction(self, chat_id: str, role: str, content: str, user_id: str):
        """Saves a turn in the T2 buffer."""
        msg = Message(chat_id=chat_id, role=role, content=content, user_id=user_id)
        return self.db.save_message(msg)

    def get_context_summary(self, chat_id: str) -> str:
        """Retrieves the summary for the current chat."""
        # TODO: Implement summary retrieval logic from Supabase
        return ""
