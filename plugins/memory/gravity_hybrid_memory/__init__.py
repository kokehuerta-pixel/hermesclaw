import logging
import os
import json
import threading
from typing import Dict, Any, List, Optional
from pathlib import Path

from agent.memory_provider import MemoryProvider
from .schema import MemoryConfig
from .utils.supabase_client import GravitySupabaseClient
from .utils.pinecone_client import GravityPineconeClient
from .tiers.t1_core import Tier1Core
from .tiers.t2_buffer import Tier2Buffer
from .tiers.t3_semantic import Tier3Semantic
from .tiers.t4_reflect import Tier4Reflection

logger = logging.getLogger(__name__)

class GravityHybridMemoryProvider(MemoryProvider):
    def __init__(self):
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t4 = None
        self._initialized = False
        self._context_cache = ""
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "gravity-hybrid-memory"

    def is_available(self) -> bool:
        return all([
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
            os.environ.get("PINECONE_API_KEY"),
            os.environ.get("PINECONE_INDEX_NAME")
        ])

    def initialize(self, session_id: str, **kwargs) -> None:
        try:
            config = MemoryConfig(
                supabase_url=os.environ["SUPABASE_URL"],
                supabase_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"],
                pinecone_api_key=os.environ["PINECONE_API_KEY"],
                pinecone_index=os.environ["PINECONE_INDEX_NAME"],
                user_id=kwargs.get("user_id", "default")
            )
            
            sb_client = GravitySupabaseClient(config)
            pc_client = GravityPineconeClient(config)

            self.t1 = Tier1Core(sb_client)
            self.t2 = Tier2Buffer(sb_client)
            self.t3 = Tier3Semantic(pc_client)
            self.t4 = Tier4Reflection(self.t1)

            self._session_id = session_id
            self._user_id = config.user_id
            self._initialized = True
            logger.info("Gravity Hybrid Memory v2.0 Initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Gravity Hybrid Memory: {e}")

    def system_prompt_block(self) -> str:
        if not self._initialized: return ""
        
        core_facts = self.t1.recall_facts(limit=5)
        
        return (
            "\n## GRAVITY HYBRID MEMORY (T1-T4)\n"
            "### Tier 1: Core Facts\n"
            f"{core_facts}\n"
            "### Tier 3: Historical Context\n"
            f"{self._context_cache if self._context_cache else 'No relevant history found yet.'}\n"
        )

    def prefetch(self, query: str, **kwargs) -> str:
        if not self._initialized or not query: return ""
        # Tier 3 Search
        self._context_cache = self.t3.search_relevant_context(query)
        return self._context_cache

    def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None:
        if not self._initialized: return
        
        chat_id = self._session_id
        
        # Tier 2: Store Messages
        self.t2.store_interaction(chat_id, "user", user_content, self._user_id)
        self.t2.store_interaction(chat_id, "assistant", assistant_content, self._user_id)
        
        # Tier 3: Index for semantic search
        self.t3.index_interaction(f"User: {user_content}\nAssistant: {assistant_content}")
        
        # Tier 4: Reflection (Learn in background)
        threading.Thread(
            target=self.t4.reflect_and_learn, 
            args=(f"User: {user_content}\nAssistant: {assistant_content}",),
            daemon=True
        ).start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "gravity_manage_fact",
                "description": "Manage Tier 1 permanent facts in Gravity Memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "delete", "list"]},
                        "key": {"type": "string", "description": "The unique name of the fact."},
                        "value": {"type": "string", "description": "The content of the fact (for add)."}
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "gravity_configure_router",
                "description": "Configure the strategic model router for specific tasks (coding, reflection, etc.).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["update", "list"]},
                        "task": {"type": "string", "description": "The task name (e.g., 'coding', 'reflection')."},
                        "model": {"type": "string", "description": "The model name to assign."}
                    },
                    "required": ["action"]
                }
            }
        ]

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "SUPABASE_URL", "description": "Supabase Project URL", "required": True},
            {"key": "SUPABASE_SERVICE_ROLE_KEY", "description": "Supabase API Key", "required": True},
            {"key": "PINECONE_API_KEY", "description": "Pinecone API Key", "required": True},
            {"key": "PINECONE_INDEX_NAME", "description": "Pinecone Index Name", "required": True}
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "gravity_manage_fact":
            action = args.get("action")
            key = args.get("key")
            value = args.get("value")
            if action == "add" and key and value:
                self.t1.learn_fact(key, value)
                return json.dumps({"success": True, "message": f"Learned fact: {key}"})
            elif action == "list":
                facts = self.t1.recall_facts()
                return json.dumps({"success": True, "facts": facts})
        elif tool_name == "gravity_configure_router":
            from agent.auxiliary_client import get_gemini_strategic_models, update_gemini_strategic_model
            action = args.get("action")
            if action == "list":
                return json.dumps({"success": True, "models": get_gemini_strategic_models()})
            elif action == "update":
                task = args.get("task")
                model = args.get("model")
                if task and model:
                    if update_gemini_strategic_model(task, model):
                        return json.dumps({"success": True, "message": f"Updated {task} model to {model}"})
                    else:
                        return json.dumps({"success": False, "error": f"Unknown task: {task}"})
        return json.dumps({"success": False, "error": "Unknown tool or missing args"})

    def shutdown(self) -> None:
        self._initialized = False

def register(ctx):
    ctx.register_memory_provider(GravityHybridMemoryProvider())
