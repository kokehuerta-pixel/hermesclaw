import logging
import os
import json
import threading
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from agent.memory_provider import MemoryProvider
from agent.auxiliary_client import get_text_auxiliary_client

logger = logging.getLogger(__name__)

# Constants for semantic memory noise filtering
NOISE_PREFIXES = ("/", "!", ".")
NOISE_KEYWORDS = [
    "Cambiado modelo", "Memoria actualizada", "Skin cambiada",
    "Sesión reanudada", "Limpiando terminal", "Cargando...",
    "Hola", "Buenas", "Gracias", "Ok", "Entendido"
]
INTENT_KEYWORDS = [
    "proyecto", "stack", "prefiero", "mi nombre", "estoy trabajando",
    "configura", "guarda", "recuerda", "contexto", "error"
]
MIN_CONTENT_LENGTH = 15

# Deferred imports for optional dependencies
supabase_client = None
pinecone_client = None

def _import_deps():
    global supabase_client, pinecone_client
    try:
        from supabase import create_client as supabase_create
        supabase_client = supabase_create
    except ImportError:
        logger.debug("supabase package not installed")
        
    try:
        from pinecone import Pinecone
        pinecone_client = Pinecone
    except ImportError:
        logger.debug("pinecone-client package not installed")

class GravityMemoryProvider(MemoryProvider):
    """Memory provider using Supabase (relational) and Pinecone (semantic)."""

    _INSTANCE: Optional['GravityMemoryProvider'] = None

    def __init__(self):
        self._supabase = None
        self._pinecone = None
        self._pc_index = None
        self._session_id = None
        self._user_id = "default"
        self._initialized = False
        GravityMemoryProvider._INSTANCE = self
        _import_deps()

    @property
    def name(self) -> str:
        return "gravity"

    def is_available(self) -> bool:
        """Check if credentials and dependencies are present."""
        has_deps = all([supabase_client, pinecone_client])
        has_creds = all([
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            os.getenv("PINECONE_API_KEY"),
            os.getenv("PINECONE_INDEX_NAME")
        ])
        return has_deps and has_creds

    def initialize(self, session_id: str, **kwargs) -> None:
        if not self.is_available():
            logger.warning("Gravity provider initialized but missing dependencies or credentials.")
            return

        self._session_id = session_id
        self._user_id = kwargs.get("user_id", "default")
        
        # Initialize Supabase
        self._supabase = supabase_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )

        # Initialize Pinecone
        pc = pinecone_client(api_key=os.getenv("PINECONE_API_KEY"))
        # Targeted index should be configured for Integrated Inference
        self._pc_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

        self._initialized = True
        logger.info(f"Gravity memory provider initialized for session {session_id}")

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._initialized or not query:
            return ""

        try:
            # Search Pinecone with Integrated Inference
            search_results = self._pc_index.search(
                namespace="gravity",
                query={
                    "inputs": {"text": query},
                    "top_k": 8
                },
                fields=["text", "category", "sub_category"]
            )

            hits = getattr(search_results.get("result", {}), "hits", [])
            if not hits:
                # Try fallback to legacy namespace (this might fail if model mismatch, but safe to try)
                search_results = self._pc_index.search(
                    namespace="conversations",
                    query={
                        "inputs": {"text": query},
                        "top_k": 5
                    },
                    fields=["text", "category"]
                )
                hits = getattr(search_results.get("result", {}), "hits", [])
                if not hits:
                    return ""

            # 3. Format context
            facts = []
            convs = []
            
            for hit in hits:
                fields = hit.get("fields", {})
                text = fields.get("text", "")
                category = fields.get("category", "conversación")
                score = hit.get("score", 0)
                
                if text:
                    sub_cat = fields.get("sub_category", "general")
                    label = f"HECHO:{sub_cat}" if category == "fact" else "CONV"
                    formatted = f"[{label} - {score:.2f}] {text}"
                    if category == "fact":
                        facts.append(formatted)
                    else:
                        convs.append(formatted)

            output_parts = []
            if facts:
                output_parts.append("**Hechos Recurridos:**\n" + "\n".join(facts))
            if convs:
                output_parts.append("**Conversaciones Previas:**\n" + "\n".join(convs))

            if not output_parts:
                return ""

            formatted_context = "\n\n".join(output_parts)
            return f"\n<gravity-context>\n{formatted_context}\n</gravity-context>\n"

        except Exception as e:
            logger.error(f"Error in gravity prefetch: {e}")
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._initialized:
            return

        def _bg_sync():
            try:
                # 1. Save to Supabase
                self._supabase.table("messages").insert([
                    {"session_id": self._session_id, "role": "user", "content": user_content},
                    {"session_id": self._session_id, "role": "assistant", "content": assistant_content}
                ]).execute()

                # 2. Targeted Fact Extraction: Only if content is significant
                is_command = user_content.strip().startswith(NOISE_PREFIXES)
                is_short = len(user_content.strip()) < MIN_CONTENT_LENGTH
                has_intent = any(kw.lower() in user_content.lower() for kw in INTENT_KEYWORDS)
                is_noisy = any(kw.lower() in user_content.lower() for kw in NOISE_KEYWORDS)

                if not is_command and not is_short and (has_intent or not is_noisy):
                    self._bg_extract_facts(user_content, assistant_content)
                else:
                    logger.debug(f"Skipping fact extraction for short/noisy turn: {user_content[:30]}...")

                # 3. Embed and Save turn to Pinecone (as Conversation history)
                # Skip administrative replies for semantic conversation memory
                is_admin_reply = any(kw in assistant_content for kw in NOISE_KEYWORDS)
                
                if is_command or is_admin_reply:
                    return

                turn_text = f"User: {user_content}\nAssistant: {assistant_content}"
                
                self._pc_index.upsert_records(
                    namespace="gravity",
                    records=[{
                        "id": f"conv-{self._session_id}-{datetime.now().timestamp()}",
                        "text": turn_text,
                        "category": "conversation",
                        "user_id": self._user_id,
                        "session_id": self._session_id,
                        "timestamp": datetime.now().isoformat()
                    }]
                )
            except Exception as e:
                logger.error(f"Error in gravity sync_turn: {e}")

        threading.Thread(target=_bg_sync, daemon=True).start()

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None, target_id: Optional[str] = None) -> None:
        if not self._initialized:
            return

        def _bg_mirror():
            try:
                if action in ['add', 'replace'] and target == 'user':
                    # 1. Save to Supabase
                    fact_id = target_id if target_id else ("fact_" + str(hash(content))[:8])
                    self._supabase.table("core_memory").upsert({
                        "user_id": self._user_id,
                        "key": fact_id,
                        "value": content,
                        "updated_at": datetime.now().isoformat()
                    }, on_conflict="user_id,key").execute()

                    # 2. Mirror to Pinecone for semantic retrieval
                    self._pc_index.upsert_records(
                        namespace="gravity",
                        records=[{
                            "id": f"fact-{self._user_id}-{fact_id}",
                            "text": content,
                            "category": "fact",
                            "sub_category": metadata.get("sub_category", "general") if metadata else "general",
                            "user_id": self._user_id,
                            "timestamp": datetime.now().isoformat()
                        }]
                    )
                    logger.info(f"Fact mirrored to Pinecone: {fact_id}")

            except Exception as e:
                logger.error(f"Error mirroring memory write to Supabase/Pinecone: {e}")

        threading.Thread(target=_bg_mirror, daemon=True).start()

    def _bg_extract_facts(self, user_content: str, assistant_content: str) -> None:
        """Consolidated fact extraction and reconciliation using local context."""
        try:
            # 1. Get similar existing facts for reconciliation context
            context_raw = self.prefetch(user_content)
            
            client, model = get_text_auxiliary_client("fact_extractor")
            if not client or not model:
                return

            prompt = (
                "You are a Memory Manager. Extract NEW, PERMANENT facts from the turn and reconcile them with existing context.\n"
                "Categories: user_preference, project_context, technical_stack, personal_info, workflow_pattern.\n\n"
                f"### EXISTING CONTEXT:\n{context_raw if context_raw else 'No previous facts found.'}\n\n"
                f"### CURRENT TURN:\nUser: {user_content}\nAssistant: {assistant_content}\n\n"
                "### INSTRUCTIONS:\n"
                "1. Identify NEW facts that are not already in the context.\n"
                "2. If a new fact updates or contradicts an existing one, mark it as 'update'.\n"
                "3. If it's truly new, mark it as 'add'.\n"
                "4. Return a JSON list: [{\"fact\": \"...\", \"category\": \"...\", \"action\": \"add|update\", \"target_id\": \"id_if_update\"}].\n"
                "Return [] if no valuable permanent facts are found."
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=600
            )
            
            raw_content = resp.choices[0].message.content
            try:
                start = raw_content.find('[')
                end = raw_content.rfind(']') + 1
                extracted = json.loads(raw_content[start:end]) if (start != -1 and end != -1) else []
            except:
                extracted = []

            for item in extracted:
                fact = item.get("fact", "").strip()
                category = item.get("category", "general_fact")
                action = item.get("action", "add")
                target_id = item.get("target_id")

                if fact:
                    # Save/Update using the standard tool path
                    self.on_memory_write(
                        "add" if action == "add" else "replace", 
                        "user", 
                        fact, 
                        metadata={"sub_category": category},
                        target_id=target_id
                    )
                    logger.info(f"Memory {action}: {fact[:50]}...")

        except Exception as e:
            logger.error(f"Error in consolidated fact extraction: {e}")
    def handle_manage_memory(self, action: str, target_id: Optional[str] = None, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Actual implementation of the manage_memory tool."""
        if not self._initialized:
            return json.dumps({"error": "Memory provider not initialized."})

        try:
            if action == "add":
                self.on_memory_write("add", "user", content, metadata=metadata)
                return json.dumps({"success": True, "message": "Fact added to long-term memory."})
            
            elif action == "search":
                # Use prefetch logic for search
                results = self.prefetch(content)
                return json.dumps({"success": True, "results": results})

            elif action == "delete":
                if not target_id:
                    return json.dumps({"error": "target_id required for delete."})
                # Delete from Supabase
                self._supabase.table("core_memory").delete().eq("key", target_id).execute()
                # Delete from Pinecone
                self._pc_index.delete(ids=[f"fact-{self._user_id}-{target_id}"], namespace="gravity")
                return json.dumps({"success": True, "message": f"Memory {target_id} deleted."})

            elif action == "update":
                if not target_id or not content:
                    return json.dumps({"error": "target_id and content required for update."})
                self.on_memory_write("replace", "user", content, metadata=metadata, target_id=target_id)
                return json.dumps({"success": True, "message": f"Memory {target_id} updated."})

            return json.dumps({"error": f"Unknown action: {action}"})

        except Exception as e:
            logger.error(f"Error in handle_manage_memory: {e}")
            return json.dumps({"error": str(e)})

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [] # We'll register via tools.py instead

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "SUPABASE_URL", "description": "Supabase Project URL", "secret": True, "required": True},
            {"key": "SUPABASE_SERVICE_ROLE_KEY", "description": "Supabase Service Role Key", "secret": True, "required": True},
            {"key": "PINECONE_API_KEY", "description": "Pinecone API Key", "secret": True, "required": True},
            {"key": "PINECONE_INDEX_NAME", "description": "Pinecone Index Name (Integrated)", "secret": True, "required": True}
        ]

def register(ctx):
    provider = GravityMemoryProvider()
    ctx.register_memory_provider(provider)
