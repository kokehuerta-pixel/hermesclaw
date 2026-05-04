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
    "Cambiado modelo",
    "Memoria actualizada",
    "Skin cambiada",
    "Sesión reanudada",
    "Limpiando terminal",
    "Cargando..."
]

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

                # 2. Extract Facts (Autonomous)
                self._bg_extract_facts(user_content, assistant_content)

                # 3. Embed and Save turn to Pinecone (as Conversation history)
                # OPTIMIZATION: Skip noisy/administrative turns for semantic memory
                is_command = user_content.strip().startswith(NOISE_PREFIXES)
                is_admin_reply = any(kw in assistant_content for kw in NOISE_KEYWORDS)
                
                if is_command or is_admin_reply:
                    logger.debug(f"Skipping noisy turn for Pinecone: {user_content[:50]}...")
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
        """Call auxiliary model to extract facts from turn and save them."""
        try:
            client, model = get_text_auxiliary_client("fact_extractor")
            if not client or not model:
                return

            prompt = (
                "Extract any NEW, PERMANENT facts from this conversation turn and categorize them.\n"
                "Categories: user_preference, project_context, technical_stack, personal_info, workflow_pattern.\n\n"
                f"User: {user_content}\n"
                f"Assistant: {assistant_content}\n\n"
                "Return a JSON list of objects: [{\"fact\": \"...\", \"category\": \"...\"}]. If nothing, return []."
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            
            raw_content = resp.choices[0].message.content
            # Simple JSON extraction
            try:
                start = raw_content.find('[')
                end = raw_content.rfind(']') + 1
                if start != -1 and end != -1:
                    extracted = json.loads(raw_content[start:end])
                else:
                    extracted = []
            except:
                extracted = []

            added_facts = []
            for item in extracted:
                fact = item.get("fact", "")
                category = item.get("category", "general_fact")
                if fact.strip():
                    # Save with metadata
                    self.on_memory_write("add", "user", fact.strip(), metadata={"sub_category": category})
                    added_facts.append(fact.strip())

            # Trigger reconciliation for the new facts
            if added_facts:
                self._reconcile_memories(added_facts)

        except Exception as e:
            logger.error(f"Error in autonomous fact extraction: {e}")

    def _reconcile_memories(self, new_facts: List[str]) -> None:
        """Identify and merge redundant or contradictory facts."""
        if not new_facts or not self._initialized:
            return

        try:
            client, model = get_text_auxiliary_client("fact_reconciler")
            if not client or not model:
                return

            for fact in new_facts:
                # 1. Search for similar facts
                similar = self.prefetch(fact)
                if not similar:
                    continue

                # 2. Ask LLM to judge redundancy or contradiction
                prompt = (
                    "Compare these two facts for a user profile. Are they redundant, contradictory, or compatible?\n\n"
                    f"New Fact: {fact}\n"
                    f"Existing Fact: {similar[0]}\n\n"
                    "If they are REDUNDANT or CONTRADICTORY, return a JSON object:\n"
                    "{\"status\": \"redundant|contradictory\", \"merged_fact\": \"...\", \"action\": \"replace|delete\"}\n"
                    "If they are COMPATIBLE, return {\"status\": \"compatible\"}."
                )

                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=500
                )

                try:
                    result = json.loads(resp.choices[0].message.content)
                    if result.get("status") in ["redundant", "contradictory"]:
                        # For now, let's just log it. In a real scenario, we'd delete the old and add the new/merged.
                        logger.info(f"Memory reconciliation: {result['status']} found between '{fact}' and '{similar[0]}'")
                        # Implementation detail: we'd need the ID of 'similar[0]' to replace it.
                except:
                    continue

        except Exception as e:
            logger.error(f"Error in memory reconciliation: {e}")

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
