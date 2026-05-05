import logging
import os
import json
import threading
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from agent.memory_provider import MemoryProvider
from agent.auxiliary_client import get_text_auxiliary_client
from tools.registry import tool_error

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
    except Exception as e:
        logger.error(f"Failed to import Supabase: {e}")
        supabase_client = None
        
    try:
        from pinecone import Pinecone
        pinecone_client = Pinecone
    except Exception as e:
        logger.error(f"Failed to import Pinecone: {e}")
        pinecone_client = None

# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/gravity.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "supabase_url": os.environ.get("SUPABASE_URL", ""),
        "supabase_key": os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        "pinecone_api_key": os.environ.get("PINECONE_API_KEY", ""),
        "pinecone_index": os.environ.get("PINECONE_INDEX_NAME", ""),
        "user_id": "default",
    }

    config_path = get_hermes_home() / "gravity.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config

# ---------------------------------------------------------------------------
# Tool Schemas
# ---------------------------------------------------------------------------

MANAGE_MEMORY_SCHEMA = {
    "name": "gravity_manage_memory",
    "description": (
        "Manage the long-term autonomous memory (Gravity). "
        "Use this to add, search, update, delete, or list permanent facts about the user, projects, or technical preferences."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "update", "delete", "list"],
                "description": "The action to perform."
            },
            "target_id": {
                "type": "string",
                "description": "The key or ID of the memory (e.g., 'user_name', 'project_stack'). Required for update/delete."
            },
            "content": {
                "type": "string",
                "description": "The memory text to add, update, or the query for search."
            },
            "category": {
                "type": "string",
                "enum": ["user_preference", "project_context", "technical_stack", "personal_info", "workflow_pattern", "general"],
                "description": "The category of the memory (optional)."
            },
            "limit": {
                "type": "integer",
                "description": "Max results for list/search (optional, default 10).",
                "default": 10
            }
        },
        "required": ["action"]
    },
}

# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class GravityMemoryProvider(MemoryProvider):
    """Memory provider using Supabase (relational) and Pinecone (semantic)."""

    def __init__(self):
        self._supabase = None
        self._pinecone = None
        self._pc_index = None
        self._session_id = None
        self._user_id = "default"
        self._initialized = False
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        _import_deps()

    @property
    def name(self) -> str:
        return "gravity"

    def is_available(self) -> bool:
        """Check if credentials and dependencies are present."""
        cfg = _load_config()
        has_deps = all([supabase_client, pinecone_client])
        has_creds = all([
            cfg.get("supabase_url"),
            cfg.get("supabase_key"),
            cfg.get("pinecone_api_key"),
            cfg.get("pinecone_index")
        ])
        return has_deps and has_creds

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to $HERMES_HOME/gravity.json."""
        config_path = Path(hermes_home) / "gravity.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def initialize(self, session_id: str, **kwargs) -> None:
        cfg = _load_config()
        if not self.is_available():
            logger.warning("Gravity provider missing dependencies or credentials.")
            return

        self._session_id = session_id
        self._user_id = kwargs.get("user_id") or cfg.get("user_id", "default")
        
        # Initialize Supabase
        self._supabase = supabase_client(
            cfg["supabase_url"],
            cfg["supabase_key"]
        )

        # Initialize Pinecone
        pc = pinecone_client(api_key=cfg["pinecone_api_key"])
        self._pc_index = pc.Index(cfg["pinecone_index"])

        self._initialized = True
        logger.info(f"Gravity memory provider initialized for session {session_id}")

    def system_prompt_block(self) -> str:
        if not self._initialized:
            return ""
        
        # Quick check for core facts count to show status
        try:
            res = self._supabase.table("core_memory").select("id", count="exact").eq("user_id", self._user_id).execute()
            count = res.count if hasattr(res, "count") else 0
        except:
            count = "?"

        return (
            "# Gravity Memory (Nativo)\n"
            f"Estado: Activo. Usuario: {self._user_id}. Hechos guardados: {count}.\n"
            "Este sistema captura automáticamente hechos importantes. Puedes gestionarlos con 'gravity_manage_memory'.\n"
            "Los hechos recuperados semánticamente se inyectarán automáticamente cuando sean relevantes."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return the pre-fetched semantic context or run a direct search."""
        if not self._initialized:
            return ""

        # 1. If we have a query, try to get fresh results if background one is empty or irrelevant
        if query:
            # For now, if we have a query but no background result, we just use the background mechanism logic
            # but synchronously if needed (though the interface expects fast return).
            # We'll stick to the background result for the system prompt injection,
            # but allow direct tool calls to use a separate path.
            pass

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
        
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background semantic search for the next turn."""
        if not self._initialized or not query:
            return

        def _run_search():
            try:
                # Search Pinecone with Integrated Inference
                search_results = self._pc_index.search(
                    namespace="gravity",
                    inputs={"text": query},
                    top_k=8,
                    fields=["text", "category", "sub_category"]
                )

                # Extract hits safely
                hits = []
                if hasattr(search_results, "result") and hasattr(search_results.result, "hits"):
                    hits = search_results.result.hits
                elif isinstance(search_results, dict):
                    hits = search_results.get("result", {}).get("hits", [])
                
                if not hits:
                    return

                # Format context
                facts = []
                convs = []
                
                for hit in hits:
                    if hasattr(hit, "fields"):
                        fields = hit.fields
                        score = hit.score
                    elif isinstance(hit, dict):
                        fields = hit.get("fields", {})
                        score = hit.get("score", 0)
                    else:
                        continue
                    
                    text = fields.get("text", "")
                    category = fields.get("category", "")
                    if not text:
                        continue

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

                if output_parts:
                    formatted_context = "\n\n".join(output_parts)
                    with self._prefetch_lock:
                        self._prefetch_result = f"\n<gravity-context>\n{formatted_context}\n</gravity-context>\n"

            except Exception as e:
                logger.error(f"Error in gravity background prefetch: {e}")

        self._prefetch_thread = threading.Thread(target=_run_search, daemon=True, name="gravity-prefetch")
        self._prefetch_thread.start()

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

                # 2. Targeted Fact Extraction
                is_command = user_content.strip().startswith(NOISE_PREFIXES)
                is_short = len(user_content.strip()) < MIN_CONTENT_LENGTH
                has_intent = any(kw.lower() in user_content.lower() for kw in INTENT_KEYWORDS)
                is_noisy = any(kw.lower() in user_content.lower() for kw in NOISE_KEYWORDS)

                if not is_command and not is_short and (has_intent or not is_noisy):
                    self._bg_extract_facts(user_content, assistant_content)

                # 3. Save turn to Pinecone (as Conversation history)
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

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self._initialized:
            return

        def _bg_mirror():
            try:
                if action in ['add', 'replace'] and target == 'user':
                    # Extract target_id from metadata if available (standard pattern)
                    target_id = metadata.get("target_id") if metadata else None
                    if not target_id:
                        # Fallback: hash the content if no ID provided
                        target_id = ("fact_" + str(hash(content))[:8])
                    # 1. Save to Supabase (Source of Truth)
                    self._supabase.table("core_memory").upsert({
                        "user_id": self._user_id,
                        "key": target_id,
                        "value": content,
                        "category": metadata.get("sub_category", "general") if metadata else "general",
                        "metadata": metadata or {},
                        "updated_at": datetime.now().isoformat()
                    }, on_conflict="user_id,key").execute()

                    # 2. Mirror to Pinecone (Semantic Index)
                    self._pc_index.upsert_records(
                        namespace="gravity",
                        records=[{
                            "id": f"fact-{self._user_id}-{target_id}",
                            "text": content,
                            "category": "fact",
                            "sub_category": metadata.get("sub_category", "general") if metadata else "general",
                            "user_id": self._user_id,
                            "timestamp": datetime.now().isoformat()
                        }]
                    )
            except Exception as e:
                logger.error(f"Error mirroring memory write: {e}")

        threading.Thread(target=_bg_mirror, daemon=True).start()

    def _bg_extract_facts(self, user_content: str, assistant_content: str) -> None:
        """Consolidated fact extraction and reconciliation."""
        try:
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
                    # Prepare metadata with category and target_id
                    metadata = {"sub_category": category}
                    if target_id:
                        metadata["target_id"] = target_id
                        
                    self.on_memory_write(
                        "add" if action == "add" else "replace", 
                        "user", 
                        fact, 
                        metadata=metadata
                    )
        except Exception as e:
            logger.error(f"Error in fact extraction: {e}")

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._initialized:
            return tool_error("Gravity memory provider not initialized.")

        if tool_name == "gravity_manage_memory":
            action = args.get("action")
            target_id = args.get("target_id")
            content = args.get("content")
            category = args.get("category")
            
            try:
                if action == "add":
                    self.on_memory_write("add", "user", content, metadata={"sub_category": category} if category else None)
                    return json.dumps({"success": True, "message": "Fact added to long-term memory."})
                
                elif action == "search":
                    if not content:
                        return tool_error("content (query) required for search.")
                    # Direct Pinecone search for immediate tool result
                    search_results = self._pc_index.search(
                        namespace="gravity",
                        inputs={"text": content},
                        top_k=args.get("limit", 10),
                        fields=["text", "category", "sub_category"]
                    )
                    hits = getattr(search_results.result, "hits", []) if hasattr(search_results, "result") else []
                    return json.dumps({"success": True, "hits": [
                        {"text": h.fields.get("text"), "score": h.score, "id": h.id} for h in hits
                    ]})

                elif action == "list":
                    limit = args.get("limit", 10)
                    res = self._supabase.table("core_memory").select("*").eq("user_id", self._user_id).limit(limit).order("updated_at", ascending=False).execute()
                    return json.dumps({"success": True, "memories": res.data})

                elif action == "delete":
                    if not target_id:
                        return tool_error("target_id (key) required for delete.")
                    self._supabase.table("core_memory").delete().eq("user_id", self._user_id).eq("key", target_id).execute()
                    self._pc_index.delete(ids=[f"fact-{self._user_id}-{target_id}"], namespace="gravity")
                    return json.dumps({"success": True, "message": f"Memory '{target_id}' deleted."})

                elif action == "update":
                    if not target_id or not content:
                        return tool_error("target_id and content required for update.")
                    
                    self.on_memory_write("replace", "user", content, metadata={"sub_category": category, "target_id": target_id} if category else {"target_id": target_id})
                    return json.dumps({"success": True, "message": f"Memory '{target_id}' updated."})

                return tool_error(f"Unknown action: {action}")
            except Exception as e:
                return tool_error(str(e))

        return tool_error(f"Unknown tool: {tool_name}")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MANAGE_MEMORY_SCHEMA]

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "supabase_url", "description": "Supabase Project URL", "secret": False, "required": True, "env_var": "SUPABASE_URL"},
            {"key": "supabase_key", "description": "Supabase Service Role Key", "secret": True, "required": True, "env_var": "SUPABASE_SERVICE_ROLE_KEY"},
            {"key": "pinecone_api_key", "description": "Pinecone API Key", "secret": True, "required": True, "env_var": "PINECONE_API_KEY"},
            {"key": "pinecone_index", "description": "Pinecone Index Name", "secret": False, "required": True, "env_var": "PINECONE_INDEX_NAME"},
            {"key": "user_id", "description": "User identifier for memory scoping", "default": "default"}
        ]

    def shutdown(self) -> None:
        self._supabase = None
        self._pc_index = None

def register(ctx):
    ctx.register_memory_provider(GravityMemoryProvider())
