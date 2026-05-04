import json
import logging
from typing import Dict, Any, Optional, List
from tools.registry import registry

logger = logging.getLogger(__name__)

def manage_memory(action: str, target_id: Optional[str] = None, content: Optional[str] = None, category: Optional[str] = None, **kwargs) -> str:
    """Manage the agent's long-term memory (Gravity)."""
    from plugins.memory.gravity import GravityMemoryProvider
    
    provider = GravityMemoryProvider._INSTANCE
    if not provider:
        return json.dumps({"error": "Gravity memory provider is not active or initialized."})
    
    # Pass category as metadata if provided
    metadata = {"sub_category": category} if category else None
    
    # Note: GravityMemoryProvider.handle_manage_memory needs to handle metadata
    return provider.handle_manage_memory(action, target_id, content, metadata=metadata)

registry.register(
    name="manage_memory",
    toolset="memory",
    schema={
        "name": "manage_memory",
        "description": "Manage the long-term autonomous memory (Gravity). Use this to add, search, update, or delete permanent facts about the user or project.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "search", "update", "delete"],
                    "description": "The action to perform."
                },
                "target_id": {
                    "type": "string",
                    "description": "The ID of the memory to update or delete (returned from search)."
                },
                "content": {
                    "type": "string",
                    "description": "The memory text to add, update, or search for."
                },
                "category": {
                    "type": "string",
                    "enum": ["user_preference", "project_context", "technical_stack", "personal_info", "workflow_pattern"],
                    "description": "The category of the memory (optional, defaults to 'general')."
                }
            },
            "required": ["action"]
        }
    },
    handler=lambda args, **kw: manage_memory(
        action=args.get("action"),
        target_id=args.get("target_id"),
        content=args.get("content"),
        category=args.get("category"),
        **kw
    )
)
