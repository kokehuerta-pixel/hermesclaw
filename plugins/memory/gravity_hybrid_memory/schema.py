from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator

class CoreFact(BaseModel):
    """Tier 1: Permanent autonomous facts."""
    key: str = Field(..., description="Unique identifier for the fact")
    value: str = Field(..., description="The content of the memory")
    category: str = Field("general", description="user_preference, project_context, etc.")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: Optional[datetime] = None

class Message(BaseModel):
    """Tier 2: Conversation buffer entry."""
    role: str
    content: str
    chat_id: str
    user_id: str
    created_at: Optional[datetime] = None

    @validator('role')
    def validate_role(cls, v):
        allowed = ['user', 'assistant', 'system', 'tool']
        if v not in allowed:
            raise ValueError(f"Role must be one of {allowed}")
        return v

class SemanticRecord(BaseModel):
    """Tier 3: Pinecone vector representation."""
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

class MemoryConfig(BaseModel):
    """Plugin configuration schema."""
    supabase_url: str
    supabase_key: str
    pinecone_api_key: str
    pinecone_index: str
    user_id: str = "default"
