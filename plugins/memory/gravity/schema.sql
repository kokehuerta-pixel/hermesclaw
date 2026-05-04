-- 1. Conversation History Table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 2. Core Memory (Profile Facts) Table
CREATE TABLE IF NOT EXISTS core_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL DEFAULT 'default',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, key)
);

-- 3. Indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_core_memory_user ON core_memory(user_id);

-- 4. RLS (Row Level Security) - Optional but recommended
-- ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE core_memory ENABLE ROW LEVEL SECURITY;
