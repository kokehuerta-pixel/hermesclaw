# PRD: Strategic Model Routing & Gravity Memory Integration

## Problem Statement

The Hermes agent currently lacks a sophisticated way to manage multiple AI models from the Gemini/Gemma family, leading to inefficient quota usage and suboptimal task performance. Additionally, the existing memory systems are often shallow or disconnected, lacking the multi-tier persistence (permanent facts vs. semantic history) required for complex, long-running agentic tasks.

## Solution

Implement a dual-system upgrade:
1. **Strategic Model Routing**: A dynamic orchestration layer in `auxiliary_client.py` that maps specific tasks (coding, vision, reflection) to the most cost-effective and capable Gemma/Gemini model.
2. **Gravity Hybrid Memory**: A 4-tier memory plugin integrating Supabase (relational/permanent) and Pinecone (vector/semantic) to provide deep, persistent context and background reflection.

## User Stories

1. As a developer, I want the agent to use `gemma-4-31b-it` for complex decisions, so that I get the highest quality reasoning without manual model switching.
2. As a user with high latency concerns, I want the agent to use `gemini-3.1-flash-lite-preview` for summaries and reflection, so that I save quota and time on non-critical tasks.
3. As an agent, I want to store permanent facts in Supabase (Tier 1), so that I remember critical user preferences across different sessions and profiles.
4. As an agent, I want to search through my entire history via Pinecone (Tier 3), so that I can retrieve relevant context from conversations that happened weeks ago.
5. As an agent, I want a background reflection process (Tier 4) to analyze our conversation and learn new facts automatically, so that I become smarter without interrupting the user.
6. As a system administrator, I want to configure all these models and memory credentials via a unified `config.yaml` and `.env` system, so that deployment is standardized.

## Implementation Decisions

- **Routing Layer**: Modified `AIAuxiliaryClient` to use a mapping dictionary `_GEMINI_STRATEGIC_MODELS` when `model="auto"` is requested for auxiliary tasks.
- **Plugin Architecture**: Created `gravity_hybrid_memory` as an "exclusive" memory provider plugin to avoid conflicts with built-in providers while maintaining a clean interface.
- **Tiers Division**:
    - **T1**: Permanent facts (`gravity.core_memory` table in Supabase).
    - **T2**: Interaction buffer (`gravity.messages` table in Supabase).
    - **T3**: Semantic search (Pinecone index).
    - **T4**: Logic-based reflection using `threading` and the `Lite` model.
- **Config Management**: Added required environment variables (`SUPABASE_URL`, `PINECONE_API_KEY`, etc.) to the system's discovery list.

## Testing Decisions

- **Unit Testing**: Test the `Tier1Core` and `Tier3Semantic` logic in isolation using mocks for Supabase and Pinecone.
- **Integration Testing**: Verify the `GravityHybridMemoryProvider` initialization and its ability to inject context into the system prompt.
- **Routing Verification**: Monitor `agent.log` to confirm that specific purposes (e.g., `fact_extractor`) trigger the correct model calls.

## Out of Scope

- Implementing similar routing for non-Google/Gemini providers (OpenAI, Anthropic) in this phase.
- Automatic creation of Pinecone indexes (must be created manually by the user for now).

## Further Notes

- The system is designed to be profile-aware, using `get_hermes_home()` to distinguish between different user environments if applicable.
- The use of `gemma-4-31b-it` as the default model ensures high-fidelity fallbacks.
