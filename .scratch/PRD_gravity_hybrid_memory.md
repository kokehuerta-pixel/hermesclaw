# PRD: Gravity Hybrid Memory Plugin (v2.0)

## Overview
Un plugin de memoria híbrida para Hermes Agent que implementa la arquitectura de 4 capas (4 Tiers) inspirada en Claude Agent 2.0. Utiliza Supabase para almacenamiento persistente y estructurado, y Pinecone para recuperación semántica.

## Objectives
1. **Aislamiento Total**: Funcionar como un plugin anexo en `plugins/memory/` sin modificar el núcleo de Hermes.
2. **Arquitectura 4-Tier**:
   - **T1 (Core)**: Hechos y preferencias (Supabase).
   - **T2 (Buffer)**: Historial de mensajes y resúmenes (Supabase).
   - **T3 (Semantic)**: Búsqueda vectorial (Pinecone).
   - **T4 (Reflection)**: Proceso de consolidación de T2/T3 a T1.
3. **Matt Pocock Standard**: Código TypeScript (o Python siguiendo principios de tipado estricto) altamente modular, validado con esquemas y resistente a errores.

## Target Architecture
```
hermes-agent/
├── plugins/
│   └── memory/
│       └── gravity_hybrid_memory/
│           ├── __init__.py      # Registro del plugin
│           ├── schema.py        # Validación (Pydantic/Zod style)
│           ├── tiers/
│           │   ├── t1_core.py   # Supabase Facts
│           │   ├── t2_buffer.py # Supabase Messages
│           │   ├── t3_semantic.py # Pinecone
│           │   └── t4_reflect.py  # Reflection Logic
│           └── utils/
│               ├── supabase_client.py
│               └── pinecone_client.py
```

## Success Criteria
- El agente puede recordar hechos guardados en Supabase.
- El agente recupera contexto histórico vía Pinecone.
- Las actualizaciones del repositorio oficial de Hermes no afectan al plugin.
- El plugin se activa vía `config.yaml` o `.env`.
