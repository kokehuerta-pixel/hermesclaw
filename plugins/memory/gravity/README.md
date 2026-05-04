# Gravity Memory Provider

Sistema de memoria de largo plazo de alto rendimiento para Hermes, utilizando Supabase (relacional) y Pinecone (semántica). 

Gravity está diseñado para ser un plugin nativo de Hermes, integrándose directamente en el ciclo de vida del agente sin necesidad de configuraciones externas complejas.

## Características

- **Almacenamiento Dual**: Mensajes y hechos en Supabase, búsqueda vectorial en Pinecone.
- **Extracción Autónoma**: Utiliza modelos auxiliares (Gemini) para extraer hechos de forma asíncrona.
- **Filtrado de Ruido**: Ignora comandos, saludos y respuestas cortas para mantener la memoria limpia.
- **Inferencia Integrada**: Optimizado para usar la inferencia integrada de Pinecone.

## Requisitos

1. **Instalar Dependencias**:
   ```bash
   pip install supabase pinecone google-generativeai
   ```

2. **Configuración de Supabase**:
   - Crea un proyecto en [Supabase](https://supabase.com).
   - En el **SQL Editor**, ejecuta el contenido de `schema.sql`.
   - Copia tu **Project URL** y **Service Role Key**.

3. **Configuración de Pinecone**:
   - Crea un índice en [Pinecone](https://pinecone.io).
   - Dimensiones: **768** (si usas `multilingual-e5-large` o similar con Integrated Inference).
   - Métrica: **Cosine**.
   - Copia tu **API Key** e **Index Name**.

4. **Variables de Entorno**:
   Añade lo siguiente a tu archivo `.env`:
   ```env
   SUPABASE_URL=tu_url_de_supabase
   SUPABASE_SERVICE_ROLE_KEY=tu_clave_de_servicio
   PINECONE_API_KEY=tu_clave_de_pinecone
   PINECONE_INDEX_NAME=tu_nombre_de_indice
   ```

## Activación

Simplemente configura el proveedor en tu `config.yaml`:
```yaml
memory:
  provider: gravity
```

Hermes detectará automáticamente el plugin y registrará la herramienta `gravity_manage_memory` para que el modelo pueda gestionar su propia memoria de forma explícita.
