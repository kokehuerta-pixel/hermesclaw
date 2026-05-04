# Gravity Memory Provider

High-performance long-term memory for Hermes using Supabase and Pinecone.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install supabase pinecone-client google-generativeai
   ```

2. **Supabase Setup**:
   - Create a new project on [Supabase](https://supabase.com).
   - Go to the **SQL Editor** and run the contents of `schema.sql`.
   - Copy your **Project URL** and **Service Role Key**.

3. **Pinecone Setup**:
   - Create an index on [Pinecone](https://pinecone.io).
   - Dimensions: **768** (for `text-embedding-004`).
   - Metric: **Cosine**.
   - Copy your **API Key** and **Index Name**.

4. **Environment Variables**:
   Add the following to your `.env` file:
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=your_index_name
   GEMINI_API_KEY=your_gemini_key
   ```

5. **Activate**:
   Update `config.yaml`:
   ```yaml
   memory:
     provider: gravity
   ```
