# Nutrition Healthcare Assistant Backend

This project has been refactored from a generic prompt-testing/RAG utility into a local-development nutrition-focused healthcare chat assistant.

The backend exposes structured JSON endpoints that another frontend application can consume. The frontend is a chat-first Vite + React + TypeScript + Tailwind app for using those nutrition assistant capabilities without exposing advanced developer details by default.

## Project Structure

```text
backend/
  app/
    main.py        FastAPI routes and JSON response envelopes
    nutrition.py   Nutrition session state, memory derivation, tools, prompt assembly
    rag.py         In-memory file ingestion and retrieval
    prompting.py   Prompt section assembly helper
    schemas.py     Pydantic request/response models
    config.py      Settings, CORS origins, upload paths
frontend/
  Vite + React + TypeScript + Tailwind chat interface
```

## What The Backend Does

- Provides a virtual nutrition assistant API, not a general chatbot API.
- Seeds the app with dummy nutrition-related chat history.
- Stores chat history in memory during runtime.
- Appends new user and assistant messages through POST endpoints.
- Dynamically derives state and memory from current chat history.
- Tracks nutrition-relevant health context such as allergies, intolerances, disease history, deficiencies, digestive issues, pregnancy/postpartum flags, restrictions, and supplement/medication mentions.
- Uses fallback seeded dummy values when runtime chat history is insufficient.
- Preserves file upload and RAG ingestion/retrieval.
- Exposes the latest user prompt, prompt sections, assembled prompt, tool definitions, retrieval results, and generated output through GET endpoints.
- Uses a hardcoded nutrition-specific system prompt.
- Uses hardcoded nutrition tool definitions for now.
- Keeps persistence in memory, except uploaded source files are saved locally under `backend/data/uploads`.
- Enables CORS for `http://localhost:5173`.

## Local Backend Setup

From the repository root:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Optional model API configuration in `backend/.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://genailab.tcs.in
OPENAI_MODEL=azure_ai/genailab-maas-DeepSeek-V3-0324
OPENAI_VERIFY_SSL=false
```

The backend uses `langchain-openai` for model-backed generation and is configured for the GenAI Lab OpenAI-compatible endpoint. It still works without an API key by using local dummy generation. To call the model API, pass `use_model: true` to `POST /generate`.

## Run The Backend

From the repository root:

```bash
backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Or from inside `backend/` after activating the virtualenv:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Optional Frontend Setup

The frontend is a Vite + React + TypeScript + Tailwind application. The first screen is the nutrition chat experience; memory, knowledge/RAG, prompts, tools, and debug views are available from the collapsed Advanced panel.

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

If the frontend supports a custom backend URL, use:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Build the frontend:

```bash
cd frontend
npm run build
```

Note: if npm fails with `UNABLE_TO_GET_ISSUER_CERT_LOCALLY` in a corporate/local certificate environment, install dependencies with:

```bash
set npm_config_strict_ssl=false
npm install
```

## Endpoint Inventory

### Health And Config

- `GET /health`
- `GET /config`

Use these to verify the service and read app metadata, active modules, allowed origins, and local-development settings.

### Chat

- `GET /chat/history`
- `GET /chat/history/latest`
- `GET /chat/session`
- `POST /chat/message`

`POST /chat/message` accepts a nutrition query, appends it to chat history, refreshes memory, creates a local dummy assistant response, appends that response, and stores the latest generation result.

Example:

```json
{
  "text": "I have lactose intolerance and PCOS. Can you suggest a high-protein breakfast?",
  "source": "frontend"
}
```

### State And Memory

- `GET /memory`
- `GET /memory/latest`
- `GET /state`
- `GET /state-and-memory`

Memory output is intentionally flat and frontend-friendly. Important fields include:

- `nutrition_goals`
- `dietary_preferences`
- `cuisine_preferences`
- `disliked_foods`
- `allergies`
- `intolerances`
- `diseases_history`
- `specific_conditions`
- `deficiency_history`
- `digestive_issues`
- `pregnancy_or_postpartum_flags`
- `food_restrictions`
- `meal_timing_habits`
- `hydration_habits`
- `activity_level`
- `supplement_or_medication_mentions`
- `safety_flags`
- `personalization_notes`
- `extraction_source_evidence`
- `fallback_flags`
- `timestamps`

### Prompt Exposure

- `GET /prompt/system`
- `GET /prompt/user/latest`
- `POST /prompt/user`
- `GET /prompt/sections`
- `GET /prompt/assembled/latest`

`POST /prompt/user` stores the active user input from another application and can append it to chat history.

Example:

```json
{
  "raw_text": "Give me a vegetarian lunch idea without peanuts.",
  "source": "external_frontend",
  "append_to_chat": true
}
```

### RAG

- `POST /rag/upload`
- `GET /rag/documents`
- `POST /rag/retrieve`
- `GET /rag/retrieval/latest`
- `GET /rag/retrieval`
- `GET /rag/retrieval?query=protein+sources`

Supported upload formats:

- `.txt`
- `.md`
- `.pdf`
- `.docx`
- `.json`
- `.csv`

`POST /rag/upload` accepts multipart form data with a `file` field.

`POST /rag/retrieve` example:

```json
{
  "query": "protein sources for vegetarian meals",
  "top_k": 5
}
```

If no documents exist or no matches are found, the backend returns an empty but valid retrieved-knowledge section.

### Tools

- `GET /tools`
- `GET /tools/active`
- `GET /tools/schema`

The current tool catalog is hardcoded and includes placeholder definitions for:

- food nutrition lookup
- hydration calculator
- calorie estimator
- protein target estimator
- grocery recommendation
- meal plan helper
- allergen checker

### Generation

- `POST /generate`
- `GET /generate/latest`

`POST /generate` assembles the final LLM input from:

- nutrition system prompt
- latest user prompt
- chat history
- derived state and memory
- retrieved knowledge
- hardcoded tool definitions

By default, generation uses a local dummy response so the backend can be tested without an API key.

Example local generation:

```json
{
  "query": "Suggest a balanced dinner for me",
  "include_retrieved_knowledge": true
}
```

Example model-backed generation:

```json
{
  "query": "Suggest a balanced dinner for me",
  "include_retrieved_knowledge": true,
  "use_model": true,
  "model": "gpt-4.1-mini",
  "temperature": 0.2
}
```

### Temporary Compatibility Routes

These original RAG routes are preserved for compatibility:

- `GET /documents`
- `POST /upload`
- `POST /retrieve`

New integrations should prefer the `/rag/*` routes.

## JSON Response Shape

Most new endpoints return a consistent envelope:

```json
{
  "success": true,
  "message": "Human-readable status message.",
  "data": {},
  "metadata": {
    "timestamp": "2026-04-21T00:00:00+00:00",
    "session_id": "demo-nutrition-session",
    "app_name": "Nutrition Healthcare Assistant Backend",
    "domain": "nutrition_healthcare"
  }
}
```

## GET vs POST Design

GET endpoints are for reading state that another application can consume:

- chat history
- current session
- memory and state
- prompt sections
- latest assembled prompt
- uploaded document list
- latest retrieval result
- tool definitions
- latest generated result

POST endpoints are for actions that create or mutate runtime state:

- submitting a chat message
- setting the active user prompt
- uploading/ingesting documents
- retrieving with a request body
- generating assistant output

## State And Memory Derivation

Memory is rebuilt from chat history after new user input. The extraction is intentionally simple and local-development friendly.

It looks for nutrition-relevant signals such as:

- allergies
- intolerances
- disease history
- medical or nutrition-related conditions
- deficiencies
- digestive concerns
- pregnancy or postpartum context
- food restrictions
- supplements and medication mentions
- preferences and goals

The memory behaves like a maintained profile. Newer explicit corrections override older values. For example:

```text
I am no longer allergic to peanuts, but I have lactose intolerance.
```

This removes the seeded peanut allergy and adds lactose intolerance.

If no meaningful runtime user history exists, the backend falls back to seeded dummy nutrition values.

## What Is Real vs Dummy

Real:

- FastAPI backend
- JSON endpoint surface
- CORS configuration
- in-memory session state
- seeded chat history
- runtime chat mutation
- prompt assembly
- document upload
- RAG chunking and retrieval
- latest retrieval and generation state

Dummy or hardcoded for now:

- single session ID: `demo-nutrition-session`
- seeded nutrition chat history
- hardcoded nutrition system prompt
- hardcoded tool catalog
- local dummy assistant response
- rule-based memory extraction

Model-backed but environment-configured:

- `langchain-openai` client
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_VERIFY_SSL`

Future-replaceable:

- persistent session storage
- multi-session routing
- richer medical/nutrition entity extraction
- real tool execution
- stronger retrieval ranking
- production authentication and authorization

## Quick Frontend Integration Examples

Read chat history:

```js
const history = await fetch("http://127.0.0.1:8000/chat/history").then((res) => res.json());
```

Submit a chat message:

```js
const response = await fetch("http://127.0.0.1:8000/chat/message", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "I have lactose intolerance. Suggest a simple breakfast.",
    source: "frontend"
  })
}).then((res) => res.json());
```

Read latest memory:

```js
const memory = await fetch("http://127.0.0.1:8000/memory/latest").then((res) => res.json());
```

Generate a response:

```js
const generation = await fetch("http://127.0.0.1:8000/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "Give me a vegetarian high-protein lunch idea",
    include_retrieved_knowledge: true
  })
}).then((res) => res.json());
```

Upload a RAG document:

```js
const formData = new FormData();
formData.append("file", file);

const upload = await fetch("http://127.0.0.1:8000/rag/upload", {
  method: "POST",
  body: formData
}).then((res) => res.json());
```

Retrieve from RAG with a query string:

```js
const retrieval = await fetch("http://127.0.0.1:8000/rag/retrieval?query=protein+sources").then((res) => res.json());
```

## Manual Smoke Tests

After starting the backend:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/config
curl http://127.0.0.1:8000/chat/history
curl http://127.0.0.1:8000/memory/latest
curl http://127.0.0.1:8000/tools
```

Post a message:

```bash
curl -X POST http://127.0.0.1:8000/chat/message ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"I am no longer allergic to peanuts, but I have lactose intolerance and PCOS. Suggest breakfast.\"}"
```

Then inspect updated memory:

```bash
curl http://127.0.0.1:8000/memory/latest
```

## Notes

- This is a local/dev-friendly backend, not a production medical system.
- The assistant prompt tells the model to avoid diagnosis, prescriptions, unsafe restrictions, and medical overreach.
- Nutrition guidance should remain practical, evidence-aligned, and personalized to user-reported context.
- Users with urgent symptoms, serious conditions, pregnancy/postpartum needs, medication concerns, or eating disorder risk should be directed to qualified healthcare professionals.
