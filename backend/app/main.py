from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from .config import UPLOAD_DIR, settings
from .nutrition import (
    APP_DOMAIN,
    APP_NAME,
    DEFAULT_SESSION_ID,
    NUTRITION_SYSTEM_PROMPT,
    TOOL_CATALOG,
    add_message,
    assemble_latest_prompt,
    empty_retrieval,
    generate_local_response,
    get_memory,
    get_prompt_sections,
    get_session,
    json_envelope,
    refresh_memory,
    retrieve_knowledge,
)
from .rag import SUPPORTED_EXTENSIONS, combine_retrieval_results, rag_store
from .schemas import (
    ChatMessageRequest,
    DocumentsResponse,
    NutritionGenerateRequest,
    PromptUserRequest,
    RetrieveRequest,
    RetrieveResponse,
    UploadResponse,
)


app = FastAPI(
    title=APP_NAME,
    version="2.0.0",
    description="Local-dev nutrition-focused healthcare chat backend with in-memory chat, memory, prompt assembly, tools, and RAG.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def metadata(session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    return {"session_id": session_id, "app_name": APP_NAME, "domain": APP_DOMAIN}


def handle_upload(file_name: str, content_type: str, payload: bytes) -> UploadResponse:
    original_name = Path(file_name).name
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(payload) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file exceeds the 10 MB size limit.")

    try:
        document_id, chunk_count = rag_store.ingest(original_name, content_type or "", payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="The file could not be processed.") from exc

    # Real local file persistence for uploaded source files; all app/session state remains in memory.
    safe_name = f"{document_id}_{original_name}"
    (UPLOAD_DIR / safe_name).write_bytes(payload)

    return UploadResponse(
        document_id=document_id,
        file_name=original_name,
        chunks_indexed=chunk_count,
        message="File uploaded and indexed successfully.",
    )


def model_or_local_generate(prompt: str, request: NutritionGenerateRequest, user_text: str, memory: dict[str, Any], retrieval: dict[str, Any]) -> tuple[str, str]:
    if not request.use_model:
        return generate_local_response(user_text, memory, retrieval), "local-dummy-generator"
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured. Set use_model=false for local dummy generation.",
        )

    model_name = resolve_model_name(request.model)
    try:
        response = invoke_model(prompt, model_name, request.temperature)
        return response.content or "The model returned an empty response.", model_name
    except Exception as first_exc:
        compact_prompt = build_compact_model_prompt(user_text, memory, retrieval)
        try:
            response = invoke_model(compact_prompt, model_name, request.temperature)
            return response.content or "The model returned an empty response.", f"{model_name} (compact-retry)"
        except Exception as retry_exc:  # pragma: no cover
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Model API call failed.",
                    "first_error": safe_exception_summary(first_exc),
                    "retry_error": safe_exception_summary(retry_exc),
                    "model_base_url": settings.openai_base_url,
                    "model_name": model_name,
                    "ssl_verification": settings.openai_verify_ssl,
                    "timeout_seconds": settings.openai_timeout_seconds,
                },
            ) from retry_exc


def invoke_model(prompt: str, model_name: str, temperature: float):
    # Model-backed generation is currently wired for the GenAI Lab OpenAI-compatible endpoint.
    # Local dummy generation remains the default so development does not require network access.
    with httpx.Client(verify=settings.openai_verify_ssl, timeout=settings.openai_timeout_seconds) as http_client:
        llm = ChatOpenAI(
            base_url=settings.openai_base_url,
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=temperature,
            http_client=http_client,
            timeout=settings.openai_timeout_seconds,
        )
        return llm.invoke(prompt)


def resolve_model_name(requested_model: str | None) -> str:
    if not requested_model:
        return settings.openai_model
    normalized = requested_model.strip()
    # Swagger/OpenAPI shows "string" as a placeholder. Treat it as unset so
    # test calls from /docs use the configured GenAI Lab model instead.
    if not normalized or normalized.lower() in {"string", "model", "optional-model-name"}:
        return settings.openai_model
    return normalized


def build_compact_model_prompt(user_text: str, memory: dict[str, Any], retrieval: dict[str, Any]) -> str:
    context = {
        "allergies": memory.get("allergies", []),
        "intolerances": memory.get("intolerances", []),
        "diseases_history": memory.get("diseases_history", []),
        "specific_conditions": memory.get("specific_conditions", []),
        "deficiency_history": memory.get("deficiency_history", []),
        "digestive_issues": memory.get("digestive_issues", []),
        "food_restrictions": memory.get("food_restrictions", []),
        "retrieved_knowledge": retrieval.get("combined_text", "")[:2000],
    }
    return (
        f"{NUTRITION_SYSTEM_PROMPT}\n\n"
        f"User question:\n{user_text}\n\n"
        f"Relevant user nutrition/health memory:\n{context}\n\n"
        "Answer warmly and practically. Do not diagnose or prescribe."
    )


def safe_exception_summary(exc: Exception) -> dict[str, Any]:
    cause = getattr(exc, "__cause__", None)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "cause_type": type(cause).__name__ if cause else None,
        "cause_message": str(cause) if cause else None,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return json_envelope(True, "Nutrition assistant backend is healthy.", {"status": "ok"}, metadata())


@app.get("/config")
def config() -> dict[str, Any]:
    return json_envelope(
        True,
        "Application configuration loaded.",
        {
            "app_name": APP_NAME,
            "domain": APP_DOMAIN,
            "allowed_origins": settings.cors_origins,
            "active_modules": ["chat", "state_memory", "prompting", "rag", "tools", "generation"],
            "environment": {
                "single_session": True,
                "default_session_id": DEFAULT_SESSION_ID,
                "memory_persistence": "in_memory",
                "uploaded_file_storage": "local_filesystem",
                "model_generation_default": "local_dummy",
                "model_provider": "langchain_openai",
                "model_base_url": settings.openai_base_url,
                "model_name": settings.openai_model,
                "model_ssl_verification": settings.openai_verify_ssl,
                "max_upload_size_bytes": settings.max_upload_size_bytes,
                "supported_upload_extensions": sorted(SUPPORTED_EXTENSIONS),
            },
        },
        metadata(),
    )


@app.get("/chat/history")
def chat_history() -> dict[str, Any]:
    session = get_session()
    return json_envelope(True, "Current chat history returned.", {"messages": session["chat_history"]}, metadata())


@app.get("/chat/history/latest")
def chat_history_latest() -> dict[str, Any]:
    session = get_session()
    latest = session["chat_history"][-1] if session["chat_history"] else None
    return json_envelope(True, "Latest chat message returned.", {"message": latest}, metadata())


@app.get("/chat/session")
def chat_session() -> dict[str, Any]:
    session = get_session()
    return json_envelope(
        True,
        "Current demo session returned.",
        {
            "session_id": session["session_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "message_count": len(session["chat_history"]),
            "latest_user_prompt": session["latest_user_prompt"],
        },
        metadata(),
    )


@app.post("/chat/message")
def chat_message(request: ChatMessageRequest) -> dict[str, Any]:
    user_message = add_message("user", request.text, request.source, request.timestamp)
    retrieval = retrieve_knowledge(request.text, settings.retrieval_limit)
    assembled = assemble_latest_prompt(include_retrieved_knowledge=True)
    memory = get_memory()
    generation_request = NutritionGenerateRequest(
        query=request.text,
        include_retrieved_knowledge=True,
        top_k=settings.retrieval_limit,
        use_model=True,
    )
    assistant_text, model_used = model_or_local_generate(
        assembled["prompt"],
        generation_request,
        request.text,
        memory,
        retrieval,
    )
    assistant_message = add_message("assistant", assistant_text, model_used)

    generation = {
        "output_text": assistant_text,
        "model_used": model_used,
        "assistant_message": assistant_message,
        "prompt": assembled["prompt"],
        "included_sections": assembled["included_sections"],
        "sections": assembled["sections"],
        "retrieved_knowledge": retrieval,
        "debug": {
            "used_model_api": True,
            "used_local_dummy_generation": False,
            "memory_fallback_flags": memory["fallback_flags"],
            "source_endpoint": "/chat/message",
        },
    }
    get_session()["latest_generation"] = generation

    return json_envelope(
        True,
        "User message appended, memory refreshed, and assistant response generated.",
        {"user_message": user_message, "assistant_message": assistant_message, "memory": memory, "generation": generation},
        metadata(),
    )


@app.get("/memory")
def memory() -> dict[str, Any]:
    return json_envelope(True, "Current maintained nutrition memory returned.", get_memory(), metadata())


@app.get("/memory/latest")
def memory_latest() -> dict[str, Any]:
    return json_envelope(True, "Latest maintained nutrition memory returned.", get_memory(), metadata())


@app.get("/state")
def state() -> dict[str, Any]:
    session = get_session()
    memory = get_memory()
    return json_envelope(
        True,
        "Current conversation state returned.",
        {
            "session_id": session["session_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "latest_user_prompt": session["latest_user_prompt"],
            "message_count": len(session["chat_history"]),
            "fallback_flags": memory["fallback_flags"],
            "safety_flags": memory["safety_flags"],
        },
        metadata(),
    )


@app.get("/state-and-memory")
def state_and_memory() -> dict[str, Any]:
    session = get_session()
    return json_envelope(
        True,
        "State and memory returned together.",
        {
            "state": {
                "session_id": session["session_id"],
                "message_count": len(session["chat_history"]),
                "latest_user_prompt": session["latest_user_prompt"],
            },
            "memory": get_memory(),
        },
        metadata(),
    )


@app.get("/prompt/system")
def prompt_system() -> dict[str, Any]:
    return json_envelope(True, "Nutrition system prompt returned.", {"system_prompt": NUTRITION_SYSTEM_PROMPT}, metadata())


@app.get("/prompt/user/latest")
def prompt_user_latest() -> dict[str, Any]:
    session = get_session()
    return json_envelope(True, "Latest user prompt returned.", session["latest_user_prompt"], metadata())


@app.post("/prompt/user")
def prompt_user(request: PromptUserRequest) -> dict[str, Any]:
    session = get_session()
    prompt = {
        "raw_text": request.raw_text,
        "timestamp": request.timestamp or session.get("updated_at"),
        "session_id": DEFAULT_SESSION_ID,
        "source": request.source,
    }
    session["latest_user_prompt"] = prompt
    if request.append_to_chat:
        add_message("user", request.raw_text, request.source, request.timestamp)
    else:
        refresh_memory()
    assembled = assemble_latest_prompt()
    return json_envelope(
        True,
        "Latest user prompt stored and prompt sections refreshed.",
        {"latest_user_prompt": get_session()["latest_user_prompt"], "assembled_prompt": assembled},
        metadata(),
    )


@app.get("/prompt/sections")
def prompt_sections() -> dict[str, Any]:
    return json_envelope(True, "Prompt sections returned.", get_prompt_sections(), metadata())


@app.get("/prompt/assembled/latest")
def prompt_assembled_latest() -> dict[str, Any]:
    assembled = assemble_latest_prompt()
    return json_envelope(True, "Latest assembled prompt returned.", assembled, metadata())


@app.post("/rag/upload", response_model=dict)
async def rag_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please choose a file to upload.")
    upload = handle_upload(file.filename, file.content_type or "", await file.read())
    return json_envelope(True, upload.message, upload.model_dump(), metadata())


@app.get("/rag/documents")
def rag_documents() -> dict[str, Any]:
    return json_envelope(True, "Uploaded RAG documents returned.", {"documents": [doc.model_dump() for doc in rag_store.list_documents()]}, metadata())


@app.post("/rag/retrieve")
def rag_retrieve(request: RetrieveRequest) -> dict[str, Any]:
    retrieval = retrieve_knowledge(request.query, request.top_k)
    return json_envelope(True, retrieval["message"], retrieval, metadata())


@app.get("/rag/retrieval/latest")
def rag_retrieval_latest() -> dict[str, Any]:
    session = get_session()
    return json_envelope(True, "Latest retrieval result returned.", session.get("latest_retrieval") or empty_retrieval(), metadata())


@app.get("/rag/retrieval")
def rag_retrieval(query: str | None = Query(default=None), top_k: int = Query(default=5, ge=1, le=10)) -> dict[str, Any]:
    if not query:
        session = get_session()
        return json_envelope(True, "Latest retrieval result returned because no query was supplied.", session.get("latest_retrieval") or empty_retrieval(), metadata())
    retrieval = retrieve_knowledge(query, top_k)
    return json_envelope(True, retrieval["message"], retrieval, metadata())


@app.get("/tools")
def tools() -> dict[str, Any]:
    return json_envelope(True, "Nutrition tool catalog returned.", {"tools": TOOL_CATALOG}, metadata())


@app.get("/tools/active")
def tools_active() -> dict[str, Any]:
    active = [tool for tool in TOOL_CATALOG if tool.get("active")]
    return json_envelope(True, "Active nutrition tools returned.", {"tools": active}, metadata())


@app.get("/tools/schema")
def tools_schema() -> dict[str, Any]:
    schemas = [
        {
            "name": tool["name"],
            "input_schema": tool["input_schema"],
            "output_schema": tool["output_schema"],
        }
        for tool in TOOL_CATALOG
    ]
    return json_envelope(True, "Nutrition tool schemas returned.", {"schemas": schemas}, metadata())


@app.post("/generate")
def generate(request: NutritionGenerateRequest | None = Body(default=None)) -> dict[str, Any]:
    request = request or NutritionGenerateRequest()
    session = get_session()
    user_text = request.query or session["latest_user_prompt"]["raw_text"]
    if request.query:
        add_message("user", request.query, "generate")

    retrieval = retrieve_knowledge(user_text, request.top_k) if request.include_retrieved_knowledge else empty_retrieval(user_text)
    assembled = assemble_latest_prompt(include_retrieved_knowledge=request.include_retrieved_knowledge)
    memory = get_memory()
    output_text, model_used = model_or_local_generate(assembled["prompt"], request, user_text, memory, retrieval)
    assistant_message = add_message("assistant", output_text, model_used)

    generation = {
        "output_text": output_text,
        "model_used": model_used,
        "assistant_message": assistant_message,
        "prompt": assembled["prompt"],
        "included_sections": assembled["included_sections"],
        "sections": assembled["sections"],
        "retrieved_knowledge": retrieval,
        "debug": {
            "used_model_api": request.use_model,
            "used_local_dummy_generation": not request.use_model,
            "memory_fallback_flags": memory["fallback_flags"],
        },
    }
    session["latest_generation"] = generation
    return json_envelope(True, "Generation completed and stored as the latest result.", generation, metadata())


@app.get("/generate/latest")
def generate_latest() -> dict[str, Any]:
    session = get_session()
    latest = session.get("latest_generation")
    if latest is None:
        latest = {
            "output_text": "",
            "model_used": None,
            "prompt": session.get("latest_assembled_prompt", ""),
            "message": "No generation has been run yet.",
        }
    return json_envelope(True, "Latest generation result returned.", latest, metadata())


# Temporary compatibility aliases for the original generic RAG routes.
@app.get("/documents", response_model=DocumentsResponse)
def list_documents_legacy() -> DocumentsResponse:
    return DocumentsResponse(documents=rag_store.list_documents())


@app.post("/upload", response_model=UploadResponse)
async def upload_document_legacy(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please choose a file to upload.")
    return handle_upload(file.filename, file.content_type or "", await file.read())


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_context_legacy(request: RetrieveRequest) -> RetrieveResponse:
    results = rag_store.retrieve(request.query, request.top_k)
    return RetrieveResponse(
        results=results,
        combined_text=combine_retrieval_results(results) if results else "",
        message=f"Retrieved {len(results)} relevant chunk(s)." if results else "No relevant retrieved knowledge was found for this query.",
    )
