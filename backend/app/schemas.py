from typing import Literal

from pydantic import BaseModel, Field, field_validator


PromptSectionName = Literal[
    "system_instructions",
    "user_input",
    "conversation_history",
    "retrieved_knowledge",
    "tool_definitions",
    "state_and_memory",
]


class PromptSections(BaseModel):
    system_instructions: str = ""
    user_input: str = ""
    conversation_history: str = ""
    retrieved_knowledge: str = ""
    tool_definitions: str = ""
    state_and_memory: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def coerce_none_to_empty(cls, value: str | None) -> str:
        return value or ""


class PromptAssemblyRequest(BaseModel):
    sections: PromptSections
    include_retrieved_knowledge: bool = True


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=10)


class RetrievedChunk(BaseModel):
    document_id: str
    file_name: str
    chunk_id: str
    score: float
    content: str


class RetrieveResponse(BaseModel):
    results: list[RetrievedChunk]
    combined_text: str
    message: str


class PromptAssemblyResponse(BaseModel):
    prompt: str
    included_sections: list[PromptSectionName]


class GenerateRequest(BaseModel):
    sections: PromptSections
    include_retrieved_knowledge: bool = True
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    prompt: str
    output_text: str
    model_used: str


class SingleSectionGenerateRequest(BaseModel):
    content: str = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    chunks_indexed: int
    message: str


class DocumentSummary(BaseModel):
    document_id: str
    file_name: str
    content_type: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]


class ChatMessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    source: str = "chat"
    timestamp: str | None = None


class PromptUserRequest(BaseModel):
    raw_text: str = Field(min_length=1, max_length=8000)
    source: str = "external_app"
    timestamp: str | None = None
    append_to_chat: bool = True


class NutritionGenerateRequest(BaseModel):
    query: str | None = Field(default=None, max_length=8000)
    include_retrieved_knowledge: bool = True
    top_k: int = Field(default=5, ge=1, le=10)
    use_model: bool = False
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
