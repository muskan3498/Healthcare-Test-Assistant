const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export type Role = 'user' | 'assistant' | 'system';

export interface Envelope<T> {
  success: boolean;
  message: string;
  data: T;
  metadata?: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  timestamp: string;
  session_id: string;
  source?: string;
  is_seed?: boolean;
}

export interface ChatHistoryData {
  messages: ChatMessage[];
}

export interface MemoryProfile {
  session_id?: string;
  nutrition_goals?: string[];
  dietary_preferences?: string[];
  cuisine_preferences?: string[];
  disliked_foods?: string[];
  allergies?: string[];
  intolerances?: string[];
  diseases_history?: string[];
  specific_conditions?: string[];
  deficiency_history?: string[];
  digestive_issues?: string[];
  pregnancy_or_postpartum_flags?: string[];
  food_restrictions?: string[];
  meal_timing_habits?: string[];
  hydration_habits?: string[];
  activity_level?: string;
  supplement_or_medication_mentions?: string[];
  safety_flags?: string[];
  personalization_notes?: string[];
  extraction_source_evidence?: unknown[];
  fallback_flags?: Record<string, unknown>;
  timestamps?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DocumentSummary {
  document_id: string;
  file_name: string;
  content_type: string;
  chunk_count: number;
}

export interface RetrievedChunk {
  document_id: string;
  file_name: string;
  chunk_id: string;
  score: number;
  content: string;
}

export interface RetrievalData {
  query?: string;
  results: RetrievedChunk[];
  combined_text: string;
  message: string;
  retrieved_at?: string;
}

export interface PromptSections {
  system_instructions?: string;
  user_input?: string;
  conversation_history?: string;
  retrieved_knowledge?: string;
  tool_definitions?: string;
  state_and_memory?: string;
}

export interface GenerationData {
  output_text?: string;
  model_used?: string | null;
  prompt?: string;
  included_sections?: string[];
  sections?: PromptSections;
  retrieved_knowledge?: RetrievalData;
  debug?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ToolDefinition {
  name: string;
  active?: boolean;
  description?: string;
  input_schema?: Record<string, unknown>;
  output_schema?: Record<string, unknown>;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<Envelope<T>> {
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json') ? await response.json() : await response.text();

  if (!response.ok) {
    throw new Error(formatError(payload));
  }

  if (payload && typeof payload === 'object' && 'data' in payload) {
    return payload as Envelope<T>;
  }

  return {
    success: true,
    message: 'Request completed.',
    data: payload as T,
  };
}

function formatError(payload: unknown): string {
  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = (payload as { detail: unknown }).detail;
    return typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2);
  }
  if (typeof payload === 'string') {
    return payload || 'Request failed.';
  }
  return 'Request failed.';
}

function jsonOptions(body: unknown): RequestInit {
  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
}

export const api = {
  baseUrl: API_BASE,
  health: () => request<Record<string, unknown>>('/health'),
  config: () => request<Record<string, unknown>>('/config'),
  chatHistory: () => request<ChatHistoryData>('/chat/history'),
  chatLatest: () => request<{ message: ChatMessage | null }>('/chat/history/latest'),
  chatSession: () => request<Record<string, unknown>>('/chat/session'),
  sendChatMessage: (text: string) => request('/chat/message', jsonOptions({ text, source: 'frontend' })),
  memoryLatest: () => request<MemoryProfile>('/memory/latest'),
  stateAndMemory: () => request<Record<string, unknown>>('/state-and-memory'),
  promptSystem: () => request<{ system_prompt: string }>('/prompt/system'),
  promptUserLatest: () => request<Record<string, unknown>>('/prompt/user/latest'),
  setPromptUser: (rawText: string) =>
    request('/prompt/user', jsonOptions({ raw_text: rawText, source: 'frontend', append_to_chat: true })),
  promptSections: () => request<PromptSections>('/prompt/sections'),
  promptAssembledLatest: () => request<{ prompt: string; included_sections?: string[]; sections?: PromptSections }>(
    '/prompt/assembled/latest',
  ),
  uploadDocument: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return request<Record<string, unknown>>('/rag/upload', { method: 'POST', body: formData });
  },
  documents: () => request<{ documents: DocumentSummary[] }>('/rag/documents'),
  retrieve: (query: string, topK = 5) => request<RetrievalData>('/rag/retrieve', jsonOptions({ query, top_k: topK })),
  retrievalLatest: () => request<RetrievalData>('/rag/retrieval/latest'),
  retrievalByQuery: (query: string, topK = 5) =>
    request<RetrievalData>(`/rag/retrieval?query=${encodeURIComponent(query)}&top_k=${topK}`),
  tools: () => request<{ tools: ToolDefinition[] }>('/tools'),
  activeTools: () => request<{ tools: ToolDefinition[] }>('/tools/active'),
  toolsSchema: () => request<{ schemas: ToolDefinition[] }>('/tools/schema'),
  generate: (body: { query: string; include_retrieved_knowledge: boolean; use_model: boolean; temperature?: number }) =>
    request<GenerationData>('/generate', jsonOptions(body)),
  generateLatest: () => request<GenerationData>('/generate/latest'),
};
