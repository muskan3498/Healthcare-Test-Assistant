const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json') ? await response.json() : await response.text();

  if (!response.ok) {
    const detail =
      typeof payload === 'object' && payload !== null && 'detail' in payload
        ? payload.detail
        : 'Request failed.';
    throw new Error(detail);
  }

  return payload;
}

export function fetchDocuments() {
  return request('/documents');
}

export function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  return request('/upload', {
    method: 'POST',
    body: formData,
  });
}

export function retrieveContext(body) {
  return request('/retrieve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export function assemblePrompt(body) {
  return request('/assemble', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export function generateResponse(body) {
  return request('/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export function generateSystemInstructions(body) {
  return request('/generate/system-instructions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export function generateUserInput(body) {
  return request('/generate/user-input', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export function generateRetrievedKnowledge(body) {
  return request('/generate/retrieved-knowledge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export function generateStateAndMemory(body) {
  return request('/generate/state-and-memory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}
