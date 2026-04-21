import { FormEvent, KeyboardEvent, ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { api, ChatMessage, DocumentSummary, GenerationData, MemoryProfile, PromptSections, RetrievalData, ToolDefinition } from './api';

type AdvancedTab = 'memory' | 'knowledge' | 'prompt' | 'tools' | 'debug';

const tabs: Array<{ id: AdvancedTab; label: string }> = [
  { id: 'memory', label: 'Memory' },
  { id: 'knowledge', label: 'Knowledge' },
  { id: 'prompt', label: 'Prompt' },
  { id: 'tools', label: 'Tools' },
  { id: 'debug', label: 'Debug' },
];

const emptyRetrieval: RetrievalData = {
  results: [],
  combined_text: '',
  message: 'No retrieval has been run yet.',
};

export default function App() {
  const didLoad = useRef(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [advancedOpen, setAdvancedOpen] = useStoredBoolean('nutrition.advancedOpen', false);
  const [activeTab, setActiveTab] = useStoredState<AdvancedTab>('nutrition.activeTab', 'memory');
  const [draft, setDraft] = useStoredState('nutrition.draft', '');

  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [session, setSession] = useState<Record<string, unknown> | null>(null);
  const [stateAndMemory, setStateAndMemory] = useState<Record<string, unknown> | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [memory, setMemory] = useState<MemoryProfile | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [retrieval, setRetrieval] = useState<RetrievalData>(emptyRetrieval);
  const [promptSections, setPromptSections] = useState<PromptSections | null>(null);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [assembledPrompt, setAssembledPrompt] = useState('');
  const [latestUserPrompt, setLatestUserPrompt] = useState<Record<string, unknown> | null>(null);
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [toolSchemas, setToolSchemas] = useState<ToolDefinition[]>([]);
  const [generation, setGeneration] = useState<GenerationData | null>(null);

  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isRetrieving, setIsRetrieving] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState('');
  const [lastFailedDraft, setLastFailedDraft] = useState('');
  const [uploadStatus, setUploadStatus] = useState('');
  const [retrievalQuery, setRetrievalQuery] = useState('');
  const [manualGenerateQuery, setManualGenerateQuery] = useState('');
  const [promptDraft, setPromptDraft] = useState('');
  const [rawMemoryOpen, setRawMemoryOpen] = useState(false);
  const [developerMemoryOpen, setDeveloperMemoryOpen] = useState(false);
  const [rawPromptOpen, setRawPromptOpen] = useState(false);
  const [rawDebugOpen, setRawDebugOpen] = useState(false);

  const connectionOk = Boolean(health);
  const latestAssistant = [...messages].reverse().find((message) => message.role === 'assistant');
  const userMessageCount = messages.filter((message) => message.role === 'user').length;

  useEffect(() => {
    if (didLoad.current) return;
    didLoad.current = true;
    void loadInitialData();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages.length, isSending]);

  async function loadInitialData() {
    setIsInitialLoading(true);
    setError('');
    const results = await Promise.allSettled([
      api.health(),
      api.config(),
      api.chatSession(),
      api.chatHistory(),
      api.memoryLatest(),
      api.stateAndMemory(),
      api.documents(),
      api.retrievalLatest(),
      api.promptSystem(),
      api.promptUserLatest(),
      api.promptSections(),
      api.promptAssembledLatest(),
      api.tools(),
      api.toolsSchema(),
      api.generateLatest(),
    ]);

    applySettled(results[0], (data) => setHealth(data));
    applySettled(results[1], (data) => setConfig(data));
    applySettled(results[2], (data) => setSession(data));
    applySettled(results[3], (data) => setMessages(safeArray(data.messages)));
    applySettled(results[4], (data) => setMemory(data));
    applySettled(results[5], (data) => setStateAndMemory(data));
    applySettled(results[6], (data) => setDocuments(safeArray(data.documents)));
    applySettled(results[7], (data) => setRetrieval(data || emptyRetrieval));
    applySettled(results[8], (data) => setSystemPrompt(data.system_prompt || ''));
    applySettled(results[9], (data) => setLatestUserPrompt(data));
    applySettled(results[10], (data) => setPromptSections(data));
    applySettled(results[11], (data) => setAssembledPrompt(data.prompt || ''));
    applySettled(results[12], (data) => setTools(safeArray(data.tools)));
    applySettled(results[13], (data) => setToolSchemas(safeArray(data.schemas)));
    applySettled(results[14], (data) => setGeneration(data));

    const firstFailure = results.find((result) => result.status === 'rejected');
    if (firstFailure?.status === 'rejected') {
      setError(`Some backend details could not be loaded: ${readError(firstFailure.reason)}`);
    }
    setIsInitialLoading(false);
  }

  async function refreshAfterConversation() {
    const results = await Promise.allSettled([api.chatHistory(), api.memoryLatest(), api.stateAndMemory(), api.generateLatest(), api.promptUserLatest(), api.promptSections(), api.promptAssembledLatest()]);
    applySettled(results[0], (data) => setMessages(safeArray(data.messages)));
    applySettled(results[1], (data) => setMemory(data));
    applySettled(results[2], (data) => setStateAndMemory(data));
    applySettled(results[3], (data) => setGeneration(data));
    applySettled(results[4], (data) => setLatestUserPrompt(data));
    applySettled(results[5], (data) => setPromptSections(data));
    applySettled(results[6], (data) => setAssembledPrompt(data.prompt || ''));
  }

  async function handleSend(event?: FormEvent, overrideText?: string) {
    event?.preventDefault();
    const text = (overrideText ?? draft).trim();
    if (!text || isSending) return;
    setIsSending(true);
    setError('');
    setLastFailedDraft('');

    try {
      await api.sendChatMessage(text);
      setDraft('');
      await refreshAfterConversation();
    } catch (sendError) {
      setError(readError(sendError));
      setLastFailedDraft(text);
    } finally {
      setIsSending(false);
    }
  }

  async function handlePromptSubmit() {
    const text = promptDraft.trim();
    if (!text) return;
    setError('');
    try {
      await api.setPromptUser(text);
      setPromptDraft('');
      await refreshAfterConversation();
    } catch (promptError) {
      setError(readError(promptError));
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  }

  async function handleUpload(file?: File) {
    if (!file || isUploading) return;
    setIsUploading(true);
    setError('');
    setUploadStatus(`Uploading ${file.name}...`);
    try {
      const response = await api.uploadDocument(file);
      setUploadStatus(response.message || `${file.name} uploaded.`);
      const docs = await api.documents();
      setDocuments(safeArray(docs.data.documents));
      const latestRetrieval = await api.retrievalLatest();
      setRetrieval(latestRetrieval.data || emptyRetrieval);
    } catch (uploadError) {
      setUploadStatus('');
      setError(readError(uploadError));
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function handleRetrieve(mode: 'get' | 'post' = 'post') {
    const query = retrievalQuery.trim() || draft.trim();
    if (!query || isRetrieving) {
      setError('Enter a retrieval query or type a draft message first.');
      return;
    }
    setIsRetrieving(true);
    setError('');
    try {
      const response = mode === 'get' ? await api.retrievalByQuery(query, 5) : await api.retrieve(query, 5);
      setRetrieval(response.data || emptyRetrieval);
    } catch (retrieveError) {
      setError(readError(retrieveError));
    } finally {
      setIsRetrieving(false);
    }
  }

  async function handleManualGenerate() {
    const query = manualGenerateQuery.trim();
    if (!query || isGenerating) return;
    setIsGenerating(true);
    setError('');
    try {
      const response = await api.generate({
        query,
        include_retrieved_knowledge: true,
        use_model: true,
        temperature: 0.2,
      });
      setGeneration(response.data);
      await refreshAfterConversation();
    } catch (generateError) {
      setError(readError(generateError));
    } finally {
      setIsGenerating(false);
    }
  }

  async function copyText(text?: string) {
    if (!text) return;
    await navigator.clipboard.writeText(text);
  }

  const memoryGroups = useMemo<Array<[string, unknown]>>(
    () => [
      ['Goals', memory?.nutrition_goals],
      ['Preferences', [...safeArray(memory?.dietary_preferences), ...safeArray(memory?.cuisine_preferences)]],
      ['Allergies', memory?.allergies],
      ['Intolerances', memory?.intolerances],
      ['Diseases / Conditions', [...safeArray(memory?.diseases_history), ...safeArray(memory?.specific_conditions)]],
      ['Deficiencies', memory?.deficiency_history],
      ['Digestive Concerns', memory?.digestive_issues],
      ['Restrictions', memory?.food_restrictions],
      ['Personalization Notes', memory?.personalization_notes],
    ],
    [memory],
  );

  return (
    <div className="min-h-screen bg-oat text-ink">
      <div className="mx-auto flex min-h-screen w-full max-w-[1440px] flex-col px-4 py-4 sm:px-6 lg:px-8">
        <header className="mb-4 flex flex-col gap-3 rounded-lg border border-leaf-900/10 bg-white/90 px-4 py-4 shadow-soft sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-leaf-700">Nutrition Healthcare Assistant</p>
            <h1 className="mt-1 text-2xl font-semibold tracking-normal text-ink sm:text-3xl">Nutrition Assistant</h1>
            <p className="mt-1 max-w-2xl text-sm text-slate-600">Personalized food, hydration, and nutrition guidance that remembers your preferences and health context.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill ok={connectionOk} label={connectionOk ? 'Backend connected' : 'Backend unavailable'} />
            <button className="icon-button" type="button" onClick={() => setAdvancedOpen(!advancedOpen)} title="Toggle advanced panel">
              <PanelIcon />
              <span>{advancedOpen ? 'Hide Advanced' : 'Advanced'}</span>
            </button>
          </div>
        </header>

        <main className={`grid flex-1 gap-4 ${advancedOpen ? 'lg:grid-cols-[minmax(0,1fr)_420px]' : 'lg:grid-cols-1'}`}>
          <section className="flex min-h-[70vh] flex-col overflow-hidden rounded-lg border border-leaf-900/10 bg-white shadow-soft">
            <div className="border-b border-slate-200 px-4 py-3 sm:px-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold">Today&apos;s nutrition chat</h2>
                  <p className="text-sm text-slate-500">{userMessageCount ? `${userMessageCount} user message${userMessageCount === 1 ? '' : 's'} in this session` : 'Ask about meals, allergies, goals, or nutrition habits to get started.'}</p>
                </div>
                {isInitialLoading ? <span className="loading-dot">Loading</span> : null}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto px-4 py-5 sm:px-6">
              {messages.length === 0 && !isInitialLoading ? (
                <EmptyState title="Ready when you are" description="Ask about meals, allergies, goals, or nutrition habits to get started." />
              ) : (
                <div className="space-y-4">
                  {messages.map((message) => (
                    <MessageBubble key={message.id} message={message} onCopy={message.role === 'assistant' ? () => void copyText(message.content) : undefined} />
                  ))}
                  {isSending ? <TypingBubble /> : null}
                </div>
              )}
              <div ref={scrollRef} />
            </div>

            <div className="border-t border-slate-200 bg-mint/50 px-4 py-4 sm:px-5">
              {error ? (
                <div className="mb-3 rounded-md border border-berry/20 bg-white px-3 py-2 text-sm text-berry">
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                    <span>{error}</span>
                    {lastFailedDraft ? (
                      <button className="text-button" type="button" onClick={() => void handleSend(undefined, lastFailedDraft)}>
                        Retry
                      </button>
                    ) : null}
                  </div>
                </div>
              ) : null}
              {uploadStatus ? <p className="mb-2 text-sm text-leaf-700">{uploadStatus}</p> : null}
              <form className="flex flex-col gap-3" onSubmit={(event) => void handleSend(event)}>
                <textarea
                  className="min-h-[92px] resize-none rounded-lg border border-slate-300 bg-white px-4 py-3 text-base outline-none transition focus:border-leaf-500 focus:ring-4 focus:ring-leaf-100"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  placeholder="Ask for meal ideas, allergy-safe swaps, hydration help, or nutrition habits..."
                  disabled={isSending}
                />
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex flex-wrap gap-2">
                    <input ref={fileInputRef} className="hidden" type="file" accept=".txt,.md,.pdf,.docx,.json,.csv" onChange={(event) => void handleUpload(event.target.files?.[0])} />
                    <button className="secondary-button" type="button" onClick={() => fileInputRef.current?.click()} disabled={isUploading || isSending} title="Upload knowledge document">
                      <UploadIcon />
                      <span>{isUploading ? 'Uploading' : 'Upload'}</span>
                    </button>
                    <button className="secondary-button" type="button" onClick={() => setDraft('')} disabled={!draft || isSending} title="Clear draft">
                      <ClearIcon />
                      <span>Clear</span>
                    </button>
                  </div>
                  <button className="primary-button" type="submit" disabled={!draft.trim() || isSending} title="Send message">
                    <SendIcon />
                    <span>{isSending ? 'Sending' : 'Send'}</span>
                  </button>
                </div>
              </form>
            </div>
          </section>

          {advancedOpen ? (
            <aside className="fixed inset-x-0 bottom-0 z-30 max-h-[86vh] overflow-hidden rounded-t-lg border border-slate-200 bg-white shadow-soft lg:static lg:max-h-none lg:rounded-lg">
              <AdvancedPanel
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                onClose={() => setAdvancedOpen(false)}
                memoryView={
                  <MemoryPanel
                    memory={memory}
                    memoryGroups={memoryGroups}
                    rawOpen={rawMemoryOpen}
                    developerOpen={developerMemoryOpen}
                    setRawOpen={setRawMemoryOpen}
                    setDeveloperOpen={setDeveloperMemoryOpen}
                  />
                }
                knowledgeView={
                  <KnowledgePanel
                    documents={documents}
                    retrieval={retrieval}
                    query={retrievalQuery}
                    setQuery={setRetrievalQuery}
                    isRetrieving={isRetrieving}
                    onRetrieve={handleRetrieve}
                  />
                }
                promptView={
                  <PromptPanel
                    latestUserPrompt={latestUserPrompt}
                    promptDraft={promptDraft}
                    setPromptDraft={setPromptDraft}
                    onPromptSubmit={handlePromptSubmit}
                    systemPrompt={systemPrompt}
                    sections={promptSections}
                    assembledPrompt={assembledPrompt}
                    rawOpen={rawPromptOpen}
                    setRawOpen={setRawPromptOpen}
                  />
                }
                toolsView={<ToolsPanel tools={tools} schemas={toolSchemas} />}
                debugView={
                  <DebugPanel
                    health={health}
                    config={config}
                    session={session}
                    stateAndMemory={stateAndMemory}
                    generation={generation}
                    query={manualGenerateQuery}
                    setQuery={setManualGenerateQuery}
                    isGenerating={isGenerating}
                    onGenerate={handleManualGenerate}
                    rawOpen={rawDebugOpen}
                    setRawOpen={setRawDebugOpen}
                  />
                }
              />
            </aside>
          ) : null}
        </main>

        <footer className="py-4 text-center text-xs text-slate-500">For urgent or serious medical issues, consult a qualified healthcare professional.</footer>
      </div>
    </div>
  );
}

function AdvancedPanel(props: {
  activeTab: AdvancedTab;
  setActiveTab: (tab: AdvancedTab) => void;
  onClose: () => void;
  memoryView: ReactNode;
  knowledgeView: ReactNode;
  promptView: ReactNode;
  toolsView: ReactNode;
  debugView: ReactNode;
}) {
  const viewMap: Record<AdvancedTab, ReactNode> = {
    memory: props.memoryView,
    knowledge: props.knowledgeView,
    prompt: props.promptView,
    tools: props.toolsView,
    debug: props.debugView,
  };

  return (
    <div className="flex h-full max-h-[86vh] flex-col lg:max-h-[calc(100vh-150px)]">
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
        <div>
          <h2 className="font-semibold">Advanced context</h2>
          <p className="text-sm text-slate-500">Memory, knowledge, prompts, tools, and debug details.</p>
        </div>
        <button className="icon-only-button" type="button" onClick={props.onClose} title="Close advanced panel">
          <CloseIcon />
        </button>
      </div>
      <div className="scrollbar-thin flex gap-2 overflow-x-auto border-b border-slate-200 px-3 py-2">
        {tabs.map((tab) => (
          <button key={tab.id} className={`tab-button ${props.activeTab === tab.id ? 'tab-button-active' : ''}`} type="button" onClick={() => props.setActiveTab(tab.id)}>
            {tab.label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto p-4">{viewMap[props.activeTab]}</div>
    </div>
  );
}

function MemoryPanel(props: {
  memory: MemoryProfile | null;
  memoryGroups: Array<[string, unknown]>;
  rawOpen: boolean;
  developerOpen: boolean;
  setRawOpen: (open: boolean) => void;
  setDeveloperOpen: (open: boolean) => void;
}) {
  const hasMemory = props.memoryGroups.some(([, values]) => safeArray(values).length > 0);
  return (
    <div className="space-y-4">
      {!hasMemory ? <EmptyState title="No memory yet" description="Your preferences and health-related nutrition context will appear here as you chat." /> : null}
      <div className="grid gap-3">
        {props.memoryGroups.map(([title, values]) => (
          <InfoCard key={title} title={title} values={safeArray(values)} />
        ))}
      </div>
      <ToggleRow label="View Raw JSON" checked={props.rawOpen} onChange={props.setRawOpen} />
      {props.rawOpen ? <CodeBlock value={props.memory} /> : null}
      <ToggleRow label="Developer View" checked={props.developerOpen} onChange={props.setDeveloperOpen} />
      {props.developerOpen ? <CodeBlock value={{ evidence: props.memory?.extraction_source_evidence, fallback: props.memory?.fallback_flags, timestamps: props.memory?.timestamps }} /> : null}
    </div>
  );
}

function KnowledgePanel(props: {
  documents: DocumentSummary[];
  retrieval: RetrievalData;
  query: string;
  setQuery: (query: string) => void;
  isRetrieving: boolean;
  onRetrieve: (mode: 'get' | 'post') => void;
}) {
  return (
    <div className="space-y-4">
      <PanelSection title="Uploaded documents">
        {props.documents.length ? (
          <div className="space-y-2">
            {props.documents.map((doc) => (
              <div key={doc.document_id} className="rounded-md border border-slate-200 bg-oat px-3 py-2">
                <p className="font-medium">{doc.file_name}</p>
                <p className="text-sm text-slate-500">{doc.chunk_count} chunks · {doc.content_type}</p>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState title="No documents" description="Upload a nutrition-related document to enhance answers." />
        )}
      </PanelSection>
      <PanelSection title="Manual retrieval">
        <textarea className="field" value={props.query} onChange={(event) => props.setQuery(event.target.value)} placeholder="Search uploaded nutrition knowledge..." rows={3} />
        <div className="mt-2 flex flex-wrap gap-2">
          <button className="secondary-button" type="button" onClick={() => props.onRetrieve('post')} disabled={props.isRetrieving}>
            <SearchIcon />
            <span>{props.isRetrieving ? 'Searching' : 'Retrieve'}</span>
          </button>
          <button className="secondary-button" type="button" onClick={() => props.onRetrieve('get')} disabled={props.isRetrieving}>
            <SearchIcon />
            <span>GET query</span>
          </button>
        </div>
      </PanelSection>
      <PanelSection title="Latest retrieval">
        {props.retrieval?.results?.length ? (
          <div className="space-y-3">
            {props.retrieval.results.map((result) => (
              <article key={result.chunk_id} className="rounded-md border border-slate-200 bg-white p-3">
                <div className="mb-2 flex items-center justify-between gap-2 text-sm">
                  <span className="font-medium">{result.file_name}</span>
                  <span className="rounded-full bg-mint px-2 py-1 text-xs text-leaf-700">Score {result.score}</span>
                </div>
                <p className="text-sm text-slate-600">{result.content}</p>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState title="No retrieval results" description={props.retrieval?.message || 'Run a retrieval query to inspect matching chunks.'} />
        )}
      </PanelSection>
    </div>
  );
}

function PromptPanel(props: {
  latestUserPrompt: Record<string, unknown> | null;
  promptDraft: string;
  setPromptDraft: (value: string) => void;
  onPromptSubmit: () => void;
  systemPrompt: string;
  sections: PromptSections | null;
  assembledPrompt: string;
  rawOpen: boolean;
  setRawOpen: (open: boolean) => void;
}) {
  return (
    <div className="space-y-4">
      <PanelSection title="Set active user prompt">
        <textarea className="field" value={props.promptDraft} onChange={(event) => props.setPromptDraft(event.target.value)} placeholder="Advanced only: store a prompt through POST /prompt/user..." rows={3} />
        <button className="secondary-button mt-2" type="button" onClick={props.onPromptSubmit} disabled={!props.promptDraft.trim()}>
          <SendIcon />
          <span>Store prompt</span>
        </button>
      </PanelSection>
      <PanelSection title="Latest user prompt">
        {props.latestUserPrompt ? <CodeBlock value={props.latestUserPrompt} compact /> : <EmptyState title="No prompt yet" description="Prompt details will appear after your first interaction." />}
      </PanelSection>
      <PanelSection title="System prompt">
        <CodeText text={props.systemPrompt || 'System prompt has not loaded yet.'} />
      </PanelSection>
      <ToggleRow label="Show full prompt sections" checked={props.rawOpen} onChange={props.setRawOpen} />
      {props.rawOpen ? (
        <>
          <PanelSection title="Prompt sections">
            <CodeBlock value={props.sections} />
          </PanelSection>
          <PanelSection title="Assembled prompt">
            <CodeText text={props.assembledPrompt || 'Assembled prompt will appear after interaction.'} />
          </PanelSection>
        </>
      ) : null}
    </div>
  );
}

function ToolsPanel({ tools, schemas }: { tools: ToolDefinition[]; schemas: ToolDefinition[] }) {
  const visibleTools = tools.length ? tools : schemas;
  return (
    <div className="space-y-3">
      {visibleTools.length ? (
        visibleTools.map((tool) => (
          <article key={tool.name} className="rounded-md border border-slate-200 bg-white p-3">
            <div className="flex items-center justify-between gap-2">
              <h3 className="font-semibold capitalize">{tool.name.replace(/_/g, ' ')}</h3>
              {tool.active !== undefined ? <span className="rounded-full bg-mint px-2 py-1 text-xs text-leaf-700">{tool.active ? 'Active' : 'Inactive'}</span> : null}
            </div>
            {tool.description ? <p className="mt-2 text-sm text-slate-600">{tool.description}</p> : null}
            <details className="mt-3">
              <summary className="cursor-pointer text-sm font-medium text-leaf-700">Schema</summary>
              <CodeBlock value={{ input_schema: tool.input_schema, output_schema: tool.output_schema }} compact />
            </details>
          </article>
        ))
      ) : (
        <EmptyState title="No tools loaded" description="Nutrition tool definitions will appear here when the backend responds." />
      )}
    </div>
  );
}

function DebugPanel(props: {
  health: Record<string, unknown> | null;
  config: Record<string, unknown> | null;
  session: Record<string, unknown> | null;
  stateAndMemory: Record<string, unknown> | null;
  generation: GenerationData | null;
  query: string;
  setQuery: (query: string) => void;
  isGenerating: boolean;
  onGenerate: () => void;
  rawOpen: boolean;
  setRawOpen: (open: boolean) => void;
}) {
  return (
    <div className="space-y-4">
      <PanelSection title="Manual generation">
        <textarea className="field" value={props.query} onChange={(event) => props.setQuery(event.target.value)} placeholder="Advanced test prompt for POST /generate..." rows={3} />
        <button className="primary-button mt-2" type="button" onClick={props.onGenerate} disabled={!props.query.trim() || props.isGenerating}>
          <SparkIcon />
          <span>{props.isGenerating ? 'Generating' : 'Run generate'}</span>
        </button>
      </PanelSection>
      <PanelSection title="Latest generation">
        <CodeText text={props.generation?.output_text || 'No generation has been run yet.'} />
      </PanelSection>
      <div className="grid grid-cols-2 gap-2">
        <StatusPill ok={Boolean(props.health)} label={props.health ? 'Health ok' : 'Health missing'} />
        <StatusPill ok={Boolean(props.config)} label={props.config ? 'Config loaded' : 'Config missing'} />
      </div>
      <ToggleRow label="Raw debug JSON" checked={props.rawOpen} onChange={props.setRawOpen} />
      {props.rawOpen ? <CodeBlock value={{ health: props.health, config: props.config, session: props.session, state_and_memory: props.stateAndMemory, generation: props.generation }} /> : null}
    </div>
  );
}

function MessageBubble({ message, onCopy }: { message: ChatMessage; onCopy?: () => void }) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <article className={`max-w-[88%] rounded-lg px-4 py-3 shadow-sm sm:max-w-[72%] ${isUser ? 'bg-leaf-700 text-white' : 'border border-slate-200 bg-white text-ink'}`}>
        <div className="mb-1 flex items-center justify-between gap-3">
          <span className={`text-xs font-semibold uppercase tracking-[0.12em] ${isUser ? 'text-leaf-100' : 'text-leaf-700'}`}>{isUser ? 'You' : 'Assistant'}</span>
          {onCopy ? (
            <button className="rounded p-1 text-slate-500 transition hover:bg-slate-100 hover:text-ink" type="button" onClick={onCopy} title="Copy assistant response">
              <CopyIcon />
            </button>
          ) : null}
        </div>
        <p className="whitespace-pre-wrap break-words text-sm leading-6 sm:text-[15px]">{message.content}</p>
      </article>
    </div>
  );
}

function TypingBubble() {
  return (
    <div className="flex justify-start">
      <div className="rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500 shadow-sm">Thinking through your nutrition context...</div>
    </div>
  );
}

function InfoCard({ title, values }: { title: string; values: unknown[] }) {
  return (
    <article className="rounded-md border border-slate-200 bg-white p-3">
      <h3 className="mb-2 text-sm font-semibold text-slate-700">{title}</h3>
      {values.length ? (
        <div className="flex flex-wrap gap-2">
          {values.map((value, index) => (
            <span key={`${String(value)}-${index}`} className="rounded-full bg-mint px-2.5 py-1 text-sm text-leaf-900">
              {String(value)}
            </span>
          ))}
        </div>
      ) : (
        <p className="text-sm text-slate-400">Not mentioned yet</p>
      )}
    </article>
  );
}

function PanelSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-md border border-slate-200 bg-slate-50/70 p-3">
      <h3 className="mb-3 text-sm font-semibold text-slate-700">{title}</h3>
      {children}
    </section>
  );
}

function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="rounded-md border border-dashed border-slate-300 bg-white/70 p-4 text-center">
      <p className="font-semibold">{title}</p>
      <p className="mt-1 text-sm text-slate-500">{description}</p>
    </div>
  );
}

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-medium ${ok ? 'border-leaf-500/20 bg-mint text-leaf-700' : 'border-berry/20 bg-white text-berry'}`}>
      <span className={`h-2 w-2 rounded-full ${ok ? 'bg-leaf-500' : 'bg-berry'}`} />
      {label}
    </span>
  );
}

function ToggleRow({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center justify-between rounded-md border border-slate-200 bg-white px-3 py-2 text-sm font-medium">
      <span>{label}</span>
      <input className="h-4 w-4 accent-leaf-700" type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function CodeBlock({ value, compact = false }: { value: unknown; compact?: boolean }) {
  return <pre className={`code-block ${compact ? 'max-h-48' : 'max-h-96'}`}>{JSON.stringify(value ?? {}, null, 2)}</pre>;
}

function CodeText({ text }: { text: string }) {
  return <pre className="code-block max-h-96">{text}</pre>;
}

function applySettled<T>(result: PromiseSettledResult<{ data: T }>, setter: (data: T) => void) {
  if (result.status === 'fulfilled') setter(result.value.data);
}

function safeArray<T = unknown>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function readError(error: unknown) {
  return error instanceof Error ? error.message : 'Something went wrong.';
}

function useStoredState<T>(key: string, initialValue: T): [T, (next: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored ? (JSON.parse(stored) as T) : initialValue;
    } catch {
      return initialValue;
    }
  });

  function update(next: T) {
    setValue(next);
    localStorage.setItem(key, JSON.stringify(next));
  }

  return [value, update];
}

function useStoredBoolean(key: string, initialValue: boolean): [boolean, (next: boolean) => void] {
  return useStoredState<boolean>(key, initialValue);
}

function SendIcon() {
  return <svg aria-hidden="true" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m22 2-7 20-4-9-9-4 20-7Z" /><path d="M22 2 11 13" /></svg>;
}
function UploadIcon() {
  return <svg aria-hidden="true" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M20 16v4H4v-4" /></svg>;
}
function CopyIcon() {
  return <svg aria-hidden="true" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></svg>;
}
function ClearIcon() {
  return <svg aria-hidden="true" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6 6 18" /><path d="m6 6 12 12" /></svg>;
}
function PanelIcon() {
  return <svg aria-hidden="true" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M15 4v16" /></svg>;
}
function CloseIcon() {
  return <svg aria-hidden="true" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6 6 18" /><path d="m6 6 12 12" /></svg>;
}
function SearchIcon() {
  return <svg aria-hidden="true" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></svg>;
}
function SparkIcon() {
  return <svg aria-hidden="true" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2 3 14h8l-1 8 11-14h-8l0-6Z" /></svg>;
}
