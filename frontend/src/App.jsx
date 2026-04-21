import { useEffect, useState } from 'react';
import {
    assemblePrompt,
    fetchDocuments,
    generateResponse,
    generateRetrievedKnowledge,
    generateStateAndMemory,
    generateSystemInstructions,
} from './api';

const sectionConfig = [
  {
    key: 'system_instructions',
    label: 'System Instructions',
    placeholder: 'Define the system-level rules or behavior for the model.',
  },
  {
    key: 'user_input',
    label: 'User Input',
    placeholder: 'Enter the current user request or test case.',
  },
  {
    key: 'conversation_history',
    label: 'Conversation History',
    placeholder: 'Add prior messages or turns to preserve continuity.',
  },
  {
    key: 'retrieved_knowledge',
    label: 'Retrieved Knowledge',
    placeholder: 'Retrieved context will appear here, or you can edit it manually.',
  },
  {
    key: 'tool_definitions',
    label: 'Tool Definitions',
    placeholder: 'Describe any available tools or tool constraints.',
  },
  {
    key: 'state_and_memory',
    label: 'State & Memory',
    placeholder: 'Add persistent background facts or state relevant to the test.',
  },
];

const initialSections = Object.fromEntries(sectionConfig.map((section) => [section.key, '']));

export default function App() {
  const [sections, setSections] = useState(initialSections);
  const [includeRetrievedKnowledge, setIncludeRetrievedKnowledge] = useState(true);
  const [retrievalQuery, setRetrievalQuery] = useState('');
  const [retrievalResults, setRetrievalResults] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [promptPreview, setPromptPreview] = useState('');
  const [modelResponse, setModelResponse] = useState('');
  const [modelName, setModelName] = useState('gpt-4.1-mini');
  const [temperature, setTemperature] = useState(0.2);
  const [statusMessage, setStatusMessage] = useState('Ready.');
  const [errorMessage, setErrorMessage] = useState('');
  const [isAssembling, setIsAssembling] = useState(false);
  const [isRetrieving, setIsRetrieving] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    loadDocuments();
  }, []);

  async function loadDocuments() {
    try {
      const data = await fetchDocuments();
      setDocuments(data.documents);
    } catch (error) {
      setErrorMessage(error.message);
    }
  }

  function updateSection(key, value) {
    setSections((current) => ({
      ...current,
      [key]: value,
    }));
  }

  async function handleAssemble() {
    setIsAssembling(true);
    setErrorMessage('');
    setStatusMessage('Assembling prompt preview...');
    try {
      const data = await assemblePrompt({
        sections,
        include_retrieved_knowledge: includeRetrievedKnowledge,
      });
      setPromptPreview(data.prompt);
      setStatusMessage(`Prompt assembled with ${data.included_sections.length} section(s).`);
    } catch (error) {
      setErrorMessage(error.message);
      setStatusMessage('Prompt assembly failed.');
    } finally {
      setIsAssembling(false);
    }
  }

  async function handleRetrieve() {
    const query = retrievalQuery.trim() || sections.user_input.trim();
    if (!query) {
      setErrorMessage('Enter a retrieval query or provide user input before retrieving.');
      return;
    }

    setIsRetrieving(true);
    setErrorMessage('');
    setStatusMessage('Searching indexed files for relevant context...');
    try {
      const data = await retrieveContext({ query, top_k: 5 });
      setRetrievalResults(data.results);
      updateSection('retrieved_knowledge', data.combined_text);
      setStatusMessage(data.message);
    } catch (error) {
      setErrorMessage(error.message);
      setStatusMessage('Retrieval failed.');
    } finally {
      setIsRetrieving(false);
    }
  }

  async function handleGenerate() {
    setIsGenerating(true);
    setErrorMessage('');
    setStatusMessage('Calling the model API...');
    try {
      const data = await generateResponse({
        sections,
        include_retrieved_knowledge: includeRetrievedKnowledge,
        model: modelName.trim() || null,
        temperature: Number(temperature),
      });
      setPromptPreview(data.prompt);
      setModelResponse(data.output_text);
      setStatusMessage(`Response generated with ${data.model_used}.`);
    } catch (error) {
      setErrorMessage(error.message);
      setStatusMessage('Model request failed.');
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleGenerateSection(sectionKey, sectionLabel) {
    const content = sections[sectionKey].trim();
    if (!content) {
      setErrorMessage(`Please provide content for ${sectionLabel} before generating.`);
      return;
    }

    setIsGenerating(true);
    setErrorMessage('');
    setStatusMessage(`Generating response for ${sectionLabel}...`);
    try {
      let data;
      switch (sectionKey) {
        case 'system_instructions':
          data = await generateSystemInstructions({
            content,
            model: modelName.trim() || null,
            temperature: Number(temperature),
          });
          break;
        case 'user_input':
          data = await generateUserInput({
            content,
            model: modelName.trim() || null,
            temperature: Number(temperature),
          });
          break;
        case 'retrieved_knowledge':
          data = await generateRetrievedKnowledge({
            content,
            model: modelName.trim() || null,
            temperature: Number(temperature),
          });
          break;
        case 'state_and_memory':
          data = await generateStateAndMemory({
            content,
            model: modelName.trim() || null,
            temperature: Number(temperature),
          });
          break;
        default:
          throw new Error('Unknown section');
      }
      setPromptPreview(data.prompt);
      setModelResponse(data.output_text);
      setStatusMessage(`Response generated for ${sectionLabel} with ${data.model_used}.`);
    } catch (error) {
      setErrorMessage(error.message);
      setStatusMessage('Model request failed.');
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleUpload(event) {
    const [file] = event.target.files || [];
    if (!file) {
      return;
    }

    setIsUploading(true);
    setErrorMessage('');
    setStatusMessage(`Uploading ${file.name}...`);
    try {
      const data = await uploadFile(file);
      await loadDocuments();
      setStatusMessage(`${data.message} ${data.chunks_indexed} chunk(s) indexed.`);
    } catch (error) {
      setErrorMessage(error.message);
      setStatusMessage('Upload failed.');
    } finally {
      event.target.value = '';
      setIsUploading(false);
    }
  }

  function clearAll() {
    setSections(initialSections);
    setPromptPreview('');
    setModelResponse('');
    setRetrievalResults([]);
    setRetrievalQuery('');
    setIncludeRetrievedKnowledge(true);
    setErrorMessage('');
    setStatusMessage('Cleared prompt sections and results.');
  }

  const busy = isUploading || isRetrieving || isAssembling || isGenerating;

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Prompt Testing Harness</p>
          <h1>Build, inspect, retrieve, and test prompts without hidden steps.</h1>
          <p className="hero-copy">
            Each prompt component stays separate and visible, retrieval is optional, and the final prompt is shown
            before or alongside model output.
          </p>
        </div>
        <div className="hero-actions">
          <button className="button secondary" type="button" onClick={handleAssemble} disabled={busy}>
            {isAssembling ? 'Assembling...' : 'Assemble Prompt'}
          </button>
          <button className="button primary" type="button" onClick={handleGenerate} disabled={busy}>
            {isGenerating ? 'Running...' : 'Generate Response'}
          </button>
        </div>
      </header>

      <main className="layout">
        <section className="panel stack">
          <div className="panel-header">
            <div>
              <h2>Prompt Sections</h2>
              <p>Keep each part separate so you can see exactly what reaches the model.</p>
            </div>
            <button className="text-button" type="button" onClick={clearAll}>
              Clear all
            </button>
          </div>

          {sectionConfig.map((section) => (
            <div key={section.key} className="field-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>{section.label}</span>
                {['system_instructions', 'user_input', 'retrieved_knowledge', 'state_and_memory'].includes(section.key) && (
                  <button
                    className="button secondary"
                    type="button"
                    onClick={() => handleGenerateSection(section.key, section.label)}
                    disabled={busy || !sections[section.key].trim()}
                    style={{ fontSize: '0.8em', padding: '0.25em 0.5em' }}
                  >
                    Test Section
                  </button>
                )}
              </div>
              <textarea
                value={sections[section.key]}
                onChange={(event) => updateSection(section.key, event.target.value)}
                placeholder={section.placeholder}
                rows={section.key === 'user_input' ? 5 : 6}
              />
            </div>
          ))}
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <div>
              <h2>RAG Workflow</h2>
              <p>Upload files, retrieve matching text, and decide whether to include it in the final prompt.</p>
            </div>
          </div>

          <div className="field-group">
            <span>Upload Documents</span>
            <label className="upload-box">
              <input type="file" accept=".txt,.md,.pdf,.docx,.json,.csv" onChange={handleUpload} disabled={busy} />
              <strong>{isUploading ? 'Uploading...' : 'Choose a file'}</strong>
              <small>Supported: txt, md, pdf, docx, json, csv</small>
            </label>
          </div>

          <div className="field-group">
            <span>Indexed Files</span>
            {documents.length === 0 ? (
              <div className="empty-state">No files uploaded yet.</div>
            ) : (
              <div className="document-list">
                {documents.map((document) => (
                  <article key={document.document_id} className="document-card">
                    <strong>{document.file_name}</strong>
                    <span>{document.chunk_count} chunk(s)</span>
                    <small>{document.content_type}</small>
                  </article>
                ))}
              </div>
            )}
          </div>

          <label className="field-group">
            <span>Retrieval Query</span>
            <textarea
              value={retrievalQuery}
              onChange={(event) => setRetrievalQuery(event.target.value)}
              placeholder="Leave this blank to retrieve using the User Input field."
              rows={4}
            />
          </label>

          <div className="inline-controls">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={includeRetrievedKnowledge}
                onChange={(event) => setIncludeRetrievedKnowledge(event.target.checked)}
              />
              Include retrieved knowledge in final prompt
            </label>

            <button className="button secondary" type="button" onClick={handleRetrieve} disabled={busy}>
              {isRetrieving ? 'Retrieving...' : 'Retrieve Context'}
            </button>
          </div>

          <div className="field-group">
            <span>Retrieved Results</span>
            {retrievalResults.length === 0 ? (
              <div className="empty-state">No retrieval results yet.</div>
            ) : (
              <div className="result-list">
                {retrievalResults.map((result) => (
                  <article key={result.chunk_id} className="result-card">
                    <div className="result-meta">
                      <strong>{result.file_name}</strong>
                      <span>Score: {result.score}</span>
                    </div>
                    <p>{result.content}</p>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <div>
              <h2>Model Run</h2>
              <p>Preview the exact prompt and send it through the backend to a real model API.</p>
            </div>
          </div>

          <div className="split-fields">
            <label className="field-group">
              <span>Model</span>
              <input value={modelName} onChange={(event) => setModelName(event.target.value)} placeholder="gpt-4.1-mini" />
            </label>

            <label className="field-group">
              <span>Temperature</span>
              <input
                type="number"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(event) => setTemperature(event.target.value)}
              />
            </label>
          </div>

          <div className="field-group">
            <span>Status</span>
            <div className="status-box">{statusMessage}</div>
          </div>

          {errorMessage ? (
            <div className="error-box" role="alert">
              {errorMessage}
            </div>
          ) : null}

          <div className="field-group">
            <span>Final Prompt Preview</span>
            <pre className="code-block">{promptPreview || 'Prompt preview will appear here.'}</pre>
          </div>

          <div className="field-group">
            <span>Model Response</span>
            <pre className="code-block">{modelResponse || 'Model output will appear here after generation.'}</pre>
          </div>
        </section>
      </main>
    </div>
  );
}
