import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import "./styles.css";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function refreshMetrics(setMetrics) {
  try {
    setMetrics(await apiRequest("/metrics"));
  } catch {
    // Metrics are optional and should not hide a successful upload or answer.
  }
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || `Request failed (${response.status})`);
  }
  return payload;
}

function App() {
  const [file, setFile] = useState(null);
  const [document, setDocument] = useState(null);
  const [question, setQuestion] = useState("");
  const [answerLength, setAnswerLength] = useState("detailed");
  const [answer, setAnswer] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    refreshMetrics(setMetrics);
  }, []);

  async function uploadDocument(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    setAnswer(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const payload = await apiRequest("/documents/upload", { method: "POST", body: formData });
      setDocument(payload.data);
      await refreshMetrics(setMetrics);
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setBusy(false);
    }
  }

  async function askQuestion(event) {
    event.preventDefault();
    if (!document || !question.trim()) return;
    setBusy(true);
    setError("");
    try {
      const payload = await apiRequest("/qa/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: document.document_id, question, answer_length: answerLength }),
      });
      setAnswer(payload.data);
      await refreshMetrics(setMetrics);
    } catch (questionError) {
      setError(questionError.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <nav className="topbar">
        <a className="brand" href="/">DOC<span>IQ</span></a>
        <div className="nav-actions">
          <span className="api-status"><i /> API connected</span>
          <a className="swagger-link" href={`${API_URL}/docs`} target="_blank" rel="noreferrer">Open Swagger</a>
        </div>
      </nav>

      <section className="intro">
        <p className="eyebrow">Document intelligence workspace</p>
        <h1>Ask your documents<br /><em>better questions.</em></h1>
        <p className="lede">Upload a PDF or text file, then query its contents with retrieval-backed answers.</p>
      </section>

      {error && <div className="alert">{error}</div>}

      <section className="workspace-grid">
        <div className="panel upload-panel">
          <div className="panel-heading"><span className="step">01</span><div><h2>Add a document</h2><p>PDF or TXT, up to 10 MB</p></div></div>
          <form onSubmit={uploadDocument}>
            <label className={`dropzone ${file ? "has-file" : ""}`}>
              <input type="file" accept=".pdf,.txt,application/pdf,text/plain" onChange={(event) => setFile(event.target.files[0])} />
              <span className="upload-mark">+</span>
              <strong>{file ? file.name : "Choose a document"}</strong>
              <small>{file ? "Ready to index" : "or drag and drop it here"}</small>
            </label>
            <button className="primary-button" type="submit" disabled={!file || busy}>{busy ? "Indexing..." : "Upload and index"}<span>→</span></button>
          </form>
          {document && <div className="document-chip"><span className="file-icon">TXT</span><div><strong>{document.filename}</strong><small>{document.chunk_count} chunks indexed; full document available</small></div><span className="check">✓</span></div>}
        </div>

        <div className="panel question-panel">
          <div className="panel-heading"><span className="step">02</span><div><h2>Ask a question</h2><p>{document ? "Your document is ready" : "Upload a document first"}</p></div></div>
          <form onSubmit={askQuestion} className="question-form">
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} disabled={!document || busy} placeholder="What would you like to understand?" rows="5" />
            <div className="answer-options" role="group" aria-label="Answer length">
              <span>Answer length</span>
              <button type="button" className={answerLength === "concise" ? "selected" : ""} onClick={() => setAnswerLength("concise")} disabled={busy}>Concise</button>
              <button type="button" className={answerLength === "detailed" ? "selected" : ""} onClick={() => setAnswerLength("detailed")} disabled={busy}>Detailed</button>
            </div>
            <button className="primary-button" type="submit" disabled={!document || !question.trim() || busy}>{busy ? "Thinking..." : "Get an answer"}<span>↗</span></button>
          </form>
        </div>
      </section>

      {answer && <section className="answer-panel">
        <div className="answer-heading">
          <div><p className="eyebrow">Document response</p><h2>Answer</h2></div>
          <div className="answer-meta">{answer.cached && <span>Cached</span>}<span>{answer.chunks_used} source chunks</span></div>
        </div>
        <div className="answer-content"><ReactMarkdown>{answer.answer}</ReactMarkdown></div>
        <footer className="answer-footer">Full document context used</footer>
      </section>}

      <section className="bottom-row">
        <div><p className="eyebrow">System pulse</p><h2>Built for a clear first pass.</h2></div>
        <div className="stats"><div><strong>{metrics?.counters?.upload_requests ?? 0}</strong><span>Uploads</span></div><div><strong>{metrics?.counters?.qa_requests ?? 0}</strong><span>Questions</span></div><div><strong>{metrics?.counters?.answers_generated ?? 0}</strong><span>Answers</span></div></div>
      </section>
      <footer className="site-footer"><span>DOCIQ / Document Q&amp;A</span><span>API: {API_URL}</span></footer>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
