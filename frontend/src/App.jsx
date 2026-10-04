import { useState } from "react";
import "./App.css";

const API_URL = "https://nmamit-rag-api.bravestone-36922c0d.centralindia.azurecontainerapps.io";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    const userQuestion = question.trim();

    if (!userQuestion || loading) return;

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: userQuestion,
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: userQuestion,
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
        },
      ]);
    } catch (error) {
      console.error("RAG API error:", error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Unable to connect to the NMAMIT RAG service. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askQuestion();
    }
  };

  const useSuggestion = (text) => {
    setQuestion(text);
  };

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <div className="brand">
          <div className="brand-icon">N</div>

          <div>
            <h1>NMAMIT RAG</h1>
            <p>Hybrid Graph Retrieval-Augmented Generation</p>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Online
        </div>
      </header>

      {/* MAIN */}
      <main className="chat-container">
        {messages.length === 0 ? (
          <section className="welcome">
            <div className="hero-icon">N</div>

            <h2>Ask about NMAMIT</h2>

            <p>
              Search the NMAMIT knowledge base using a hybrid Graph RAG
              architecture powered by Neo4j, Qdrant and Groq.
            </p>

            <div className="suggestions">
              <button
                onClick={() =>
                  useSuggestion("Where is NMAMIT located?")
                }
              >
                Where is NMAMIT located?
              </button>

              <button
                onClick={() =>
                  useSuggestion("What courses are offered at NMAMIT?")
                }
              >
                What courses are offered?
              </button>

              <button
                onClick={() =>
                  useSuggestion(
                    "Tell me about the Information Science department."
                  )
                }
              >
                Information Science department
              </button>

              <button
                onClick={() =>
                  useSuggestion("What is NMAMIT known for?")
                }
              >
                What is NMAMIT known for?
              </button>
            </div>
          </section>
        ) : (
          <section className="messages">
            {messages.map((message, index) => (
              <div
                className={`message ${message.role}`}
                key={`${message.role}-${index}`}
              >
                <div className="message-label">
                  {message.role === "user" ? "You" : "NMAMIT RAG"}
                </div>

                <div className="message-content">
                  {message.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="message assistant">
                <div className="message-label">NMAMIT RAG</div>

                <div className="message-content typing">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      {/* INPUT */}
      <div className="input-area">
        <div className="input-box">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask something about NMAMIT..."
            rows={1}
            disabled={loading}
          />

          <button
            className="send-button"
            onClick={askQuestion}
            disabled={loading || !question.trim()}
            aria-label="Send question"
          >
            ↑
          </button>
        </div>

        <p className="footer">
          NMAMIT Hybrid Graph RAG · Neo4j · Qdrant · Groq
        </p>
      </div>
    </div>
  );
}

export default App;