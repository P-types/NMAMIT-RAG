import { useState, useEffect } from "react";
import "./App.css";

const API_URL =
  "https://nmamit-rag-api.bravestone-36922c0d.centralindia.azurecontainerapps.io";

const BACKGROUNDS = [
  {
    name: "Grey Dots",
    className: "bg-grey-dots",
  },
  {
    name: "Dark",
    className: "bg-dark",
  },
  {
    name: "Light",
    className: "bg-light",
  },
  {
    name: "Plain Grey",
    className: "bg-grey",
  },
];

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const [background, setBackground] = useState(() => {
    return localStorage.getItem("nmamit-background") || "bg-grey-dots";
  });

  const [showBackgrounds, setShowBackgrounds] = useState(false);

  useEffect(() => {
    document.body.className = background;
    localStorage.setItem("nmamit-background", background);

    return () => {
      document.body.className = "";
    };
  }, [background]);

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

  return (
    <div className="app">
      {/* ================================================
          HEADER
      ================================================= */}

      <header className="header">
        <div className="brand">
          <div className="brand-mark">N</div>

          <div>
            <h1>NMAMIT RAG</h1>
            <span>Hybrid Graph Retrieval System</span>
          </div>
        </div>

        {/* ================================================
            BACKGROUND SWITCHER
        ================================================= */}

        <div className="background-switcher">
          <button
            className="background-button"
            onClick={() => setShowBackgrounds((prev) => !prev)}
            title="Change background"
          >
            <span className="background-icon">◐</span>
            <span>Background</span>
          </button>

          {showBackgrounds && (
            <div className="background-menu">
              <div className="background-menu-title">
                Choose background
              </div>

              {BACKGROUNDS.map((item) => (
                <button
                  key={item.className}
                  className={`background-option ${
                    background === item.className ? "active" : ""
                  }`}
                  onClick={() => {
                    setBackground(item.className);
                    setShowBackgrounds(false);
                  }}
                >
                  <span
                    className={`background-preview ${item.className}`}
                  />

                  <span>{item.name}</span>

                  {background === item.className && (
                    <span className="check">✓</span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </header>

      {/* ================================================
          MAIN CHAT
      ================================================= */}

      <main className="chat-container">
        {messages.length === 0 ? (
          <section className="welcome">
            <div className="welcome-icon">✦</div>

            <h2>NMAMIT Hybrid Graph RAG</h2>

            <p>
              Ask questions about NMAMIT and retrieve answers
              using the hybrid Graph RAG system.
            </p>

            <div className="suggestions">
              <button
                onClick={() => {
                  setQuestion("What is NMAMIT?");
                }}
              >
                What is NMAMIT?
              </button>

              <button
                onClick={() => {
                  setQuestion(
                    "What departments are available at NMAMIT?"
                  );
                }}
              >
                NMAMIT departments
              </button>

              <button
                onClick={() => {
                  setQuestion(
                    "Tell me about the Information Science and Engineering department."
                  );
                }}
              >
                Information Science & Engineering
              </button>
            </div>
          </section>
        ) : (
          <section className="messages">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message-row ${message.role}`}
              >
                <div className="message-avatar">
                  {message.role === "user" ? "U" : "N"}
                </div>

                <div className="message-content">
                  <div className="message-role">
                    {message.role === "user"
                      ? "You"
                      : "NMAMIT RAG"}
                  </div>

                  <div className="message-text">
                    {message.content}
                  </div>
                </div>
              </div>
            ))}

            {loading && (
              <div className="message-row assistant">
                <div className="message-avatar">N</div>

                <div className="message-content">
                  <div className="message-role">NMAMIT RAG</div>

                  <div className="typing">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      {/* ================================================
          INPUT
      ================================================= */}

      <div className="input-area">
        <div className="input-wrapper">
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
            disabled={!question.trim() || loading}
          >
            {loading ? "..." : "↑"}
          </button>
        </div>

        <div className="input-hint">
          Press Enter to ask • Shift + Enter for a new line
        </div>
      </div>
    </div>
  );
}

export default App;