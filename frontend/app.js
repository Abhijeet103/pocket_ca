import React, { useEffect, useState } from "https://esm.sh/react@18";
import { createRoot } from "https://esm.sh/react-dom@18/client";

function App() {
  const [userId, setUserId] = useState("demo-user");
  const [sessionId, setSessionId] = useState("");
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Ask me about Indian tax law, deductions, or your tax calculation.",
    },
  ]);
  const [sending, setSending] = useState(false);
  const [rebuildStatus, setRebuildStatus] = useState("idle");

  async function loadHealth() {
    try {
      const response = await fetch("/api/health");
      const payload = await response.json();
      setRebuildStatus(payload.rebuild.status);
    } catch (error) {
      setRebuildStatus("unavailable");
    }
  }

  useEffect(() => {
    loadHealth();
  }, []);

  async function sendMessage(event) {
    event.preventDefault();
    const trimmed = message.trim();
    if (!trimmed || sending) {
      return;
    }

    const nextMessages = [...messages, { role: "user", content: trimmed }];
    setMessages(nextMessages);
    setMessage("");
    setSending(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: userId,
          session_id: sessionId || null,
          message: trimmed,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Could not get chatbot response.");
      }

      if (payload.session_id) {
        setSessionId(payload.session_id);
      }

      setMessages((current) => [
        ...current,
        { role: "assistant", content: payload.response },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `Error: ${error.message}`,
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  async function triggerRebuild() {
    try {
      setRebuildStatus("starting");
      const response = await fetch("/api/rebuild", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ clear_graph: true }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Could not trigger rebuild.");
      }

      setRebuildStatus(payload.status);
    } catch (error) {
      setRebuildStatus(`error: ${error.message}`);
    }
  }

  return (
    React.createElement("div", { className: "app-shell" },
      React.createElement("aside", { className: "sidebar" },
        React.createElement("h1", null, "PocketCA"),
        React.createElement("p", { className: "sidebar-copy" },
          "Graph-RAG Indian tax assistant with memory and tax tools."
        ),
        React.createElement("label", { className: "field-label" }, "User ID"),
        React.createElement("input", {
          className: "text-input",
          value: userId,
          onChange: (event) => setUserId(event.target.value),
          placeholder: "demo-user",
        }),
        React.createElement("label", { className: "field-label" }, "Session ID"),
        React.createElement("input", {
          className: "text-input",
          value: sessionId,
          onChange: (event) => setSessionId(event.target.value),
          placeholder: "auto-created on first message",
        }),
        React.createElement("button", {
          className: "secondary-button",
          onClick: triggerRebuild,
          type: "button",
        }, "Rebuild Knowledge Graph"),
        React.createElement("p", { className: "status-line" },
          `Rebuild status: ${rebuildStatus}`
        )
      ),
      React.createElement("main", { className: "chat-panel" },
        React.createElement("div", { className: "messages" },
          messages.map((entry, index) =>
            React.createElement("div", {
              key: `${entry.role}-${index}`,
              className: `message-bubble ${entry.role}`,
            },
              React.createElement("div", { className: "message-role" }, entry.role),
              React.createElement("div", { className: "message-content" }, entry.content)
            )
          )
        ),
        React.createElement("form", { className: "composer", onSubmit: sendMessage },
          React.createElement("textarea", {
            className: "composer-input",
            value: message,
            onChange: (event) => setMessage(event.target.value),
            placeholder: "Ask a tax-law or tax-calculation question...",
            rows: 4,
          }),
          React.createElement("button", {
            className: "primary-button",
            type: "submit",
            disabled: sending,
          }, sending ? "Sending..." : "Send")
        )
      )
    )
  );
}

createRoot(document.getElementById("root")).render(React.createElement(App));
