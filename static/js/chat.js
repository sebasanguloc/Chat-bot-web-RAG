(() => {
  const chat = document.getElementById("chat");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const submitBtn = document.getElementById("chat-submit");
  const statusBadge = document.getElementById("status-badge");

  function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
  }

  function addMessage({ text, role, sources = [], meta = "" }) {
    const wrapper = document.createElement("div");
    wrapper.className = `message message--${role}`;

    const bubble = document.createElement("div");
    bubble.className = "message__bubble";
    bubble.textContent = text;
    wrapper.appendChild(bubble);

    if (sources.length > 0) {
      const details = document.createElement("details");
      details.className = "sources";
      const summary = document.createElement("summary");
      summary.textContent = `📎 Fuentes consultadas (${sources.length})`;
      details.appendChild(summary);

      sources.forEach((s) => {
        const item = document.createElement("div");
        item.className = "source-item";
        item.innerHTML = `
          <div class="source-item__title">${escapeHtml(s.libro)} — Pág. ${escapeHtml(String(s.pagina))}</div>
          <div class="source-item__excerpt">${escapeHtml(s.fragmento)}…</div>
        `;
        details.appendChild(item);
      });

      wrapper.appendChild(details);
    }

    if (meta) {
      const metaEl = document.createElement("div");
      metaEl.className = "message__meta";
      metaEl.textContent = meta;
      wrapper.appendChild(metaEl);
    }

    chat.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function addTypingIndicator() {
    const wrapper = document.createElement("div");
    wrapper.className = "message message--bot";
    wrapper.id = "typing-indicator";
    wrapper.innerHTML = `
      <div class="message__bubble">
        <span class="typing"><span></span><span></span><span></span></span>
      </div>
    `;
    chat.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
  }

  async function sendMessage(message) {
    submitBtn.disabled = true;
    const typingEl = addTypingIndicator();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      typingEl.remove();

      if (!res.ok) {
        addMessage({ text: data.error || "Error desconocido.", role: "error" });
        return;
      }

      addMessage({
        text: data.respuesta,
        role: "bot",
        sources: data.fuentes || [],
        meta: `~${data.tokens_contexto_aprox} tokens de contexto`,
      });
    } catch (err) {
      typingEl.remove();
      addMessage({
        text: "No se pudo conectar con el servidor. Intenta de nuevo en unos segundos.",
        role: "error",
      });
    } finally {
      submitBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    addMessage({ text: message, role: "user" });
    input.value = "";
    input.style.height = "auto";
    sendMessage(message);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  });

  async function checkHealth() {
    const start = Date.now();
    statusBadge.textContent = "Conectando…";
    statusBadge.className = "status-badge status-badge--loading";

    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      const elapsed = Date.now() - start;

      if (data.status === "ok") {
        statusBadge.textContent = `✓ ${data.fragmentos} fragmentos indexados`;
        statusBadge.className = "status-badge status-badge--ok";
      } else if (data.status === "sin_indice") {
        statusBadge.textContent = "⚠ Base de conocimiento vacía";
        statusBadge.className = "status-badge status-badge--error";
      } else {
        statusBadge.textContent = "⚠ Servicio no disponible";
        statusBadge.className = "status-badge status-badge--error";
      }

      if (elapsed > 5000) {
        addMessage({
          text: "El servicio estaba dormido (plan gratuito de Render) y acaba de despertar. Ya puedes preguntar con normalidad.",
          role: "bot",
        });
      }
    } catch (err) {
      statusBadge.textContent = "⚠ Despertando el servicio (~1 min)…";
      statusBadge.className = "status-badge status-badge--loading";
      setTimeout(checkHealth, 8000);
    }
  }

  checkHealth();
  input.focus();
})();
