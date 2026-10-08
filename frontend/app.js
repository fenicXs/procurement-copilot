const chat = document.getElementById("chat");
const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");

const ABSTAIN_MARKER = "I don't have sufficiently grounded information";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderQuestion(question) {
  const turn = el("div", "turn");
  turn.appendChild(el("div", "bubble question", question));
  chat.appendChild(turn);
  chat.scrollTop = chat.scrollHeight;
  return turn;
}

function renderAnswer(turn, data) {
  const abstained = data.answer.includes(ABSTAIN_MARKER);

  const answerBubble = el(
    "div",
    "bubble answer" + (abstained ? " abstained" : ""),
    data.answer
  );
  turn.appendChild(answerBubble);

  if (!abstained) {
    const status = el(
      "span",
      "status " + (data.is_verified ? "verified" : "unverified"),
      data.is_verified ? "Grounded in evidence" : "Not fully verified"
    );
    turn.appendChild(status);
  }

  if (data.citations && data.citations.length > 0) {
    const citations = el("div", "citations");
    for (const c of data.citations) {
      const label = c.section_heading
        ? `${c.section_heading} (p. ${c.page_start}-${c.page_end})`
        : `p. ${c.page_start}-${c.page_end}`;
      citations.appendChild(el("span", "citation-chip", label));
    }
    turn.appendChild(citations);
  }

  chat.scrollTop = chat.scrollHeight;
}

function renderError(turn, message) {
  turn.appendChild(el("div", "bubble answer error-bubble", message));
  chat.scrollTop = chat.scrollHeight;
}

async function ask(question) {
  const turn = renderQuestion(question);
  askBtn.disabled = true;

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      renderError(turn, body.detail || "Something went wrong. Please try again.");
      return;
    }

    const data = await res.json();
    if (data.error) {
      renderError(turn, data.error);
      return;
    }
    renderAnswer(turn, data);
  } catch (err) {
    renderError(turn, "Could not reach the server. Please try again.");
  } finally {
    askBtn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  ask(question);
});

document.querySelectorAll(".example").forEach((btn) => {
  btn.addEventListener("click", () => {
    input.value = btn.textContent;
    form.requestSubmit();
  });
});
