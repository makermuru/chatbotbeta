// Replace this after deploying the Render service.
const API_BASE_URL = "https://YOUR-RENDER-SERVICE.onrender.com";

const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const messages = document.querySelector("#messages");
const status = document.querySelector("#connection-status");

function appendMessage(role, text) {
  const message = document.createElement("article");
  message.className = `message ${role}`;
  const sender = document.createElement("strong");
  sender.textContent = role === "user" ? "나" : "챗봇";
  const body = document.createElement("p");
  body.textContent = text;
  message.append(sender, body);
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error("Health check failed");
    status.textContent = "서버 연결됨";
    status.className = "status online";
  } catch {
    status.textContent = "서버 연결 실패 — API 주소를 확인하세요.";
    status.className = "status offline";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  appendMessage("user", question);
  input.value = "";
  input.disabled = true;
  try {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) throw new Error("Request failed");
    const data = await response.json();
    appendMessage("bot", data.answer);
  } catch {
    appendMessage("bot", "서버에 연결하지 못했어요. 잠시 후 다시 시도해주세요.");
  } finally {
    input.disabled = false;
    input.focus();
  }
});

checkHealth();
