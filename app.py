import os
import re
import json
import tempfile
import asyncio
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from groq import Groq
from duckduckgo_search import DDGS
import edge_tts

VERSION = "1.2.1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title=f"Lova AI v{VERSION}")

# simple memory (resets when server restarts)
SESSIONS = {}

VOICES = {
    "female": "en-US-AriaNeural",
    "male": "en-US-GuyNeural",
    "female_hindi": "hi-IN-SwaraNeural",
    "male_hindi": "hi-IN-MadhurNeural",
}


def is_hindi(text: str) -> bool:
    return sum(1 for c in text if "\u0900" <= c <= "\u097F") >= 3


def protect_owner(text: str) -> bool:
    t = (text or "").lower()
    keys = [
        "owner name", "your owner", "who is your owner", "creator name", "saksham",
        "owner address", "owner phone", "owner school", "personal information of owner"
    ]
    return any(k in t for k in keys)


def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return ""
        return "\n".join([f"- {r.get('title','')}: {r.get('body','')}" for r in results])
    except Exception:
        return ""


async def tts(text: str, voice_choice: str = "female") -> Optional[str]:
    try:
        if is_hindi(text):
            voice = VOICES["female_hindi"] if voice_choice == "female" else VOICES["male_hindi"]
        else:
            voice = VOICES["female"] if voice_choice == "female" else VOICES["male"]
        path = tempfile.mktemp(suffix=".mp3")
        await edge_tts.Communicate(text, voice=voice).save(path)
        return path
    except Exception:
        return None


def ask_llm(user_text: str, history: list, name: str) -> str:
    if protect_owner(user_text):
        return "You are not allowed to ask about my owner's personal name tag or private personal information."

    keys = ["who", "what", "when", "where", "price", "stock", "news", "today", "current", "kya", "kaun"]
    extra = search_web(user_text) if any(k in user_text.lower() for k in keys) else ""

    system = f"""
You are Lova AI v{VERSION}, premium assistant by Saksham Jha.
User name-tag: {name}
Reply in user's language (English/Hindi).
For accounting questions: educational help only, not official CA advice.
Never reveal private owner personal details.
Refuse illegal harmful requests briefly.
Keep answers clear and useful.
"""
    messages = [{"role": "system", "content": system}] + history[-8:]
    content = f"User: {user_text}\n\nInfo:\n{extra}\n\nAnswer:" if extra else user_text
    messages.append({"role": "user", "content": content})

    try:
        r = client.chat.completions.create(messages=messages, model="openai/gpt-oss-20b")
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, error: {e}"


HTML_PAGE = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Lova AI v{VERSION}</title>
  <style>
    body {{
      margin:0; font-family: Inter, system-ui, sans-serif;
      background: radial-gradient(1000px 500px at 0% 0%, rgba(212,175,55,.12), transparent 50%),
                  linear-gradient(180deg,#070709,#121218);
      color:#f2f2f2; min-height:100vh;
    }}
    .wrap {{ max-width:900px; margin:0 auto; padding:24px 16px; }}
    .card {{
      background:rgba(255,255,255,.04); border:1px solid rgba(212,175,55,.35);
      border-radius:18px; padding:20px; box-shadow:0 20px 50px rgba(0,0,0,.35);
    }}
    h1 {{ color:#d4af37; letter-spacing:3px; margin:0 0 6px; font-size:2rem; }}
    .muted {{ color:#9a9a9a; margin-bottom:16px; }}
    input, select, textarea, button {{
      width:100%; box-sizing:border-box; margin:8px 0; padding:12px;
      border-radius:10px; border:1px solid #333; background:#0f0f14; color:#fff;
    }}
    button {{
      background:linear-gradient(135deg,#d4af37,#f0d789); color:#111;
      border:none; font-weight:700; cursor:pointer;
    }}
    #chat {{
      height:320px; overflow:auto; background:#0b0b10; border-radius:12px;
      padding:12px; border:1px solid #222; margin:10px 0;
      white-space:pre-wrap; line-height:1.45;
    }}
    .row {{ display:flex; gap:10px; }}
    .row > * {{ flex:1; }}
    .hidden {{ display:none; }}
    .loro {{
      margin-top:14px; padding:12px; border:1px dashed rgba(212,175,55,.5);
      border-radius:12px; color:#d4af37;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div id="gate" class="card">
      <h1>LOVA</h1>
      <div class="muted">Premium Voice AI · Update {VERSION}</div>
      <label>Name Tag</label>
      <input id="name" placeholder="Enter your name tag"/>
      <label>Voice</label>
      <select id="voice">
        <option value="female">Female</option>
        <option value="male">Male</option>
      </select>
      <button onclick="enterApp()">Enter Lova</button>
      <p id="gateMsg" class="muted"></p>
    </div>

    <div id="main" class="card hidden">
      <h1>LOVA <span style="font-size:0.9rem;color:#aaa">v{VERSION}</span></h1>
      <div class="muted" id="welcome"></div>
      <div id="chat">No messages yet.</div>
      <textarea id="msg" rows="3" placeholder="Type your message..."></textarea>
      <div class="row">
        <button onclick="sendMsg()">Send</button>
        <button onclick="clearChat()" style="background:#333;color:#fff">Clear</button>
      </div>
      <audio id="player" controls style="width:100%; margin-top:10px;"></audio>
      <div class="loro" onclick="loro()">
        🔒 Loro (Locked) — opens at Update 2.1.1 (current {VERSION})
      </div>
      <p id="loroMsg" class="muted"></p>
    </div>
  </div>

<script>
let sessionId = Math.random().toString(36).slice(2);

function enterApp() {{
  const name = document.getElementById('name').value.trim();
  const voice = document.getElementById('voice').value;
  if(!name) {{ document.getElementById('gateMsg').innerText = 'Please enter a name tag.'; return; }}
  fetch('/api/login', {{
    method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{session_id: sessionId, name, voice}})
  }}).then(r=>r.json()).then(d=>{{
    document.getElementById('gate').classList.add('hidden');
    document.getElementById('main').classList.remove('hidden');
    document.getElementById('welcome').innerText = d.message;
  }});
}}

function sendMsg() {{
  const message = document.getElementById('msg').value.trim();
  if(!message) return;
  fetch('/api/chat', {{
    method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{session_id: sessionId, message}})
  }}).then(r=>r.json()).then(d=>{{
    document.getElementById('chat').innerText = d.history;
    document.getElementById('msg').value = '';
    if(d.audio_url) {{
      const p = document.getElementById('player');
      p.src = d.audio_url;
      p.play();
    }}
  }});
}}

function clearChat() {{
  fetch('/api/clear', {{
    method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{session_id: sessionId}})
  }}).then(r=>r.json()).then(d=>{{
    document.getElementById('chat').innerText = d.history;
  }});
}}

function loro() {{
  document.getElementById('loroMsg').innerText =
    'Loro is locked. It will open at Update 2.1.1. Current update is {VERSION}.';
}}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.post("/api/login")
async def login(data: dict):
    sid = data.get("session_id")
    name = (data.get("name") or "").strip()
    voice = data.get("voice") or "female"
    if not sid or not name:
        return JSONResponse({"ok": False, "message": "Name tag required"}, status_code=400)
    SESSIONS[sid] = {"name": name, "voice": voice, "history": []}
    return {"ok": True, "message": f"Welcome, {name}. Lova v{VERSION} is ready."}


@app.post("/api/chat")
async def chat(data: dict):
    sid = data.get("session_id")
    message = (data.get("message") or "").strip()
    sess = SESSIONS.get(sid)
    if not sess:
        return JSONResponse({"error": "Login first"}, status_code=401)
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    reply = ask_llm(message, sess["history"], sess["name"])
    sess["history"].append({"role": "user", "content": message})
    sess["history"].append({"role": "assistant", "content": reply})

    audio_path = await tts(reply, sess.get("voice", "female"))
    audio_url = None
    if audio_path:
        # expose temp file once
        token = os.path.basename(audio_path)
        SESSIONS.setdefault("_audio", {})[token] = audio_path
        audio_url = f"/api/audio/{token}"

    hist = "\n\n".join(
        [("You: " if m["role"] == "user" else "Lova: ") + m["content"] for m in sess["history"][-12:]]
    )
    return {"reply": reply, "history": hist, "audio_url": audio_url}


@app.get("/api/audio/{token}")
async def audio(token: str):
    path = SESSIONS.get("_audio", {}).get(token)
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="audio/mpeg")


@app.post("/api/clear")
async def clear(data: dict):
    sid = data.get("session_id")
    sess = SESSIONS.get(sid)
    if sess:
        sess["history"] = []
    return {"history": "No messages yet."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
