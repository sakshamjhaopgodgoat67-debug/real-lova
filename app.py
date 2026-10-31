import gradio as gr
import os
import json
import tempfile
import asyncio
import concurrent.futures
from datetime import datetime

import numpy as np
import scipy.io.wavfile as wav
from groq import Groq
from duckduckgo_search import DDGS
import edge_tts

VERSION = "1.2.1"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

REVIEWS_FILE = "reviews.json"
user_chats = {}          # name -> chat history
user_profile = {"name": None, "voice": "female"}

VOICES = {
    "female": "en-US-AriaNeural",
    "male": "en-US-GuyNeural",
    "female_hindi": "hi-IN-SwaraNeural",
    "male_hindi": "hi-IN-MadhurNeural",
}

COMMANDS = {
    "grok": "https://grok.com",
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "instagram": "https://www.instagram.com",
    "chatgpt": "https://chatgpt.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "lova": "https://real-lova.onrender.com",
}


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(timeout=120)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def is_hindi(text: str) -> bool:
    return sum(1 for c in text if "\u0900" <= c <= "\u097F") >= 3


def protect_owner(text: str) -> bool:
    t = (text or "").lower()
    keys = [
        "owner name", "your owner", "who is your owner", "creator name", "saksham",
        "owner address", "owner phone", "owner school", "personal information of owner",
        "owner's personal", "tell me about your owner personally"
    ]
    return any(k in t for k in keys)


def find_open_command(text: str):
    t = (text or "").lower()
    for name, url in COMMANDS.items():
        for p in [f"open {name}", f"{name} kholo", f"kholo {name}", f"launch {name}"]:
            if p in t:
                return name, url
    return None, None


def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return ""
        return "\n".join([f"- {r.get('title','')}: {r.get('body','')}" for r in results])
    except Exception:
        return ""


def load_reviews():
    try:
        if os.path.exists(REVIEWS_FILE):
            with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_reviews(reviews):
    try:
        with open(REVIEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(reviews[:40], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def format_reviews():
    reviews = load_reviews()
    if not reviews:
        return "_No reviews yet. Be the first._"
    lines = []
    for r in reviews[:12]:
        stars = "★" * int(r["stars"]) + "☆" * (5 - int(r["stars"]))
        lines.append(f"**{stars}** — **{r['name']}** ({r['time']})\n{r['comment']}")
    return "\n\n".join(lines)


def submit_review(stars, comment, name):
    if not stars:
        return "Please choose stars (1–5).", format_reviews()
    reviews = load_reviews()
    reviews.insert(0, {
        "stars": int(stars),
        "comment": (comment or "").strip() or "No comment",
        "name": (name or user_profile.get("name") or "Anonymous").strip(),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    save_reviews(reviews)
    return "Thank you for your review!", format_reviews()


def transcribe(audio):
    if audio is None:
        return ""
    try:
        sr, data = audio
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        m = np.max(np.abs(data))
        if m > 0:
            data = data / m
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, sr, (data * 32767).astype(np.int16))
            path = f.name
        with open(path, "rb") as file:
            tr = client.audio.transcriptions.create(
                file=(path, file.read()),
                model="whisper-large-v3",
                language=None,
            )
        os.unlink(path)
        return (tr.text or "").strip()
    except Exception as e:
        return f"Transcription error: {e}"


async def speak(text, voice_choice="female"):
    try:
        if is_hindi(text):
            v = VOICES["female_hindi"] if voice_choice == "female" else VOICES["male_hindi"]
        else:
            v = VOICES["female"] if voice_choice == "female" else VOICES["male"]
        path = tempfile.mktemp(suffix=".mp3")
        await edge_tts.Communicate(text, voice=v).save(path)
        return path
    except Exception:
        return None


def format_history(name):
    hist = user_chats.get(name, [])
    if not hist:
        return f"No messages yet for **{name}**. Use Voice or Text mode."
    return "\n\n".join([
        ("**You:** " if m["role"] == "user" else "**Lova:** ") + m["content"]
        for m in hist[-12:]
    ])


def ask_lova(text, voice_choice):
    name = user_profile.get("name")
    text = (text or "").strip()
    if not text:
        return "Please type or speak something.", format_history(name), None, ""

    history = user_chats.setdefault(name, [])

    if protect_owner(text):
        reply = "You are not allowed to ask about my owner's personal name tag or private personal information."
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        user_chats[name] = history
        return reply, format_history(name), run_async(speak(reply, voice_choice)), ""

    cmd, url = find_open_command(text)
    if cmd and url:
        reply = f"Opening {cmd.capitalize()} for you, {name}."
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        user_chats[name] = history
        html = f'<script>window.open("{url}","_blank");</script><p>Opening <a href="{url}" target="_blank">{cmd}</a></p>'
        return reply, format_history(name), run_async(speak(reply, voice_choice)), html

    keys = ["who", "what", "when", "where", "price", "stock", "news", "today", "current", "kya", "kaun", "kab"]
    extra = search_web(text) if any(k in text.lower() for k in keys) else ""

    system = f"""You are Lova AI v{VERSION}, a dark premium assistant by Saksham Jha.
User name-tag: {name}
Call the user by their name-tag sometimes.
Reply in the user's language (English or Hindi).
Educational accounting help only (not official CA advice).
Never reveal private owner personal details.
Refuse illegal harmful requests briefly.
Be useful and clear."""

    messages = [{"role": "system", "content": system}] + history[-8:]
    messages.append({"role": "user", "content": f"User: {text}\n\nInfo:\n{extra}\n\nAnswer:" if extra else text})

    try:
        r = client.chat.completions.create(messages=messages, model="openai/gpt-oss-20b")
        reply = r.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Sorry, error: {e}"

    history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    user_chats[name] = history
    audio = run_async(speak(reply, voice_choice))
    return reply, format_history(name), audio, ""


def login(name, voice):
    name = (name or "").strip()
    if not name:
        return gr.update(visible=True), gr.update(visible=False), "Please enter a name tag.", "female", ""
    user_profile["name"] = name
    user_profile["voice"] = voice or "female"
    user_chats.setdefault(name, [])
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        f"Welcome, **{name}**. Lova v{VERSION} is ready.\n(Your name tag is your identity in this app.)",
        voice or "female",
        format_history(name),
    )


def process_text(msg, voice):
    try:
        return ask_lova(msg, voice or "female")
    except Exception as e:
        return f"Error: {e}", format_history(user_profile.get("name") or "guest"), None, ""


def process_voice(audio, voice):
    try:
        text = transcribe(audio)
        if not text:
            return "", "I could not hear you.", format_history(user_profile.get("name") or "guest"), None, ""
        reply, hist, audio_path, html = ask_lova(text, voice or "female")
        return text, reply, hist, audio_path, html
    except Exception as e:
        return "", f"Error: {e}", format_history(user_profile.get("name") or "guest"), None, ""


def clear_chat():
    name = user_profile.get("name")
    if name:
        user_chats[name] = []
    return "", "Chat cleared.", None, ""


# ======================
# DARK UI
# ======================
css = """
.gradio-container {
  max-width: 100% !important;
  background: #050507 !important;
  color: #e6e6e6 !important;
}
footer {display:none !important;}
.dark-card {
  background: #0d0d12 !important;
  border: 1px solid #222 !important;
}
button.primary {
  background: #2a2a32 !important;
  color: #f0f0f0 !important;
  border: 1px solid #444 !important;
  font-weight: 600 !important;
}
"""

with gr.Blocks(title=f"Lova AI v{VERSION}", css=css, theme=gr.themes.Base()) as demo:
    voice_state = gr.State("female")

    # Gate
    with gr.Column(visible=True) as gate:
        gr.HTML(f"""
        <div style="text-align:center;padding:48px 16px;">
          <h1 style="margin:0;letter-spacing:6px;color:#ddd;">LOVA</h1>
          <p style="color:#777;margin:8px 0 0;">Dark Premium AI · v{VERSION}</p>
          <p style="color:#666;margin-top:14px;">Enter your Name Tag (this is your identity)</p>
        </div>
        """)
        name_in = gr.Textbox(label="Name Tag", placeholder="Your name")
        voice_in = gr.Radio(["female", "male"], value="female", label="Reply voice")
        login_btn = gr.Button("Enter Lova", variant="primary")
        gate_msg = gr.Markdown("")

    # Main
    with gr.Column(visible=False) as main:
        gr.HTML(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
          padding:12px 14px;border:1px solid #222;border-radius:12px;margin-bottom:12px;background:#0d0d12;">
          <div>
            <div style="letter-spacing:3px;color:#ddd;font-weight:700;">LOVA</div>
            <div style="color:#666;font-size:12px;">Voice · Text · Privacy · Reviews</div>
          </div>
          <div style="color:#888;">v{VERSION}</div>
        </div>
        """)
        status = gr.Markdown("Welcome")

        with gr.Tab("🎤 Voice mode"):
            gr.Markdown("Record → **Send**")
            audio_in = gr.Audio(sources=["microphone"], type="numpy", format="wav", label="Microphone")
            heard = gr.Textbox(label="You said", interactive=False)
            v_btn = gr.Button("Send", variant="primary")

        with gr.Tab("💬 Text mode"):
            text_in = gr.Textbox(label="Message", lines=3, placeholder="English or Hindi... (open Grok)")
            t_btn = gr.Button("Send", variant="primary")

        with gr.Tab("⭐ Review"):
            gr.Markdown("Rate Lova and leave a suggestion")
            with gr.Row():
                rev_name = gr.Textbox(label="Name", placeholder="Optional")
                rev_stars = gr.Radio(["1", "2", "3", "4", "5"], value="5", label="Stars")
            rev_comment = gr.Textbox(label="Compliment / suggestion", lines=3)
            rev_btn = gr.Button("Submit review", variant="primary")
            rev_status = gr.Markdown("")
            rev_list = gr.Markdown(format_reviews())

        history = gr.Markdown("No messages yet.")
        reply = gr.Textbox(label="Lova", lines=5)
        audio_out = gr.Audio(label="Voice reply", autoplay=True)
        open_html = gr.HTML()
        clear_btn = gr.Button("Clear chat")

        gr.Markdown(
            """
            **Privacy**
            - Your name tag is used as identity in this session
            - Owner personal private info questions are blocked
            - Do not share passwords / OTPs / bank details
            """
        )

    login_btn.click(login, [name_in, voice_in], [gate, main, status, voice_state, history])

    t_btn.click(process_text, [text_in, voice_state], [reply, history, audio_out, open_html])
    text_in.submit(process_text, [text_in, voice_state], [reply, history, audio_out, open_html])
    v_btn.click(process_voice, [audio_in, voice_state], [heard, reply, history, audio_out, open_html])
    clear_btn.click(clear_chat, outputs=[reply, history, audio_out, open_html])
    rev_btn.click(submit_review, [rev_stars, rev_comment, rev_name], [rev_status, rev_list])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
