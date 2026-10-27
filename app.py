import gradio as gr
import os
import json
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import asyncio
from datetime import datetime
from groq import Groq
from duckduckgo_search import DDGS
import edge_tts

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

chat_history = []
REVIEWS_FILE = "reviews.json"

WELCOME_TEXT = (
    "Hello, I am Lova. Welcome to Lova, a smart voice AI assistant. "
    "Here is how you can use me. "
    "Go to the Voice tab, record your message, then press Send. "
    "Or type in the Text tab. "
    "You can say things like open Grok, or open YouTube. "
    "You can also leave a review with stars and suggestions. Enjoy!"
)

COMMANDS = {
    "grok": "https://grok.com",
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "instagram": "https://www.instagram.com",
    "chatgpt": "https://chatgpt.com",
    "whatsapp": "https://web.whatsapp.com",
    "facebook": "https://www.facebook.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "maps": "https://maps.google.com",
    "wikipedia": "https://wikipedia.org",
    "lova": "https://lova-gtf7.onrender.com",
}

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
            json.dump(reviews, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def submit_review(stars, comment, name):
    if not stars:
        return "Please select stars from 1 to 5.", format_reviews()
    reviews = load_reviews()
    reviews.insert(0, {
        "stars": int(stars),
        "comment": (comment or "").strip() or "No comment",
        "name": (name or "").strip() or "Anonymous",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    save_reviews(reviews[:50])
    return "✅ Thank you for your review!", format_reviews()

def format_reviews():
    reviews = load_reviews()
    if not reviews:
        return "<span style='color:#888'>No reviews yet. Be the first to rate Lova.</span>"
    html = ""
    for r in reviews[:12]:
        stars = "★" * r["stars"] + "☆" * (5 - r["stars"])
        html += f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
        border-radius:14px;padding:14px 16px;margin-bottom:10px;">
            <div style="color:#ffd35a;font-size:1.05rem;">{stars}</div>
            <div style="color:#fff;margin-top:6px;">{r['comment']}</div>
            <div style="color:#777;font-size:0.8rem;margin-top:6px;">{r['name']} · {r['time']}</div>
        </div>
        """
    return html

def find_open_command(text):
    if not text:
        return None, None
    t = text.lower().strip()
    for name, url in COMMANDS.items():
        for p in [f"open {name}", f"open the {name}", f"{name} kholo", f"kholo {name}",
                  f"launch {name}", f"start {name}"]:
            if p in t:
                return name, url
    return None, None

def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No information found."
        return "\n\n".join([f"{i}. {r['title']}\n{r['body']}" for i, r in enumerate(results, 1)])
    except Exception as e:
        return f"Search error: {str(e)}"

def transcribe_audio(audio):
    if audio is None:
        return ""
    try:
        sample_rate, audio_data = audio
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)
        m = np.max(np.abs(audio_data))
        if m > 0:
            audio_data = audio_data / m
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, sample_rate, (audio_data * 32767).astype(np.int16))
            path = f.name
        with open(path, "rb") as file:
            tr = client.audio.transcriptions.create(
                file=(path, file.read()), model="whisper-large-v3", language=None
            )
        os.unlink(path)
        return tr.text.strip()
    except Exception as e:
        return f"Transcription error: {str(e)}"

def is_mostly_hindi(text):
    hindi = set("अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")
    return sum(1 for c in text if c in hindi) >= 3

def format_history():
    if not chat_history:
        return "No messages yet. Try saying **open Grok** or record your voice."
    out = ""
    for m in chat_history[-12:]:
        if m["role"] == "user":
            out += f"**You:** {m['content']}\n\n"
        else:
            out += f"**Lova:** {m['content']}\n\n"
    return out

def ask_llm(user_text):
    global chat_history
    if not user_text or not user_text.strip():
        return "Please say or type something.", format_history(), None
    if user_text.startswith("Transcription error"):
        return user_text, format_history(), None

    name, url = find_open_command(user_text)
    if name and url:
        reply = f"Opening {name.capitalize()} for you."
        chat_history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
        html = f'<script>window.open("{url}", "_blank");</script><p style="color:#7dd3fc">Opening <a href="{url}" target="_blank" style="color:#38bdf8">{name}</a>...</p>'
        return reply, format_history(), html

    keys = ["who","what","when","where","stock","price","news","weather","current","latest","today","kya","kaun","kab","kahan"]
    extra = search_web(user_text) if any(k in user_text.lower() for k in keys) else ""
    messages = [{
        "role": "system",
        "content": "You are Lova, a stylish helpful AI created by Saksham Jha. Reply in the user's language. Keep replies short and natural."
    }] + chat_history[-6:] + [{
        "role": "user",
        "content": f"User: {user_text}\n\nInfo:\n{extra}\n\nAnswer:" if extra else user_text
    }]
    try:
        r = client.chat.completions.create(messages=messages, model="openai/gpt-oss-20b")
        reply = r.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Sorry, error: {str(e)}"
    chat_history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
    return reply, format_history(), None

async def tts(text, voice=None):
    try:
        if voice is None:
            voice = "hi-IN-SwaraNeural" if is_mostly_hindi(text) else "en-US-AriaNeural"
        c = edge_tts.Communicate(text, voice=voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            await c.save(f.name)
            return f.name
    except Exception:
        return None

def start_app():
    audio = asyncio.run(tts(WELCOME_TEXT, voice="en-US-AriaNeural"))
    return gr.update(visible=False), gr.update(visible=True), audio, format_reviews()

def process_voice(audio):
    user_text = transcribe_audio(audio)
    if not user_text:
        return "", "I couldn't hear you.", format_history(), None, ""
    reply, hist, html = ask_llm(user_text)
    return user_text, reply, hist, asyncio.run(tts(reply)), html or ""

def process_text(user_text):
    reply, hist, html = ask_llm(user_text)
    return reply, hist, asyncio.run(tts(reply)), html or ""

def clear_chat():
    global chat_history
    chat_history = []
    return "", "", "Chat cleared.", None, ""

# ========== STYLISH CSS ==========
css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

.gradio-container {
    max-width: 100% !important;
    min-height: 100vh !important;
    padding: 0 !important;
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(56,189,248,0.18), transparent 50%),
        radial-gradient(900px 500px at 100% 0%, rgba(167,139,250,0.15), transparent 45%),
        linear-gradient(180deg, #070b14 0%, #0b1220 100%) !important;
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    color: #e8eef8 !important;
}
footer {display:none !important;}

.main-wrap {padding: 28px 20px 40px; max-width: 1100px; margin: 0 auto;}

.glass {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
}

button.primary, .start-btn button {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    box-shadow: 0 8px 24px rgba(56,189,248,0.25) !important;
}
"""

with gr.Blocks(title="Lova AI", css=css, theme=gr.themes.Base()) as demo:

    # WELCOME
    with gr.Column(visible=True, elem_classes=["main-wrap"]) as welcome_panel:
        gr.HTML("""
        <div style="min-height:75vh;display:flex;align-items:center;justify-content:center;">
          <div style="width:min(440px,92vw);text-align:center;padding:40px 28px;
            background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
            border-radius:28px;box-shadow:0 20px 60px rgba(0,0,0,0.45);
            backdrop-filter:blur(16px);">
            <div style="width:92px;height:92px;margin:0 auto 18px;border-radius:28px;
              background:linear-gradient(135deg,#38bdf8,#a78bfa);display:flex;align-items:center;
              justify-content:center;font-size:48px;box-shadow:0 12px 40px rgba(56,189,248,0.35);
              animation:floaty 3s ease-in-out infinite;">✨</div>
            <h1 style="margin:0;font-size:2.6rem;letter-spacing:4px;
              background:linear-gradient(90deg,#fff,#93c5fd);-webkit-background-clip:text;
              -webkit-text-fill-color:transparent;">LOVA</h1>
            <p style="color:#9fb0c7;margin:10px 0 6px;font-size:1.05rem;">Smart Voice AI Assistant</p>
            <p style="color:#6b7c93;margin:0 0 22px;font-size:0.92rem;">Created by Saksham Jha</p>
            <p style="color:#8494aa;font-size:0.9rem;line-height:1.55;margin:0 0 8px;">
              Tap start to meet Lova. She’ll greet you and show how to use voice, chat, and commands.
            </p>
          </div>
        </div>
        <style>
        @keyframes floaty {0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
        </style>
        """)
        with gr.Row(elem_classes=["start-btn"]):
            start_btn = gr.Button("Start Lova →", variant="primary", size="lg")

    welcome_audio = gr.Audio(visible=False, autoplay=True)

    # MAIN APP
    with gr.Column(visible=False, elem_classes=["main-wrap"]) as main_panel:
        gr.HTML("""
        <div style="display:flex;align-items:center;justify-content:space-between;
          margin-bottom:18px;padding:16px 18px;border-radius:18px;
          background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:42px;height:42px;border-radius:14px;
              background:linear-gradient(135deg,#38bdf8,#a78bfa);display:flex;
              align-items:center;justify-content:center;">✨</div>
            <div>
              <div style="font-weight:700;font-size:1.25rem;letter-spacing:2px;">LOVA</div>
              <div style="color:#7f8ea3;font-size:0.8rem;">Voice · Chat · Commands</div>
            </div>
          </div>
          <div style="color:#6b7c93;font-size:0.8rem;">by Saksham Jha</div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Tab("🎤 Voice"):
                    gr.Markdown("Record your voice, then press **Send**")
                    audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Microphone", format="wav")
                    user_said = gr.Textbox(label="You said", interactive=False)
                    voice_send = gr.Button("Send", variant="primary")
                with gr.Tab("💬 Chat"):
                    text_input = gr.Textbox(label="Message or command", placeholder="Try: open Grok", lines=3)
                    text_send = gr.Button("Send", variant="primary")
                with gr.Tab("⭐ Reviews"):
                    gr.Markdown("Rate Lova and leave a compliment or suggestion")
                    with gr.Row():
                        rev_name = gr.Textbox(label="Name", placeholder="Optional")
                        rev_stars = gr.Radio(["1","2","3","4","5"], label="Stars", value="5")
                    rev_comment = gr.Textbox(label="Your review", lines=3, placeholder="What should improve?")
                    rev_btn = gr.Button("Submit review", variant="primary")
                    rev_status = gr.Markdown()
                    rev_list = gr.HTML(format_reviews())

            with gr.Column(scale=5):
                history_box = gr.Markdown("Conversation will appear here.")
                reply_box = gr.Textbox(label="Lova's reply", lines=4)
                audio_output = gr.Audio(label="Lova's voice", autoplay=True)
                open_box = gr.HTML()
                clear_btn = gr.Button("Clear chat")

    start_btn.click(fn=start_app, outputs=[welcome_panel, main_panel, welcome_audio, rev_list])
    voice_send.click(fn=process_voice, inputs=audio_input, outputs=[user_said, reply_box, history_box, audio_output, open_box])
    text_send.click(fn=process_text, inputs=text_input, outputs=[reply_box, history_box, audio_output, open_box])
    text_input.submit(fn=process_text, inputs=text_input, outputs=[reply_box, history_box, audio_output, open_box])
    clear_btn.click(fn=clear_chat, outputs=[user_said, reply_box, history_box, audio_output, open_box])
    rev_btn.click(fn=submit_review, inputs=[rev_stars, rev_comment, rev_name], outputs=[rev_status, rev_list])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
