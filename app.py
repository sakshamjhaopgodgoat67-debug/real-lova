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

# ======================
# CONFIG
# ======================
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

# ======================
# REVIEWS
# ======================

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
        return "Please select a star rating (1 to 5).", format_reviews()
    reviews = load_reviews()
    reviews.insert(0, {
        "stars": int(stars),
        "comment": (comment or "").strip() or "No comment",
        "name": (name or "").strip() or "Anonymous",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    reviews = reviews[:50]
    save_reviews(reviews)
    return "Thank you for your review!", format_reviews()


def format_reviews():
    reviews = load_reviews()
    if not reviews:
        return "_No reviews yet. Be the first!_"
    lines = []
    for r in reviews[:15]:
        stars = "★" * r["stars"] + "☆" * (5 - r["stars"])
        lines.append(f"**{stars}** — **{r['name']}** ({r['time']})\n{r['comment']}\n")
    return "\n".join(lines)


# ======================
# CORE AI
# ======================

def find_open_command(text):
    if not text:
        return None, None
    t = text.lower().strip()
    for name, url in COMMANDS.items():
        for p in [f"open {name}", f"open the {name}", f"{name} kholo", f"kholo {name}",
                  f"{name} open karo", f"launch {name}", f"start {name}"]:
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
                file=(path, file.read()),
                model="whisper-large-v3",
                language=None
            )
        os.unlink(path)
        return tr.text.strip()
    except Exception as e:
        return f"Transcription error: {str(e)}"


def is_mostly_hindi(text):
    hindi = set("अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञड़ढ़")
    return sum(1 for c in text if c in hindi) >= 3


def ask_llm(user_text):
    global chat_history
    if not user_text or not user_text.strip():
        return "Please say or type something.", format_history(), None
    if user_text.startswith("Transcription error"):
        return user_text, format_history(), None

    name, url = find_open_command(user_text)
    if name and url:
        reply = f"Opening {name.capitalize()} for you."
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": reply})
        html = f'<script>window.open("{url}", "_blank");</script><p>Opening <a href="{url}" target="_blank">{name}</a>...</p>'
        return reply, format_history(), html

    keywords = ["who", "what", "when", "where", "stock", "price", "news", "weather",
                "current", "latest", "today", "kya", "kaun", "kab", "kahan", "kaise", "kyun"]
    extra = search_web(user_text) if any(k in user_text.lower() for k in keywords) else ""

    messages = [{
        "role": "system",
        "content": "You are Lova, a helpful AI created by Saksham Jha. Reply in the user's language (English or Hindi). Keep replies short."
    }]
    messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": f"User: {user_text}\n\nInfo:\n{extra}\n\nAnswer:" if extra else user_text})

    try:
        r = client.chat.completions.create(messages=messages, model="openai/gpt-oss-20b")
        reply = r.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Sorry, error: {str(e)}"

    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": reply})
    return reply, format_history(), None


def format_history():
    if not chat_history:
        return "No messages yet. Try: **open Grok** or record your voice."
    return "\n\n".join([
        f"**{'You' if m['role']=='user' else 'Lova'}:** {m['content']}"
        for m in chat_history
    ])


async def text_to_speech(text, voice=None):
    try:
        if voice is None:
            voice = "hi-IN-SwaraNeural" if is_mostly_hindi(text) else "en-US-AriaNeural"
        c = edge_tts.Communicate(text, voice=voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            await c.save(f.name)
            return f.name
    except Exception:
        return None


def make_welcome_audio():
    return asyncio.run(text_to_speech(WELCOME_TEXT, voice="en-US-AriaNeural"))


def start_app():
    """After welcome: show main UI + play welcome audio"""
    audio = make_welcome_audio()
    return (
        gr.update(visible=False),  # welcome panel hide
        gr.update(visible=True),   # main panel show
        audio,
        format_reviews(),
    )


def process_voice(audio):
    user_text = transcribe_audio(audio)
    if not user_text:
        return "", "I couldn't hear you.", format_history(), None, ""
    reply, hist, html = ask_llm(user_text)
    voice = asyncio.run(text_to_speech(reply))
    return user_text, reply, hist, voice, html or ""


def process_text(user_text):
    reply, hist, html = ask_llm(user_text)
    voice = asyncio.run(text_to_speech(reply))
    return reply, hist, voice, html or ""


def clear_chat():
    global chat_history
    chat_history = []
    return "", "", "Chat cleared.", None, ""


# ======================
# UI - Black & White + Welcome
# ======================

css = """
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: #0a0a0a !important;
    color: #f5f5f5 !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
footer { display: none !important; }
.bw-card {
    background: #111 !important;
    border: 1px solid #333 !important;
    border-radius: 16px !important;
    padding: 20px !important;
}
button.primary {
    background: #fff !important;
    color: #000 !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
}
.welcome-face {
    font-size: 72px;
    line-height: 1.2;
    text-align: center;
    animation: pop 0.6s ease;
}
@keyframes pop {
    0% { transform: scale(0.3); opacity: 0; }
    70% { transform: scale(1.1); }
    100% { transform: scale(1); opacity: 1; }
}
"""

with gr.Blocks(title="Lova AI", css=css, theme=gr.themes.Base()) as demo:

    # ---------- WELCOME SCREEN ----------
    with gr.Column(visible=True) as welcome_panel:
        gr.HTML(
            """
            <div style="min-height:70vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:40px;background:#0a0a0a;color:#fff;">
                <div class="welcome-face">😊</div>
                <h1 style="margin:16px 0 8px;font-size:2.4rem;letter-spacing:2px;">LOVA</h1>
                <p style="color:#aaa;max-width:420px;margin-bottom:24px;">
                    Smart Voice AI Assistant by <b>Saksham Jha</b>
                </p>
                <p style="color:#888;max-width:480px;font-size:0.95rem;">
                    Tap below to start. Lova will greet you and explain how to use the app.
                </p>
            </div>
            """
        )
        start_btn = gr.Button("Start Lova", variant="primary", size="lg")

    welcome_audio = gr.Audio(visible=False, autoplay=True)

    # ---------- MAIN APP ----------
    with gr.Column(visible=False) as main_panel:
        gr.HTML(
            """
            <div style="padding:20px 20px 8px;border-bottom:1px solid #222;background:#0a0a0a;">
                <h2 style="margin:0;color:#fff;letter-spacing:3px;">LOVA</h2>
                <p style="margin:4px 0 0;color:#888;font-size:0.9rem;">Voice · Chat · Commands · by Saksham Jha</p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("🎤 Voice"):
                gr.Markdown("**Record** → then press **Send**")
                audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record", format="wav")
                user_said = gr.Textbox(label="You said", interactive=False)
                voice_send = gr.Button("Send", variant="primary")

            with gr.Tab("💬 Text"):
                text_input = gr.Textbox(label="Message or command", placeholder="e.g. open Grok / hello", lines=2)
                text_send = gr.Button("Send", variant="primary")

            with gr.Tab("⭐ Review"):
                gr.Markdown("Rate Lova and leave a suggestion")
                with gr.Row():
                    rev_name = gr.Textbox(label="Your name", placeholder="Optional")
                    rev_stars = gr.Radio(choices=["1", "2", "3", "4", "5"], label="Stars", value="5")
                rev_comment = gr.Textbox(label="Compliment or suggestion", lines=3, placeholder="What did you like? What can improve?")
                rev_btn = gr.Button("Submit review", variant="primary")
                rev_status = gr.Markdown("")
                rev_list = gr.Markdown(format_reviews())

        history_box = gr.Markdown("No messages yet.")
        reply_box = gr.Textbox(label="Lova's Reply", lines=3)
        audio_output = gr.Audio(label="Lova's Voice", autoplay=True)
        open_box = gr.HTML()
        clear_btn = gr.Button("Clear chat")

    # ---------- EVENTS ----------
    start_btn.click(
        fn=start_app,
        outputs=[welcome_panel, main_panel, welcome_audio, rev_list]
    )

    voice_send.click(
        fn=process_voice,
        inputs=audio_input,
        outputs=[user_said, reply_box, history_box, audio_output, open_box]
    )
    text_send.click(
        fn=process_text,
        inputs=text_input,
        outputs=[reply_box, history_box, audio_output, open_box]
    )
    text_input.submit(
        fn=process_text,
        inputs=text_input,
        outputs=[reply_box, history_box, audio_output, open_box]
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[user_said, reply_box, history_box, audio_output, open_box]
    )
    rev_btn.click(
        fn=submit_review,
        inputs=[rev_stars, rev_comment, rev_name],
        outputs=[rev_status, rev_list]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
