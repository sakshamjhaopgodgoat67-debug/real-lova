import gradio as gr
import os
import re
import tempfile
import asyncio
from urllib.parse import quote

import numpy as np
import scipy.io.wavfile as wav

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None

from groq import Groq
from duckduckgo_search import DDGS
import edge_tts

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

# ======================
# CONFIG
# ======================
VERSION = "1.2.1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

chat_history = []
user_profile = {"name": None, "voice": "female"}

VOICES = {
    "female": "en-US-AriaNeural",
    "male": "en-US-GuyNeural",
    "female_hindi": "hi-IN-SwaraNeural",
    "male_hindi": "hi-IN-MadhurNeural",
}

# ======================
# HELPERS
# ======================

def is_hindi(text: str) -> bool:
    return sum(1 for c in text if "\u0900" <= c <= "\u097F") >= 3


def protect_owner_questions(text: str) -> bool:
    t = (text or "").lower()
    triggers = [
        "owner name", "your owner", "who is your owner", "creator name",
        "saksham", "real name of owner", "where does your owner live",
        "owner address", "owner phone", "owner school", "personal information of owner",
        "owner's personal", "tell me about your owner personally"
    ]
    return any(x in t for x in triggers)


def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No information found."
        return "\n\n".join(
            [f"{i}. {r.get('title', '')}\n{r.get('body', '')}" for i, r in enumerate(results, 1)]
        )
    except Exception as e:
        return f"Search error: {e}"


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
                language=None,
            )
        os.unlink(path)
        return (tr.text or "").strip()
    except Exception as e:
        return f"Transcription error: {e}"


async def speak(text: str, voice_choice: str = "female"):
    try:
        if is_hindi(text):
            voice = VOICES["female_hindi"] if voice_choice == "female" else VOICES["male_hindi"]
        else:
            voice = VOICES["female"] if voice_choice == "female" else VOICES["male"]
        communicate = edge_tts.Communicate(text, voice=voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            await communicate.save(f.name)
            return f.name
    except Exception:
        return None


def make_stock_graph(symbol: str):
    symbol = (symbol or "").upper().strip()
    if not symbol:
        return None, "Please provide a symbol like AAPL, TSLA, RELIANCE.NS, USDINR=X"
    if not HAS_YF:
        return None, "Stock package not available."
    if not HAS_MPL:
        return None, "Graph package not available."

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo", interval="1d")
        if df is None or df.empty:
            return None, f"No data found for {symbol}."

        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
        ax.plot(df.index, df["Close"], color="#d4af37", linewidth=2)
        ax.set_title(f"{symbol} — Last 1 Month", color="white", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.set_facecolor("#0f0f12")
        fig.patch.set_facecolor("#0b0b0e")
        ax.tick_params(colors="#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.grid(True, alpha=0.2, color="#666666")

        path = tempfile.mktemp(suffix=".png")
        fig.tight_layout()
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path, f"Graph ready for {symbol}."
    except Exception as e:
        return None, f"Graph error: {e}"


def generate_image(prompt: str):
    if not prompt or not prompt.strip():
        return None, "Please describe the image."
    url = f"https://image.pollinations.ai/prompt/{quote(prompt.strip())}?width=768&height=768&nologo=true"
    return url, "Image generated."


def format_history():
    if not chat_history:
        return "No messages yet."
    lines = []
    for m in chat_history[-16:]:
        who = "You" if m["role"] == "user" else "Lova"
        lines.append(f"**{who}:** {m['content']}")
    return "\n\n".join(lines)


def ask_lova(user_text: str, voice_choice: str):
    global chat_history

    if not user_profile.get("name"):
        return "Please enter your name tag first.", format_history(), None, None

    if not user_text or not str(user_text).strip():
        return "Please say or type something.", format_history(), None, None

    text = str(user_text).strip()

    if protect_owner_questions(text):
        reply = "You are not allowed to ask about my owner's personal name tag or private personal information."
        chat_history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        audio = asyncio.run(speak(reply, voice_choice))
        return reply, format_history(), audio, None

    g = re.search(r"\b(?:graph|chart)\s+([A-Za-z0-9\.\=\-\^]+)", text, re.I)
    if g:
        img, msg = make_stock_graph(g.group(1))
        reply = msg
        chat_history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        audio = asyncio.run(speak(reply, voice_choice))
        return reply, format_history(), audio, img

    im = re.search(r"\b(?:generate image of|draw|create image of)\s+(.+)$", text, re.I)
    if im:
        url, msg = generate_image(im.group(1))
        reply = msg + (f" Prompt: {im.group(1)}" if url else "")
        chat_history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        audio = asyncio.run(speak("Your image is ready.", voice_choice))
        return reply, format_history(), audio, url

    keys = ["who", "what", "when", "where", "price", "stock", "forex", "news", "today", "current", "kya", "kaun", "kab"]
    extra = search_web(text) if any(k in text.lower() for k in keys) else ""

    system = f"""
You are Lova AI v{VERSION}, a premium smart assistant created by Saksham Jha.
User name-tag: {user_profile.get('name')}

Rules:
- Be helpful, clear, and high-quality.
- Support English and Hindi; reply in the user's language.
- For accounting/CA-style questions: help with calculations, explanations, and learning.
  Always add: "This is educational help, not official CA/legal advice."
- For stocks/forex: give general information and caution that markets are risky.
- Never reveal private personal details about the owner.
- If asked for illegal harmful wrongdoing, refuse briefly and suggest safe alternatives.
- Keep answers practical and well structured.
"""

    messages = [{"role": "system", "content": system}]
    messages.extend(chat_history[-8:])
    prompt = f"User: {text}\n\nWeb info:\n{extra}\n\nAnswer:" if extra else text
    messages.append({"role": "user", "content": prompt})

    try:
        r = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-20b",
        )
        reply = r.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Sorry, error: {e}"

    chat_history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    audio = asyncio.run(speak(reply, voice_choice))
    return reply, format_history(), audio, None


# ======================
# UI ACTIONS
# ======================

def save_name(name, voice):
    name = (name or "").strip()
    if not name:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "Please enter a name tag.",
            voice or "female",
        )
    user_profile["name"] = name
    user_profile["voice"] = voice or "female"
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        f"Welcome, {name}. Lova v{VERSION} is ready.",
        voice or "female",
    )


def process_text(message, voice):
    reply, hist, audio, media = ask_lova(message, voice or user_profile.get("voice", "female"))
    return reply, hist, audio, media


def process_voice(audio, voice):
    text = transcribe_audio(audio)
    if not text:
        return "", "I could not hear you.", format_history(), None, None
    reply, hist, audio_path, media = ask_lova(text, voice or user_profile.get("voice", "female"))
    return text, reply, hist, audio_path, media


def clear_chat():
    global chat_history
    chat_history = []
    return "", "Chat cleared.", None, None


def loro_locked():
    return f"🔒 Loro is locked.\nIt will open in Update 2.1.1.\nCurrent update is {VERSION}."


def graph_only(symbol, voice):
    img, msg = make_stock_graph(symbol)
    audio = asyncio.run(speak(msg, voice or "female"))
    return msg, img, audio


def image_only(prompt, voice):
    url, msg = generate_image(prompt)
    audio = asyncio.run(speak("Image generated.", voice or "female"))
    return msg, url, audio


# ======================
# PREMIUM UI
# ======================

css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');
.gradio-container {
  max-width: 100% !important;
  min-height: 100vh !important;
  padding: 0 !important;
  background:
    radial-gradient(1000px 500px at 0% 0%, rgba(212,175,55,0.12), transparent 50%),
    radial-gradient(900px 500px at 100% 0%, rgba(120,120,140,0.12), transparent 45%),
    linear-gradient(180deg, #070709 0%, #101016 100%) !important;
  font-family: Inter, system-ui, sans-serif !important;
  color: #f3f3f3 !important;
}
footer {display:none !important;}
.wrap {max-width: 1100px; margin: 0 auto; padding: 22px 16px 40px;}
button.primary {
  background: linear-gradient(135deg, #d4af37, #f0d789) !important;
  color: #111 !important; border: none !important; border-radius: 12px !important;
  font-weight: 700 !important;
}
"""

with gr.Blocks(title=f"Lova AI v{VERSION}", css=css, theme=gr.themes.Base()) as demo:
    voice_state = gr.State("female")

    # NAME TAG GATE
    with gr.Column(visible=True, elem_classes=["wrap"]) as gate:
        gr.HTML(f"""
        <div style="min-height:70vh;display:flex;align-items:center;justify-content:center;">
          <div style="width:min(460px,96vw);padding:34px 26px;border-radius:22px;
            background:rgba(255,255,255,0.04);border:1px solid rgba(212,175,55,0.35);
            box-shadow:0 20px 60px rgba(0,0,0,0.45);text-align:center;">
            <div style="font-family:Playfair Display,serif;font-size:2.4rem;letter-spacing:4px;
              color:#d4af37;margin-bottom:6px;">LOVA</div>
            <div style="color:#aaa;margin-bottom:18px;">Premium Voice AI · Update {VERSION}</div>
            <div style="color:#ccc;font-size:0.95rem;line-height:1.5;margin-bottom:8px;">
              Enter your <b>Name Tag</b> to continue.<br>
              Personal owner details stay protected.
            </div>
          </div>
        </div>
        """)
        name_in = gr.Textbox(label="Name Tag", placeholder="Enter your name")
        voice_in = gr.Radio(["female", "male"], value="female", label="Voice")
        gate_btn = gr.Button("Enter Lova", variant="primary")
        gate_msg = gr.Markdown("")

    # MAIN APP
    with gr.Column(visible=False, elem_classes=["wrap"]) as main:
        gr.HTML(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
          padding:14px 16px;border-radius:16px;margin-bottom:14px;
          background:rgba(255,255,255,0.03);border:1px solid rgba(212,175,55,0.25);">
          <div>
            <div style="font-family:Playfair Display,serif;font-size:1.6rem;color:#d4af37;letter-spacing:3px;">LOVA</div>
            <div style="color:#888;font-size:0.85rem;">Update {VERSION} · Premium Assistant</div>
          </div>
          <div style="color:#d4af37;font-weight:600;">v{VERSION}</div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Tab("🎤 Voice"):
                    audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Record", format="wav")
                    heard = gr.Textbox(label="You said", interactive=False)
                    v_send = gr.Button("Send", variant="primary")
                with gr.Tab("💬 Chat"):
                    text_in = gr.Textbox(
                        label="Message",
                        lines=3,
                        placeholder="Ask anything useful... (graph AAPL / generate image of ...)"
                    )
                    t_send = gr.Button("Send", variant="primary")
                with gr.Tab("📈 Market Graph"):
                    sym = gr.Textbox(label="Symbol", placeholder="AAPL, TSLA, RELIANCE.NS, USDINR=X")
                    g_btn = gr.Button("Generate Graph", variant="primary")
                with gr.Tab("🖼️ Image"):
                    img_prompt = gr.Textbox(label="Image prompt", placeholder="A luxury robot in a gold room")
                    i_btn = gr.Button("Generate Image", variant="primary")

            with gr.Column(scale=5):
                status = gr.Markdown("Welcome.")
                history = gr.Markdown("No messages yet.")
                reply = gr.Textbox(label="Lova", lines=6)
                audio_out = gr.Audio(label="Voice reply", autoplay=True)
                media_out = gr.Image(label="Image / Graph")
                clear = gr.Button("Clear chat")

        with gr.Row():
            loro_btn = gr.Button("🔒 Loro (Locked)", scale=1)
            loro_msg = gr.Markdown(f"Current update: **{VERSION}** · Loro opens at **2.1.1**")

    gate_btn.click(
        fn=save_name,
        inputs=[name_in, voice_in],
        outputs=[gate, main, status, voice_state],
    )

    t_send.click(fn=process_text, inputs=[text_in, voice_state], outputs=[reply, history, audio_out, media_out])
    text_in.submit(fn=process_text, inputs=[text_in, voice_state], outputs=[reply, history, audio_out, media_out])
    v_send.click(fn=process_voice, inputs=[audio_in, voice_state], outputs=[heard, reply, history, audio_out, media_out])
    clear.click(fn=clear_chat, outputs=[reply, history, audio_out, media_out])
    g_btn.click(fn=graph_only, inputs=[sym, voice_state], outputs=[reply, media_out, audio_out])
    i_btn.click(fn=image_only, inputs=[img_prompt, voice_state], outputs=[reply, media_out, audio_out])
    loro_btn.click(fn=loro_locked, outputs=loro_msg)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
