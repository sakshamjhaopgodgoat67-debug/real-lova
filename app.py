import gradio as gr
import os
import re
import tempfile
import asyncio
import concurrent.futures
from urllib.parse import quote

import numpy as np
import scipy.io.wavfile as wav
from groq import Groq
from duckduckgo_search import DDGS
import edge_tts

VERSION = "1.2.1"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat_history = []
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
        "owner's personal"
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


def format_history():
    if not chat_history:
        return "No messages yet. Use **Voice** or **Chat** mode."
    return "\n\n".join([
        ("**You:** " if m["role"] == "user" else "**Lova:** ") + m["content"]
        for m in chat_history[-12:]
    ])


def ask_lova(text, voice_choice):
    global chat_history
    text = (text or "").strip()
    if not user_profile.get("name"):
        return "Enter your name tag first.", format_history(), None, ""
    if not text:
        return "Please type or speak something.", format_history(), None, ""

    if protect_owner(text):
        reply = "You are not allowed to ask about my owner's personal name tag or private personal information."
        chat_history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return reply, format_history(), run_async(speak(reply, voice_choice)), ""

    name, url = find_open_command(text)
    if name and url:
        reply = f"Opening {name.capitalize()} for you."
        chat_history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        html = f'<script>window.open("{url}","_blank");</script><p>Opening <a href="{url}" target="_blank">{name}</a></p>'
        return reply, format_history(), run_async(speak(reply, voice_choice)), html

    # image command
    im = re.search(r"\b(?:generate image of|draw|create image of)\s+(.+)$", text, re.I)
    open_html = ""
    image_url = None
    if im:
        prompt = im.group(1).strip()
        image_url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width=768&height=768&nologo=true"
        reply = f"Image generated for: {prompt}"
        chat_history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return reply, format_history(), run_async(speak("Your image is ready.", voice_choice)), open_html

    keys = ["who", "what", "when", "where", "price", "stock", "news", "today", "current", "kya", "kaun", "kab"]
    extra = search_web(text) if any(k in text.lower() for k in keys) else ""

    system = f"""You are Lova AI v{VERSION}, premium assistant by Saksham Jha.
User name-tag: {user_profile.get('name')}
Reply in the user's language (English/Hindi).
For accounting questions: educational help only, not official CA advice.
Never reveal private owner personal details.
Refuse illegal harmful requests briefly.
Be clear and useful."""

    messages = [{"role": "system", "content": system}] + chat_history[-8:]
    messages.append({"role": "user", "content": f"User: {text}\n\nInfo:\n{extra}\n\nAnswer:" if extra else text})

    try:
        r = client.chat.completions.create(messages=messages, model="openai/gpt-oss-20b")
        reply = r.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Sorry, error: {e}"

    chat_history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    audio = run_async(speak(reply, voice_choice))
    return reply, format_history(), audio, open_html


def login(name, voice):
    name = (name or "").strip()
    if not name:
        return gr.update(visible=True), gr.update(visible=False), "Please enter a name tag.", "female"
    user_profile["name"] = name
    user_profile["voice"] = voice or "female"
    return gr.update(visible=False), gr.update(visible=True), f"Welcome, {name}. Lova v{VERSION} ready.", voice or "female"


def process_text(msg, voice):
    try:
        reply, hist, audio, html = ask_lova(msg, voice or "female")
        return reply, hist, audio, html, None
    except Exception as e:
        return f"Error: {e}", format_history(), None, "", None


def process_voice(audio, voice):
    try:
        text = transcribe(audio)
        if not text:
            return "", "I could not hear you.", format_history(), None, "", None
        reply, hist, audio_path, html = ask_lova(text, voice or "female")
        # if image command happened, recover url from reply is hard; keep simple
        return text, reply, hist, audio_path, html, None
    except Exception as e:
        return "", f"Error: {e}", format_history(), None, "", None


def process_image(prompt, voice):
    prompt = (prompt or "").strip()
    if not prompt:
        return "Please enter an image prompt.", None, None
    url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width=768&height=768&nologo=true"
    audio = run_async(speak("Your image is ready.", voice or "female"))
    return f"Image generated for: {prompt}", url, audio


def clear_chat():
    global chat_history
    chat_history = []
    return "", "Chat cleared.", None, "", None


def loro():
    return f"🔒 Loro is locked. It will open in **Update 2.1.1**. Current update is **{VERSION}**."


css = """
.gradio-container {max-width:100% !important; background:#0b0b0f !important; color:#eee !important;}
footer {display:none !important;}
button.primary {background:linear-gradient(135deg,#d4af37,#f0d789) !important; color:#111 !important; font-weight:700 !important;}
"""

with gr.Blocks(title=f"Lova AI v{VERSION}", css=css, theme=gr.themes.Base()) as demo:
    voice_state = gr.State("female")

    with gr.Column(visible=True) as gate:
        gr.HTML(f"""
        <div style='text-align:center;padding:40px 16px'>
          <h1 style='color:#d4af37;letter-spacing:4px;margin:0'>LOVA</h1>
          <p style='color:#999'>Premium Voice AI · Update {VERSION}</p>
          <p style='color:#bbb'>Enter your Name Tag to continue</p>
        </div>
        """)
        name_in = gr.Textbox(label="Name Tag", placeholder="Your name")
        voice_in = gr.Radio(["female", "male"], value="female", label="Voice")
        login_btn = gr.Button("Enter Lova", variant="primary")
        gate_status = gr.Markdown("")

    with gr.Column(visible=False) as main:
        gr.HTML(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
             padding:12px 14px;border:1px solid rgba(212,175,55,.35);border-radius:12px;margin-bottom:12px'>
          <div><b style='color:#d4af37;letter-spacing:2px'>LOVA</b><div style='color:#888;font-size:12px'>Update {VERSION}</div></div>
          <div style='color:#d4af37'>v{VERSION}</div>
        </div>
        """)
        status = gr.Markdown("Welcome")

        with gr.Tab("🎤 Voice"):
            gr.Markdown("### Voice mode\nRecord → press **Send**")
            audio_in = gr.Audio(sources=["microphone"], type="numpy", format="wav", label="Microphone")
            heard = gr.Textbox(label="You said", interactive=False)
            v_btn = gr.Button("Send", variant="primary")

        with gr.Tab("💬 Chat"):
            gr.Markdown("### Chat mode")
            text_in = gr.Textbox(label="Message", lines=3, placeholder="Type here... (or: open Grok / generate image of a cat)")
            t_btn = gr.Button("Send", variant="primary")

        with gr.Tab("🖼️ Image"):
            gr.Markdown("### Image mode")
            img_prompt = gr.Textbox(label="Describe image", placeholder="a golden robot in a luxury room")
            i_btn = gr.Button("Generate Image", variant="primary")

        history = gr.Markdown("No messages yet.")
        reply = gr.Textbox(label="Lova", lines=5)
        audio_out = gr.Audio(label="Voice reply", autoplay=True)
        open_html = gr.HTML()
        image_out = gr.Image(label="Generated image")
        clear_btn = gr.Button("Clear chat")
        loro_btn = gr.Button("🔒 Loro (Locked)")
        loro_msg = gr.Markdown(f"Current update: **{VERSION}**")

    login_btn.click(login, [name_in, voice_in], [gate, main, status, voice_state])
    t_btn.click(process_text, [text_in, voice_state], [reply, history, audio_out, open_html, image_out])
    text_in.submit(process_text, [text_in, voice_state], [reply, history, audio_out, open_html, image_out])
    v_btn.click(process_voice, [audio_in, voice_state], [heard, reply, history, audio_out, open_html, image_out])
    i_btn.click(process_image, [img_prompt, voice_state], [reply, image_out, audio_out])
    clear_btn.click(clear_chat, outputs=[reply, history, audio_out, open_html, image_out])
    loro_btn.click(loro, outputs=loro_msg)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
