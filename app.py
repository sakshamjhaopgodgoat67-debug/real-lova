import gradio as gr
import os
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import asyncio
from groq import Groq
from duckduckgo_search import DDGS
import edge_tts

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
chat_history = []

def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No information found."
        summary = ""
        for i, r in enumerate(results, 1):
            summary += f"{i}. {r['title']}\n{r['body']}\n\n"
        return summary.strip()
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
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, sample_rate, (audio_data * 32767).astype(np.int16))
            temp_filename = f.name
        with open(temp_filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, file.read()),
                model="whisper-large-v3",
                language=None
            )
        os.unlink(temp_filename)
        return transcription.text.strip()
    except Exception as e:
        return f"Transcription error: {str(e)}"

def is_mostly_hindi(text):
    hindi_chars = set("अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञड़ढ़")
    return sum(1 for c in text if c in hindi_chars) >= 3

def ask_llm(user_text):
    global chat_history
    if not user_text or not user_text.strip():
        return "Please say or type something.", format_history()
    if user_text.startswith("Transcription error"):
        return user_text, format_history()

    search_keywords = ["who", "what", "when", "where", "stock", "price", "news", "weather", "current", "latest", "today", "kya", "kaun", "kab", "kahan", "kaise", "kyun"]
    need_search = any(word in user_text.lower() for word in search_keywords)
    extra_info = search_web(user_text) if need_search else ""

    messages = [{
        "role": "system",
        "content": "You are Lova, a helpful friendly AI created by Saksham Jha. You understand English and Hindi. Reply in the same language the user used. Keep replies short and natural."
    }]
    for msg in chat_history[-6:]:
        messages.append(msg)

    prompt = f"User: {user_text}\n\nReal-world info:\n{extra_info}\n\nAnswer helpfully:" if extra_info else user_text
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(messages=messages, model="openai/gpt-oss-20b")
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Sorry, error: {str(e)}"

    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": reply})
    return reply, format_history()

def format_history():
    if not chat_history:
        return "No messages yet. Start by recording or typing."
    text = ""
    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "Lova"
        text += f"**{role}:** {msg['content']}\n\n"
    return text

async def text_to_speech(text):
    try:
        voice = "hi-IN-SwaraNeural" if is_mostly_hindi(text) else "en-US-AriaNeural"
        communicate = edge_tts.Communicate(text, voice=voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            await communicate.save(f.name)
            return f.name
    except Exception:
        return None

def process_voice(audio):
    user_text = transcribe_audio(audio)
    if not user_text:
        return "", "I couldn't hear you. Please record again.", format_history(), None
    reply, history_text = ask_llm(user_text)
    audio_path = asyncio.run(text_to_speech(reply))
    return user_text, reply, history_text, audio_path

def process_text(user_text):
    reply, history_text = ask_llm(user_text)
    audio_path = asyncio.run(text_to_speech(reply))
    return reply, history_text, audio_path

def clear_chat():
    global chat_history
    chat_history = []
    return "", "", "Chat cleared.", None

css = """
.gradio-container { max-width: 100% !important; padding: 16px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Lova AI - Smart Voice Assistant", css=css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎙️ Lova AI\n### Smart Voice & Chat Assistant · Created by Saksham Jha\nSpeak or type in **English** or **Hindi**")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab("🎤 Voice"):
                gr.Markdown("**Step 1:** Record  →  **Step 2:** Click Send")
                audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your voice", format="wav")
                user_said = gr.Textbox(label="You said", interactive=False)
                voice_send_btn = gr.Button("Send", variant="primary", size="lg")
            with gr.Tab("💬 Text"):
                text_input = gr.Textbox(label="Type your message", placeholder="Type in English or Hindi...", lines=3)
                text_send_btn = gr.Button("Send", variant="primary", size="lg")

        with gr.Column(scale=1):
            history_box = gr.Markdown("No messages yet.")
            reply_box = gr.Textbox(label="Lova's Reply", lines=4)
            audio_output = gr.Audio(label="Lova's Voice", autoplay=True)
            clear_btn = gr.Button("Clear Chat", variant="secondary")

    voice_send_btn.click(fn=process_voice, inputs=audio_input, outputs=[user_said, reply_box, history_box, audio_output])
    text_send_btn.click(fn=process_text, inputs=text_input, outputs=[reply_box, history_box, audio_output])
    text_input.submit(fn=process_text, inputs=text_input, outputs=[reply_box, history_box, audio_output])
    clear_btn.click(fn=clear_chat, outputs=[user_said, reply_box, history_box, audio_output])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
