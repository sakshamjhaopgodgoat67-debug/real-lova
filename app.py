import gradio as gr
import os
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import asyncio
from groq import Groq
from duckduckgo_search import DDGS
import edge_tts

# ======================
# CONFIG
# ======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

chat_history = []

# ======================
# FUNCTIONS
# ======================

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
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Save temporary wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, sample_rate, (audio_data * 32767).astype(np.int16))
            temp_filename = f.name

        # Use Groq Whisper API (no heavy model in memory)
        with open(temp_filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_filename, file.read()),
                model="whisper-large-v3",
                language="en"
            )

        os.unlink(temp_filename)
        return transcription.text.strip()

    except Exception as e:
        return f"Transcription error: {str(e)}"


def ask_llm(user_text):
    global chat_history

    if not user_text or not user_text.strip():
        return "Please say or type something.", format_history()

    search_keywords = ["who", "what", "when", "where", "stock", "price", "news", "weather", "current", "latest", "today", "kya", "kaun", "kab", "kahan"]
    need_search = any(word in user_text.lower() for word in search_keywords)
    extra_info = search_web(user_text) if need_search else ""

    messages = [
        {
            "role": "system",
            "content": "You are Lova, a helpful AI assistant created by Saksham Jha. You understand English and Hindi. Keep replies short and natural."
        }
    ]

    for msg in chat_history[-6:]:
        messages.append(msg)

    if extra_info:
        prompt = f"User: {user_text}\n\nReal-world info:\n{extra_info}\n\nAnswer helpfully:"
    else:
        prompt = user_text

    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-20b",
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Sorry, error: {str(e)}"

    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": reply})

    return reply, format_history()


def format_history():
    if not chat_history:
        return "No messages yet."
    text = ""
    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "Lova"
        text += f"**{role}:** {msg['content']}\n\n"
    return text


async def text_to_speech(text):
    try:
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            await communicate.save(f.name)
            return f.name
    except Exception:
        return None


def process_voice(audio):
    user_text = transcribe_audio(audio)
    if not user_text or user_text.startswith("Transcription error"):
        return user_text or "I couldn't hear you clearly.", format_history(), None
    reply, history_text = ask_llm(user_text)
    audio_path = asyncio.run(text_to_speech(reply))
    return reply, history_text, audio_path


def process_text(user_text):
    reply, history_text = ask_llm(user_text)
    audio_path = asyncio.run(text_to_speech(reply))
    return reply, history_text, audio_path


def clear_chat():
    global chat_history
    chat_history = []
    return "", "Chat cleared.", None


# ======================
# INTERFACE
# ======================

with gr.Blocks(title="Lova AI - Smart Voice & Chat Assistant") as demo:
    gr.Markdown("""
    # 🎙️ Lova AI
    ### Smart Voice & Chat Assistant by Saksham Jha
    
    Speak or type in **English** or **Hindi**.
    """)

    history_box = gr.Markdown("No messages yet.")

    with gr.Tab("🎤 Voice"):
        audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Speak here")
        voice_btn = gr.Button("Talk to Lova", variant="primary")

    with gr.Tab("💬 Text"):
        text_input = gr.Textbox(label="Type your message", placeholder="Type in English or Hindi...")
        text_btn = gr.Button("Send", variant="primary")

    reply_box = gr.Textbox(label="Lova's Reply")
    audio_output = gr.Audio(label="Lova's Voice", autoplay=True)
    clear_btn = gr.Button("Clear Chat")

    voice_btn.click(fn=process_voice, inputs=audio_input, outputs=[reply_box, history_box, audio_output])
    text_btn.click(fn=process_text, inputs=text_input, outputs=[reply_box, history_box, audio_output])
    text_input.submit(fn=process_text, inputs=text_input, outputs=[reply_box, history_box, audio_output])
    clear_btn.click(fn=clear_chat, outputs=[reply_box, history_box, audio_output])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)