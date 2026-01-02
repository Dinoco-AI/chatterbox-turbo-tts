import numpy as np
import torch
import gradio as gr
from chatterbox.tts_turbo import ChatterboxTurboTTS

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading model on {device}...")
model = ChatterboxTurboTTS.from_pretrained(device=device)

REFERENCE_AUDIO = "styletts2.mp3"

def generate_speech_stream(text):
    for chunk, sr in model.generate_stream(text, audio_prompt_path=REFERENCE_AUDIO, chunk_size=50):
        yield sr, chunk

demo = gr.Interface(
    fn=generate_speech_stream,
    inputs=gr.Textbox(
        label="Text to speak",
        placeholder="Enter text here...",
        value="Hello! This is a streaming test. The audio plays as it generates. Each chunk plays immediately as it's ready."
    ),
    outputs=gr.Audio(label="Generated Speech", autoplay=True, streaming=True),
    title="Chatterbox Turbo TTS - Streaming",
    description="Enter text and hear it generate in real-time"
)

if __name__ == "__main__":
    demo.launch(share=True)
