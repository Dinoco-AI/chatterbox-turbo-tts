import numpy as np
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Config
REFERENCE_AUDIO = "reference.wav"  # Your 5+ second reference audio
TEXT = "Hello! This is a streaming test. The audio plays as it generates."
OUTPUT_FILE = "streaming_output.wav"

# Load model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading model on {device}...")
model = ChatterboxTurboTTS.from_pretrained(device=device)

# Stream and collect chunks
print(f"Generating: {TEXT}")
chunks = []
for i, (chunk, sr) in enumerate(model.generate_stream(TEXT, audio_prompt_path=REFERENCE_AUDIO, chunk_size=50)):
    print(f"Chunk {i+1}: {len(chunk)} samples ({len(chunk)/sr:.2f}s)")
    chunks.append(chunk)

# Save combined audio
audio = np.concatenate(chunks)
ta.save(OUTPUT_FILE, torch.from_numpy(audio).unsqueeze(0), sr)
print(f"Saved to {OUTPUT_FILE} ({len(audio)/sr:.2f}s total)")
