import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import soundfile as sf
import subprocess
import numpy as np
from transformers import AutoProcessor, Wav2Vec2ForCTC
from faster_whisper import WhisperModel

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
AUDIO_FILE = "punjabi.mp3"
MODEL_ID   = "facebook/mms-1b-all"

# ─────────────────────────────────────────────
# LOAD AUDIO — convert to 16kHz WAV via ffmpeg
# ─────────────────────────────────────────────
def load_audio(path):
    print(f"\nLoading audio: {path}")
    wav_path = "temp_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", path,
        "-ar", "16000",
        "-ac", "1",
        wav_path
    ], capture_output=True)
    waveform, sample_rate = sf.read(wav_path)
    os.remove(wav_path)
    print(f"  Duration: {len(waveform)/sample_rate:.1f} seconds")
    return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

# ─────────────────────────────────────────────
# MMS TRANSCRIPTION (Punjabi + Urdu)
# ─────────────────────────────────────────────
def mms_transcribe(processor, model, waveform, lang, label):
    print(f"\n  MMS transcribing → {label}...")
    processor.tokenizer.set_target_lang(lang)
    model.load_adapter(lang)
    inputs = processor(
        waveform.squeeze(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs).logits
    ids    = torch.argmax(outputs, dim=-1)
    result = processor.batch_decode(ids)[0]
    print(f"  {label}: {result}")
    return result

# ─────────────────────────────────────────────
# WHISPER TRANSLATION → English
# ─────────────────────────────────────────────
def whisper_to_english(audio_path):
    print("\n  Whisper translating → English...")
    print("  (Downloads ~1.5GB first time)")
    whisper = WhisperModel("medium", device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(
        audio_path,
        language="ur",
        task="translate",
        beam_size=5,
        vad_filter=True,
    )
    result = " ".join(s.text.strip() for s in segments)
    print(f"  English: {result}")
    return result

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"ERROR: '{AUDIO_FILE}' not found.")
        return

    print("=" * 55)
    print("  META MMS + WHISPER — Phase 3 Pipeline")
    print("=" * 55)
    print("\n  MMS  → Punjabi Gurmukhi + Urdu (direct ASR)")
    print("  Whisper → English (best translation model)")

    # Load MMS once
    print("\nLoading MMS model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    print("✓ MMS loaded.")

    # Load audio once
    waveform = load_audio(AUDIO_FILE)

    # MMS: Punjabi and Urdu
    punjabi = mms_transcribe(processor, model, waveform,
                              lang="pan",
                              label="Punjabi Gurmukhi")

    urdu    = mms_transcribe(processor, model, waveform,
                              lang="urd-script_arabic",
                              label="Urdu")

    # Whisper: English
    english = whisper_to_english(AUDIO_FILE)

    # Results
    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    print(f"\n🟣 PUNJABI (Gurmukhi):\n{punjabi}\n")
    print(f"🔵 URDU:\n{urdu}\n")
    print(f"🟢 ENGLISH:\n{english}\n")

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(f"PUNJABI (Gurmukhi):\n{punjabi}\n\n")
        f.write(f"URDU:\n{urdu}\n\n")
        f.write(f"ENGLISH:\n{english}\n")
    print("✓ Saved to results.txt")

if __name__ == "__main__":
    main()