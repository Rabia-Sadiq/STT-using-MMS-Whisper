# Phase 3 — Meta MMS + Whisper

Offline Punjabi/Urdu speech transcription using Meta's MMS (Massively Multilingual Speech) combined with Whisper for English output. The lightest pipeline — designed for low-spec hardware.

---

## How It Works

```
Audio → ffmpeg (resample to 16kHz mono WAV)
      → MMS [lang=pan] → Punjabi Gurmukhi
      → MMS [lang=urd-script_arabic] → Urdu

Audio → Whisper [task=translate, lang=ur] → English
```

**Why MMS?**
MMS uses a language adapter architecture — a single shared base model with small per-language adapters. This makes it dramatically more memory efficient than SeamlessM4T while still providing native ASR for Punjabi and Urdu. Whisper handles English because MMS is an ASR-only model and cannot translate between languages.

---

## Output Languages

| Language | Script | Quality | Method |
|---|---|---|---|
| English | Latin | Excellent | Whisper direct translation |
| Urdu | Arabic (Nastaliq) | Partial | MMS native ASR |
| Punjabi Gurmukhi | Gurmukhi | Good — phonetically faithful | MMS native ASR |
| Punjabi Shahmukhi | — | Not supported | pnb absent from mms-1b-all |

---

## Requirements

- Python 3.9+
- 4 GB RAM minimum (lightest of all three phases)
- FFmpeg installed and in PATH
- No GPU required

---

## Installation

```bash
mkdir mms_transcriber
cd mms_transcriber
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install transformers torch torchaudio soundfile sentencepiece faster-whisper librosa
```

### Install FFmpeg (Windows)

> **Note:** If FFmpeg is blocked by your organization's Device Guard policy, use the `librosa` audio loader instead — see the Configuration section below.

Download from: https://github.com/BtbN/FFmpeg-Builds/releases

Download `ffmpeg-master-latest-win64-gpl.zip` (NOT the shared variant), extract, add `bin` folder to System PATH.

Verify:
```bash
ffmpeg -version
```

---

## Model Downloads

Downloaded automatically on first run and cached locally.

| Model | Size | Purpose |
|---|---|---|
| MMS mms-1b-all | ~1.2 GB | Punjabi and Urdu native ASR |
| Whisper medium | ~1.5 GB | English translation |

**Total: ~2.7 GB** — lightest pipeline of all three phases.

---

## Usage

1. Place your audio file in the project folder
2. Edit `transcribe.py` and set your filename:

```python
AUDIO_FILE = "your_audio.mp3"
```

3. Run:

```bash
python transcribe.py
```

Results are printed to terminal and saved to `results.txt`.

**Supported audio formats:** `.mp3` `.wav` `.m4a` `.ogg` `.flac` `.mp4`

---

## Configuration

```python
AUDIO_FILE = "audio.mp3"              # Your audio file
MODEL_ID   = "facebook/mms-1b-all"   # MMS model
```

### FFmpeg Blocked? Use librosa instead

If FFmpeg is blocked by your organization, replace the `load_audio` function with:

```python
def load_audio(path):
    import librosa
    print(f"\nLoading audio: {path}")
    samples, _ = librosa.load(path, sr=16000, mono=True)
    waveform = torch.tensor(samples).unsqueeze(0)
    print(f"  Duration: {len(samples)/16000:.1f} seconds")
    return waveform
```

librosa is pure Python and does not require FFmpeg.

---

## Sample Output

**Input (spoken):** Hello ki hall ay, theek ho, ki karday pay ho, baki sb da ki hall ay

```
PUNJABI (Gurmukhi):
ਪਹਿਲੋ ਕੀ ਹਾਲੇ ਕੀੰਗ ਤੋਂ ਕੀ ਕਰਦੇ ਪਏ ਓ ਵਾਕੀ ਸਾਰਿਆਂ ਗਾ ਕੀ ਹਾਲੇ

URDU:
لوگ کی ہالے ین تو کی کردے پے او باقی ساریاں گا کھیاے

ENGLISH:
Hello, how are you? What are you doing? How are the rest of the people?
```

---

## Hardware Performance

| Hardware | Latency (8s clip) | RAM Usage |
|---|---|---|
| CPU only (7.4 GB RAM) | ~80-150 seconds | ~2.0-2.5 GB |
| CPU only (16 GB RAM) | ~60-120 seconds | ~2.0-2.5 GB |

Most suitable pipeline for machines with limited RAM.

---

## Key Findings

- **Lightest pipeline** — 2.7 GB total, ~2.0-2.5 GB RAM during inference.
- **MMS is ASR-only** — it cannot translate between languages. Whisper is required as a separate model for English output.
- **Punjabi Shahmukhi not available** — `pnb` language code is absent from `mms-1b-all` adapter list. Only Gurmukhi (`pan`) is supported.
- **Urdu output is partial** — MMS phonetically transcribes Urdu but mixed Lahori dialect causes some words to be garbled.
- **Hallucination warning** — comparative testing showed that using MMS intermediate text as input to Whisper's translation causes semantic hallucinations and logic inversions on complex sentences. If translation accuracy is the primary concern, use Phase 1 (Whisper-only) instead.

---

## MMS Language Codes

MMS uses specific language codes for its adapters:

| Language | Code | Script |
|---|---|---|
| Punjabi (Gurmukhi) | `pan` | Gurmukhi |
| Urdu | `urd-script_arabic` | Arabic (Nastaliq) |
| English | `eng` | Latin (ASR only — cannot translate) |
| Punjabi Shahmukhi | Not available | — |

Check all available languages:

```python
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
print(len(processor.tokenizer.all_special_tokens), "languages available")
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` or blocked | Switch to librosa loader — see Configuration above |
| `torchcodec` import error | Replace `torchaudio.load()` with the librosa or soundfile loader |
| `Module not found` | Make sure venv is activated |
| Poor Urdu output | Expected — use Phase 2 (SeamlessM4T) for better Urdu |
| Hallucinations in English | Use Phase 1 (Whisper-only) for accuracy-critical content |

---

## When to Use This Phase

**Use Phase 3 when:**
- Machine has less than 8 GB RAM
- Speed matters more than perfect Urdu accuracy
- You only need Punjabi Gurmukhi and English output
- GPU is not available

**Do not use Phase 3 when:**
- Punjabi Shahmukhi output is required — use Phase 2
- Natural colloquial Urdu is required — use Phase 2
- High translation accuracy on technical/complex content — use Phase 1

---

## License

| Component | License |
|---|---|
| MMS (mms-1b-all) | CC-BY-NC 4.0 — non-commercial use only |
| Whisper (faster-whisper) | MIT |
| PyTorch | BSD |

---

## Related

- [Phase 1 — Whisper + NLLB](../punjabi_transcriber/README.md) — best English and formal Urdu output
- [Phase 2 — SeamlessM4T v2](../seamless_transcriber/README.md) — best Urdu and only Shahmukhi support
