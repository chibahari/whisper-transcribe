# Multilingual Interview Transcription

Transcription pipeline for research interviews on AI incident reporting in
Malaysia. Interviews are 45 min – 3 h long, 2–6 speakers, and mix English
and Malay with frequent code-switching.

Uses `mesolitica/Malaysian-whisper-large-v3-turbo-v3` — a Whisper fine-tune
trained explicitly on Manglish (Malay-English code-switching) — via HF
transformers on MPS. See [`CLAUDE.md`](CLAUDE.md) for the iteration history
that led to this choice.

## Requirements

- macOS on Apple Silicon (uses MPS for Whisper and pyannote)
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `ffmpeg` on `PATH` (used for the initial audio conversion)
- A HuggingFace access token with the `pyannote/speaker-diarization-3.1`
  model terms accepted at https://hf.co/pyannote/speaker-diarization-3.1
- ~2 GB free disk for the mesolitica model (auto-downloaded on first run;
  see notes below if the download stalls)

## Setup

```bash
uv sync
source .venv/bin/activate
```

Create `.env` in the project root:

```dotenv
HF_TOKEN="hf_..."
INPUT_FILE="data/raw/your-interview.mp3"
RUN_NAME="descriptive_run_name"

# Speaker count for diarization. Set NUM_SPEAKERS if the roster is known
# (strongest constraint). Otherwise use MIN_SPEAKERS / MAX_SPEAKERS.
# NUM_SPEAKERS=3
MIN_SPEAKERS=2
MAX_SPEAKERS=6
```

## Run

```bash
python main.py
```

This executes the full pipeline: convert → enhance → diarize → transcribe
→ merge. Outputs land in `output/{RUN_NAME}/`:

- `clean.wav` — 16 kHz mono WAV after loudness normalisation and noise
  reduction.
- `raw_transcript.json` — full pipeline output (segments with per-segment
  timestamps, forced language, and a `_lang_counts` summary).
- `final_transcript.txt` — human-readable transcript with speaker labels
  (`[SPEAKER_00]`, `[SPEAKER_01]`, …). Consecutive same-speaker segments
  are joined into single turns.

Expect ~14 min of runtime for a 25 min interview on an M-series Mac
(after the first-run model download).

### If the mesolitica model download stalls

Unauthenticated HuggingFace connections can silently stall on large model
downloads. If the first `python main.py` hangs at model load, kill it and
pre-download with the parallel Rust downloader:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download \
  mesolitica/Malaysian-whisper-large-v3-turbo-v3
```

## Configuration reference

All configuration goes through `.env`:

| Variable | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | yes | HuggingFace token for the gated pyannote model |
| `INPUT_FILE` | yes | Path to the source recording (any format ffmpeg reads) |
| `RUN_NAME` | yes | Subdirectory name under `output/` |
| `NUM_SPEAKERS` | either this or the range | Exact speaker count when known |
| `MIN_SPEAKERS` | either this or `NUM_SPEAKERS` | Lower bound for pyannote's speaker search |
| `MAX_SPEAKERS` | either this or `NUM_SPEAKERS` | Upper bound for pyannote's speaker search |

### Tuning knobs

- `en_bias` (default 0.3) in `transcribe.transcribe()` — force English if
  `p(en)` at the language-token position exceeds this threshold. Set lower
  for more English bias, higher for less. Chosen because English mis-routing
  was the observed failure mode on the source recordings; lower for
  Malay-dominant material.

## Iterating on Whisper configs

`experiment.py` is a config-comparison harness — useful when tuning
transcription parameters or trying alternate backends against a specific
clip. It has three backends: `mlx_whisper` (historical baselines),
`hf_transformers` (mesolitica via pipeline), and
`hf_transformers_per_chunk_lang` (mesolitica + VAD + per-chunk language
detect — this is what `transcribe.py` uses in production).

```bash
# Run one or more named configs against a clip
uv run python experiment.py --clip tmp/clips/full_sample.wav \
    --configs C_rich_prompt M4_detect_lang
```

Per-config outputs land in `output/experiments/{clip}/{config}.{json,txt}`
alongside a `summary.txt`. See
[`output/experiments/README.md`](output/experiments/README.md) for a full
list of configs and what each one tested.

## Repository layout

```
main.py                 pipeline entrypoint
preprocess.py           WAV conversion + normalise + noise reduction
transcribe.py           diarize(), vad_chunks(), transcribe(), merge_…()
experiment.py           config-comparison harness (multi-backend)
data/raw/               source recordings (gitignored)
tmp/clips/              test clips for experiment.py
output/{RUN_NAME}/      final transcript + JSON + cleaned WAV + diarization.json
output/experiments/     per-config comparison artefacts
writeup.md              older design history (pre-mesolitica; kept for context)
CLAUDE.md               per-session changelog
```
