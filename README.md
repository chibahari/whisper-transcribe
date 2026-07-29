# Multilingual Interview Transcription (Whisper MLX)

Transcription pipeline for research interviews on AI incident reporting in
Malaysia. Interviews are 45 min – 3 h long, 2–6 speakers, and mix English
and Malay with frequent code-switching.

For the design story, iteration history, and known limitations, see
[`writeup.md`](writeup.md).

## Requirements

- macOS on Apple Silicon (uses MLX for Whisper and MPS for pyannote)
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `ffmpeg` on `PATH` (used for the initial audio conversion)
- A HuggingFace access token with the `pyannote/speaker-diarization-3.1`
  model terms accepted at https://hf.co/pyannote/speaker-diarization-3.1

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
- `raw_transcript.json` — Whisper's full output (segments, words,
  timestamps, `compression_ratio`, `avg_logprob`, …).
- `final_transcript.txt` — human-readable transcript with speaker labels
  (`[SPEAKER_00]`, `[SPEAKER_01]`, …). Segments with suspiciously
  repetitive Whisper output are prefixed with
  `[UNCERTAIN — please review]` — spot-check these.

Expect ~14 min of runtime for a 25 min interview on an M-series Mac.

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

Interview-topic-specific vocabulary lives in `transcribe.py::TRANSCRIBE_PROMPT`.
Edit it for a different interview series (see [`writeup.md`](writeup.md)
§7 for rationale).

## Iterating on Whisper configs

`experiment.py` is a config-comparison harness — useful when tuning
Whisper decoder parameters against a specific clip.

```bash
# Run one or more named configs against a clip
python experiment.py --clip tmp/clips/full_sample.wav --configs C_rich_prompt

# Compare multiple configs side by side
python experiment.py --clip tmp/clips/full_sample.wav \
    --configs A_baseline C_rich_prompt
```

Per-config outputs land in `output/experiments/{clip}/{config}.{json,txt}`
alongside a `summary.txt` with loop counts and timings. Configs are
defined at the top of `experiment.py`; add new ones there.

## Repository layout

```
main.py                 pipeline entrypoint
preprocess.py           WAV conversion + normalise + noise reduction
transcribe.py           diarize(), vad_chunks(), transcribe(), merge_…()
experiment.py           config-comparison harness (mlx_whisper only)
data/raw/               source recordings (gitignored)
tmp/                    intermediate WAVs + test clips
output/{RUN_NAME}/      final transcript + JSON + cleaned WAV + diarization.json
output/experiments/     per-config comparison artefacts
writeup.md              design history and known limitations
CLAUDE.md               per-session changelog
```
