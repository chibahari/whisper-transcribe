# Multilingual Interview Transcription (Whisper MLX)

Transcription pipeline for research interviews on AI incident reporting in
Malaysia. Interviews are 45 min – 3 h long, 2–6 speakers, and mix English and
Malay with frequent code-switching.

## Pipeline

`main.py` runs the four stages in order:

1. **Convert** input audio to 16 kHz mono WAV (`preprocess.convert_to_wav`).
2. **Enhance**: pydub loudness normalisation → `noisereduce` (first 0.5 s
   used as noise profile) (`preprocess.enhance_audio`).
3. **Diarize** with `pyannote/speaker-diarization-3.1` on MPS
   (`transcribe.diarize`). Requires a HuggingFace token; the model is gated.
4. **Transcribe** with `mlx_whisper` using `whisper-large-v3-mlx`
   (`transcribe.transcribe`).
5. **Merge** speaker labels into the transcript by time-overlap
   (`transcribe.merge_diarization_and_transcript`).

`.env` supplies `HF_TOKEN`, `INPUT_FILE`, and `RUN_NAME`. Outputs land in
`output/{RUN_NAME}/`.

## Whisper configuration — what worked

Two problems drove the initial iteration:

1. Whisper getting stuck in **repetition loops** ("Saya rasa kita memang ada
   sebuah division…" repeated 11× in the middle of one interview) and
   emitting **YouTube-style hallucinations** ("Terima kasih kerana
   menonton", "Jangan lupa untuk beri komentar di bawah ini") during silent
   passages.
2. **Over-translation to Malay** when the interviewer was speaking English,
   plus wrong proper nouns (e.g. "Naksa" for "NACSA").

The current `transcribe.transcribe()` uses:

```python
mlx_whisper.transcribe(
    wav_path,
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
    language=None,                        # auto-detect per 30 s window
    word_timestamps=True,
    condition_on_previous_text=False,     # breaks cross-window feedback loops
    temperature=(0.0, 0.2, 0.4, 0.6),     # fallback ladder on failure
    no_speech_threshold=0.6,
    compression_ratio_threshold=1.35,     # flag suspiciously repetitive output
    logprob_threshold=-1.0,
    hallucination_silence_threshold=2.0,  # ← the key fix for loops
    task="transcribe",
    initial_prompt=TRANSCRIBE_PROMPT,     # ← the key fix for language + names
)
```

`TRANSCRIBE_PROMPT` names the interview topic, the organisations involved
(MCMC, NSC, NACSA, CSM, BNM), the relevant law (CMA 1998), and domain
vocabulary (`data breach, ransomware, phishing, SMB port 445, telco
licensees, aduan, penyedia rangkaian, …`). Whisper conditions its decoder
on this text, which anchors the vocabulary and reduces language flipping.

## What we tried, and what we learnt

Iteration harness: `experiment.py`. Each config is a named dict of
`mlx_whisper.transcribe` kwargs; results land under
`output/experiments/{clip}/`.

| Config | Change vs baseline | Cross-segment loops (6-min clip) | YouTube-style hallucinations |
|---|---|---|---|
| A — baseline | current settings before iteration | 3 catastrophic regions (incl. 55 blank segs stuck at 258.8 s) | yes |
| B — hallucination guard | + `hallucination_silence_threshold=2.0` | 0 | 0 |
| C — rich prompt (kept) | B + expanded `initial_prompt` | 0 | 0 |
| D — turbo model | C but `whisper-large-v3-turbo` | 0 | 0 |
| E — aggressive temp ladder | temperatures `(0.0, 0.4, 0.8, 1.0)` | 5 (worse than baseline) | 0 |
| F — extra fallback step | 5-step ladder ending at 0.8 | 1 (worse than C) | 0 |
| G — tight CR threshold | F + `compression_ratio_threshold=1.2` | 2 (worse than C) | 0 |

**Wins.**
- `hallucination_silence_threshold=2.0` (config B) was the single biggest
  fix. It requires `word_timestamps=True` (already set) and skips silent
  runs longer than the threshold when a hallucination is suspected.
- The **rich prompt** (config C) preserved English utterances that config B
  translated to Malay, and got proper nouns right (NACSA vs. "Naksa").

**Traps.**
- **Turbo (`large-v3-turbo`) is not a win here.** It runs faster but
  introduces its own hallucination pattern — spurious "Use" tokens
  sprinkled through the output.
- **Wider temperature fallback makes things worse.** Whisper's fallback
  keeps the *last* attempt when the ladder is exhausted. High temperatures
  (0.8, 1.0) generate more looping content in confusing audio regions, so
  raising the top of the ladder amplifies loops rather than escaping them.
- **Tightening `compression_ratio_threshold`** only helps if the fallback
  attempts *actually escape* the loop. On stubborn regions they don't, so
  tightening just makes fallback trigger more often without changing the
  kept output.

**Remaining known issue.** Two single-segment internal loops on the sample
(at 83 s "keperluan…" and 268 s "kebanyakan…"), ~4 % of the transcript.
These look tied to speaker hesitation/filler that Whisper misreads as
repetition. Configuration alone cannot fix them — see next section.

## Not yet tried (natural next steps)

- **VAD-based pre-chunking**: break audio on silence before Whisper's 30 s
  window boundary, so internal loops can't accumulate within a chunk.
- **Uncertainty flagging in post**: mark any segment with
  `compression_ratio > 6.0` as `[UNCERTAIN — please review]` so reviewers
  can spot-check.
- **Pre-processing tweaks**: `preprocess.enhance_audio` currently uses
  `noisereduce` with `prop_decrease=0.8` and a 0.5 s noise profile. Worth
  trying `prop_decrease=0.5` (less aggressive) or skipping noise reduction
  for regions with already-clean audio.
- **Diarization headroom**: `main.py` caps `max_speakers=3` even though the
  requirements allow up to 6.

## Layout

```
main.py                 pipeline entrypoint
preprocess.py           WAV conversion + normalise + noise reduction
transcribe.py           diarize(), transcribe(), merge_…()
experiment.py           config-comparison harness (mlx_whisper only)
data/raw/               source recordings
tmp/                    intermediate WAVs + test clips
output/{RUN_NAME}/      final transcript + JSON + cleaned WAV
output/experiments/     per-config comparison artefacts
```

## Run

```bash
# uv-managed venv
uv sync
source .venv/bin/activate

# populate .env with HF_TOKEN, INPUT_FILE, RUN_NAME
python main.py

# to iterate on Whisper configs against a single clip:
python experiment.py --clip tmp/clips/full_sample.wav --configs C_rich_prompt
```
