I want to create an app that transcribes research interview recordings. The research is about AI incident reporting in Malaysia. Here are the requirements:

- Interviews are often multilingual with frequent code switching and mixture of vocabulary between languages. The languages typically used are English and Malay.
- Interviews can be between 45 minutes and 3 hours long.
- There are a minimum of two speakers, but speaker count can go up to six. Each speaker needs to be identified.
- Quality of recordings might not be super clear.

## Session log — 2026-07-28

Iterated on `mlx_whisper.transcribe` parameters to fix two reported problems:
repetition loops and poor multilingual quality.

### Test setup
- Sample audio: `data/raw/AIIR MCMC sample.mp3` (25 min 36 s).
- Config-comparison harness: `experiment.py`. Runs a named config against a
  clip and writes `output/experiments/{clip}/{config}.{json,txt}` plus a
  `summary.txt` with loop counts, high-compression-ratio counts, and timing.
- Short 6-min clip `tmp/clips/clip_10-16min.wav` (extracted from 10:00-16:00
  of the sample, a region likely to contain code-switching + a loop) used for
  iterating on configs. Full-sample WAV cached at
  `tmp/clips/full_sample.wav`.

### Configs tried (all in `experiment.py`)
- `A_baseline` — current settings (large-v3, temp ladder to 0.6,
  compression_ratio_threshold=1.35, basic prompt).
- `B_hallu_guard` — add `hallucination_silence_threshold=2.0`.
- `C_rich_prompt` — B + expanded `initial_prompt` with proper nouns.
- `D_turbo` — swap model to `whisper-large-v3-turbo` + rich prompt.
- `E_aggressive_temp` — temperature ladder `(0.0, 0.4, 0.8, 1.0)`.
- `F_extra_fallback` — 5-step ladder ending at 0.8.
- `G_tight_cr` — F + `compression_ratio_threshold=1.2`.

### Result summary
- `A_baseline` on the 6-min clip: **3 catastrophic hallucinations**
  ("Terima kasih kerana menonton", "Malaysia masuk Malaysia masuk", and a
  "Use Use Use" loop followed by ~55 blank segments all stuck at 258.8s).
- `C_rich_prompt` on the 6-min clip: **0 loops, 0 YouTube hallucinations**;
  preserves English speech that B translated to Malay; transcribes "NACSA"
  correctly (baseline wrote "Naksa").
- `C_rich_prompt` on the full 25-min sample: 0 cross-segment loops, 0
  YouTube hallucinations. **2 remaining single-segment internal loops** at
  83 s ("keperluan…") and 268 s ("kebanyakan…") — ~4 % of the transcript.
- `D_turbo` — turbo introduces spurious "Use" tokens sprinkled through the
  output. Rejected.
- `E`, `F`, `G` (wider/longer temperature ladders, tighter CR) — **worse**
  than C. At high temperatures Whisper generates *more* looping content in
  confusing regions, and it keeps the last fallback attempt, so raising the
  top temperature amplifies loops rather than escaping them.

### Applied change
`transcribe.py::transcribe` now uses config C: `hallucination_silence_threshold=2.0`
plus the expanded `TRANSCRIBE_PROMPT` (proper nouns for MCMC, NSC, NACSA,
CSM, BNM, CMA 1998, SII, aduan, penyedia rangkaian, etc.).

### Still-open avenues (not attempted)
- **VAD-based pre-chunking**: break audio on silence before Whisper's own
  30 s windows so single-segment internal loops can't accumulate.
- **Post-pass uncertainty flagging**: after transcription, mark any segment
  with `compression_ratio > 6.0` as `[UNCERTAIN — please review]` in the
  final transcript.
- **Audio pre-processing tweaks**: `preprocess.py` uses `noisereduce` with
  `prop_decrease=0.8` and the first 0.5 s as noise profile. This may be
  over-suppressing speaker-hesitation cues that trigger the internal loops;
  worth trying with `prop_decrease=0.5` or skipping noise reduction on
  cleaner regions.
- **Diarization quality**: `main.py` currently uses `max_speakers=3`, but
  the requirements allow up to 6. Not investigated in this session.

### Files touched this session
- `transcribe.py` — updated `transcribe()` with hallucination guard and
  richer prompt (module-level `TRANSCRIBE_PROMPT` constant).
- `experiment.py` — **new**, config-comparison harness.
- `output/experiments/` — **new**, per-config transcripts for the 6-min
  clip, the 4-min clip covering the loop region, and the full sample.
- `tmp/clips/` — **new**, extracted test clips.
