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

## Session log — 2026-07-29

Focus: diarization configurability, merge robustness, and loop containment.

### Applied changes

- **Configurable speaker count.** `main.py` now reads `NUM_SPEAKERS` (exact)
  or `MIN_SPEAKERS` / `MAX_SPEAKERS` (range) from `.env`. Errors if none
  are set. `transcribe.diarize()` accepts an optional `num_speakers`
  argument and forwards only the supplied kwargs to pyannote. Previous
  hard-coded `min=2, max=3` under-covered the 2–6 requirement.
- **Merge bug fix.** `merge_diarization_and_transcript.get_speaker()` used
  strict containment (`seg.start <= start and seg.end >= end`), which
  labelled ~47 % of Whisper segments as UNKNOWN whenever a segment
  straddled a diarization turn boundary. Rewrote to use max time-overlap.
  On the sample: UNKNOWN dropped from 64/136 (47 %) to 2/60 (3 %).
- **Uncertainty tag.** Added `UNCERTAIN_CR_THRESHOLD` and
  `UNCERTAIN_TAG`. Segments with `compression_ratio > 2.4` are prefixed
  with `[UNCERTAIN — please review]` in `final_transcript.txt`.
- **VAD pre-chunking.** New `transcribe.vad_chunks()` uses Silero VAD
  (`min_silence=0.5 s`, `max_chunk=30 s`) to split audio at natural
  silences. `transcribe()` now loops over chunks, calls `mlx_whisper` per
  chunk with the audio as a numpy array, and offsets segment timestamps
  back to absolute time. `silero-vad` added to `pyproject.toml`.

### Experiments this session

**Validation run (pre-VAD)** — `MAX_SPEAKERS=6`, threshold=6.0.
- 10-segment cross-segment loop at 244–271 s, `cr=8.51`. **New** loop
  location vs the 2026-07-28 session's 268 s loop — Whisper's decoder is
  non-deterministic in confusing regions, so a single run is not a
  reliable measurement.
- Diarization returned 2 speakers (plausible — 1-on-1 interview).

**Threshold 6.0 → 2.4** — no visible effect on this transcript. The CR
distribution is bimodal (real segments < 2.2, catastrophic loops > 8) with
nothing in between. Subtle in-segment loops (e.g. "mempunyai pengaruhan" ×5
at `cr=2.17`) still slip through, because gzip overhead dominates on short
strings — a fundamental limitation of per-segment CR.

**`prop_decrease=0.5` (lighter noise reduction).** Worse than 0.8. Peak
`cr=17.03`; three new loop regions (213 s, 577–605 s, 1212 s) appeared in
the exact stretches the 0.8 baseline transcribed cleanly (data breach /
ransomware and SMB port 445 discussion). Reverted.

**VAD pre-chunking (Silero).** Kept — big win. 133 chunks (mean 10.5 s,
min 5 s, max 48.5 s). `cr > 6.0` segments 10 → 1. Recovered dialog in
the former loop region ("From aduan perspective, aduan is safe." / "So,
from aduan, they will segregate based on which is related to
cybersecurity…"). Remaining loop is a single-chunk 24 s "iaitu iaitu…" run
at 1484 s, bounded by the VAD chunk boundary and flagged by the tag.
Whisper runtime fell ~20 min → ~14 min.

### Still-open avenues (not attempted)

- **Short in-segment loops still miss.** Per-segment `compression_ratio`
  can't catch them; a text-level heuristic (repeated n-gram detector or
  character diversity ratio) would.
- **Stray single-segment YouTube hallucinations** ("Terima kasih.") still
  leak through at chunk-trailing silences. Could tune
  `hallucination_silence_threshold` per chunk or add a text-level filter.
- **VAD chunks can exceed `max_chunk_s`** when a speech region has no
  silences ≥ 0.5 s (48 s max observed). A forced mid-region split with a
  small overlap would fully cap chunk length.

### Files touched this session

- `main.py` — env-driven speaker count.
- `transcribe.py` — `diarize()` speaker kwargs, `vad_chunks()` +
  chunked `transcribe()`, max-overlap merge, uncertainty tag constants.
- `preprocess.py` — tried `prop_decrease=0.5`, reverted to `0.8`.
- `pyproject.toml`, `uv.lock` — added `silero-vad`.
- `.env` — new speaker-count keys with docs.
- `README.md`, `CLAUDE.md` — updated for this session.
- **new**: `output/MCMC_test_pd05/`, `output/MCMC_test_vad/`
  (comparison run artefacts).
- **new**: `output/MCMC_test/diarization.json` — cached so future
  merge-only iterations skip the ~2 min diarize step.

## Session log — 2026-08-06

Focus: swap the transcription model to a Malaysia-specific fine-tune, add
per-chunk language routing, clean up the project.

### Root cause of residual mistranslation

User feedback after tagging ~5 min of `MCMC_test/final_transcript.txt`:
biggest problem was English speech (especially interviewer utterances in
the first 3 min) being decoded as gibberish Malay. Also short internal
loops the per-segment compression-ratio filter didn't catch, plus stray
"hallucinated Malay" at silences. The failure mode was mesolitica's own
Malay-biased language head picking Malay for short English-only chunks,
then dutifully outputting Malay tokens.

### Applied changes

- **Model swap.** `transcribe.py` now uses
  `mesolitica/Malaysian-whisper-large-v3-turbo-v3` via HF transformers +
  MPS (bfloat16) instead of `mlx_whisper` with `whisper-large-v3-mlx`.
  The Malaysian fine-tune was trained explicitly on Manglish
  (Malay-English code-switching), Mandarin, and Tamil. No MLX conversion
  of the fine-tune is published; the mlx backend is retained only in
  `experiment.py` for historical baseline configs.
- **Per-chunk language detection with English bias.** `transcribe.py`
  now runs Silero VAD to pre-chunk the audio, does a one-step
  language-token logit probe per chunk, and forces `language=en` when
  `p(en) > 0.3` (default `en_bias`), else the top-probability language.
  This bypasses the pipeline's whole-chunk auto-detect that mis-routes
  short English utterances to Malay.
- **Speaker-turn merging in the final transcript.**
  `merge_diarization_and_transcript()` now concatenates consecutive
  same-speaker segments into single turns. Mesolitica pipeline output is
  word-level (~800 segments on the 26 min sample); without merging, the
  final transcript would look shattered.
- **Deps**: `uv add transformers accelerate hf_transfer`.
  `transformers==5.14.1`, `accelerate==1.14.0`, `hf-transfer==0.1.9`.
  `hf_transfer` was needed for a reliable download —
  unauthenticated HF Hub connections silently stalled twice on the
  1.6 GB safetensors download; the rust-based parallel downloader
  completed in ~9 min.
- **Cleanup.** Deleted `~700 MB` of stale preprocessing intermediates
  from `tmp/`; deleted `output/MCMC_test_pd05/`, `output/MCMC_test_vad/`
  (session-log-era comparison artefacts); deleted `diagnostic.py` (the
  browser-based tagging tool from earlier this session had a Safari
  `file://` sandbox issue that made JSON export unreliable — abandoned
  in favor of qualitative user feedback); deleted `.DS_Store`,
  `__pycache__`, and two orphan May-era transcript files at the root
  of `output/`.

### Experiments this session

All in `output/experiments/`; per-config details in
`output/experiments/README.md`.

**Iteration on the mesolitica pipeline** (all evaluated on full 26 min
sample vs. `C_rich_prompt` baseline):

- `M_mesolitica_turbo` — stride_length_s=5. Fixed cross-segment loops
  and most mistranslation, but introduced intra-turn duplication (same
  clause transcribed once in English and once in formal Malay from
  chunk overlap regions). 803 raw segments, 19195 chars, 700 s runtime.
- `M2_stride0` — stride=0. Duplication fixed (16251 chars, ~15%
  smaller). Still failed on 0–13s opening ("Okay, jadi kita mulakan…")
  and 397–430s interviewer stretch (all formal Malay). 433 s runtime.
- `M3_lang_en` — forced English globally. Fixed interviewer stretches
  but over-anglicized Manglish content. Rejected.
- `M4_detect_lang` — VAD + per-chunk language probe + en_bias=0.3.
  **Fixed all three previously-flagged regions.** Language routing
  split ~52/48 en/ms on the sample. 968 segments, 16461 chars,
  840 s runtime (20% slower than M2 due to the extra probe pass; user
  is optimizing for quality, not speed).

### Still-open avenues (not attempted)

- **Uncertainty flagging is gone.** Mesolitica pipeline output doesn't
  expose per-segment compression_ratio, so the `[UNCERTAIN — please
  review]` tag from the prior session's transcript is no longer emitted.
  M4 produced 0 loops on the 26 min sample so this hasn't bitten yet;
  if loops resurface, a text-level detector (repeated n-gram or low
  character diversity) would replace it.
- **Gemini audio transcription** as an A/B backend. Deferred at user's
  request — proceed only if mesolitica quality regresses on future
  interviews.
- **`transcribeprecise` task token.** Mesolitica model card advertises
  a custom word-timestamp task token; we register it in the tokenizer
  but still call the standard `transcribe` task. Word-level timestamps
  aren't needed for the current merge logic.
- **Documenting bias.** `en_bias=0.3` was chosen because English
  mis-routing was the observed failure. If the source audio shifts
  toward more Malay-dominant interviews, this threshold likely needs
  lowering.

### Files touched this session

- `transcribe.py` — full rewrite: mesolitica pipeline loader,
  per-chunk language probe with English bias, speaker-turn merging in
  `merge_diarization_and_transcript()`. `SAMPLE_RATE` and `vad_chunks`
  preserved for `experiment.py`'s new backend.
- `experiment.py` — added `BACKEND_HF` and
  `BACKEND_HF_PER_CHUNK_LANG` backends and their transcribe functions;
  added configs `M_mesolitica_turbo`, `M2_stride0`, `M3_lang_en`,
  `M4_detect_lang`. MLX baselines A–G kept intact.
- `pyproject.toml`, `uv.lock` — added `transformers`, `accelerate`,
  `hf_transfer`.
- `.claude/settings.local.json` — expanded permissions for git safe
  subcommands, ffmpeg, and read-only bash to reduce per-command
  approvals during this session's iteration.
- **new**: `output/experiments/README.md` — documents every config in
  `experiment.py::CONFIGS` and why each was tried/rejected/adopted.
- **new**: `output/experiments/full_sample/M4_detect_lang*.{json,txt}`
  and `M4_detect_lang_merged.txt` — the adopted config's output on
  the sample, plus the diarized+merged view.
- **deleted**: `diagnostic.py`, `output/MCMC_test_pd05/`,
  `output/MCMC_test_vad/`, ~700 MB of `tmp/` preprocessing
  intermediates, `output/final_transcript.txt` +
  `output/raw_transcript.json` (May-era orphans).

## Session log — 2026-09-14

Focus: restore the `[UNCERTAIN — please review]` tag (lost in the mlx →
mesolitica migration) using a text-level detector instead of per-segment
`compression_ratio`.

### Applied change

- **Text-level loop detector.** `transcribe.py` gains
  `contains_loop(text)` which flags a turn when it contains either
  `LOOP_UNIGRAM_RUN=5` copies of the same word in a row or
  `LOOP_BIGRAM_RUN=4` copies of the same 2-word phrase in a row (non-
  overlapping). Tokenization is `[\w']+` so punctuation doesn't break
  the run.
- **Applied at the turn level, not the segment level.** Mesolitica emits
  ~word-sized segments (~16 chars mean); a per-segment tag would be
  meaningless. `merge_diarization_and_transcript()` now prefixes each
  concatenated turn with `UNCERTAIN_TAG` when `contains_loop(turn_text)`
  fires.

### Validation

- Historic loop strings from prior sessions (`kebanyakan × 15+`,
  `iaitu × 8`, `Use × 5`, `mempunyai pengaruhan × 4`,
  `Malaysia masuk × 4`) all flag True.
- Clean examples do not flag, including natural stutter cases:
  `"yes, yes, I understand"`, `"very very important"`, `"okay okay so
  we started"`, `"and and and"`, `"I think, I think, I think we can"`
  (3× "I think" — under the 4-repeat bigram threshold),
  `"no no no no"` (4× "no" — under the 5-repeat unigram threshold).
- Re-scanning M4's merged output flags **0/65 turns**, matching the
  prior session's observation that M4 had no loops on the sample.
- Synthetic plumbing test: injecting 5 word-segments of `"kebanyakan"`
  into a fake transcript produces exactly one tagged turn.

### Rationale for thresholds

- `unigram_run=5`: real loops observed in prior sessions had 10-25+
  repeats, so 5 is well inside the safety margin. Natural affirmations
  ("yes yes yes", "no no no no") sit at 3–4.
- `bigram_run=4`: real 2-word loops (`mempunyai pengaruhan × 4+`) sit
  at 4+ non-overlapping repeats. Natural doubled clauses
  ("I think, I think, I think") sit at 3.
- Non-overlapping bigram counting is important: an overlapping counter
  false-positives on 3× "I think" by triple-counting shifted windows.

### Still-open avenues (not attempted)

- Character-diversity ratio as a second heuristic — not needed given
  the n-gram detector caught every historic loop cleanly.
- Per-chunk `hallucination_silence_threshold` for stray silence-
  triggered hallucinations at chunk boundaries (open since the
  2026-07-29 session).

### Markdown transcript output

Second change this session: switch the final-transcript emitter from
`final_transcript.txt` to `final_transcript.md`. Reviewer workflow is
"read for information + light edits to fix transcription errors" and
needs to work in any editor, so Markdown fits — plain text under the
hood, renders in VS Code / GitHub / Obsidian, edits stay trivial.
Rejected `.docx` (Word-specific, not platform-agnostic) and `.html`
(harder to hand-edit).

Format:

- Header block: `# Interview transcript`, italic source filename and
  model ID, `---` rule.
- Per turn: `**SPEAKER_XX** · MM:SS–MM:SS` on one line, blank line,
  turn body as a paragraph. Timestamps are MM:SS (was raw seconds).
- Flagged turns: blockquote note `> **Review needed** — possible
  transcription loop` between the speaker line and the body, chosen
  over an inline tag so the flag is visually prominent for the
  reviewer.
- `UNCERTAIN_TAG` constant removed (dead — the blockquote text is
  emitted inline in `merge_diarization_and_transcript`).

`main.py` passes `source_name=Path(INPUT_FILE).name` and writes to
`{output_dir}/final_transcript.md`. Regenerated
`output/MCMC_test/final_transcript.md` from cached raw transcript +
diarization; deleted the stale `.txt`.

### Files touched this session

- `transcribe.py` — added `contains_loop`, `LOOP_UNIGRAM_RUN`,
  `LOOP_BIGRAM_RUN`, `_format_time`; rewrote
  `merge_diarization_and_transcript` to emit Markdown with header
  metadata and a blockquote for flagged turns; new `source_name`
  parameter.
- `main.py` — passes `source_name` and writes `.md` instead of `.txt`.
- `output/MCMC_test/final_transcript.md` — regenerated.
- `output/MCMC_test/final_transcript.txt` — deleted (stale).
- `CLAUDE.md` — this log.
