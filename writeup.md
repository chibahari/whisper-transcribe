# Multilingual Interview Transcription — Project Writeup

This document is a narrative record of the design and iteration behind this
transcription pipeline. It is written for a future engineer picking up the
project — everything a reviewer or successor needs to understand not just
*what* the code does, but *why* each decision was made, what alternatives
were tried, and what remains open.

For usage instructions see `README.md`. For per-session changelogs see
`CLAUDE.md`.

> **Status (2026-08-06).** The current pipeline transcribes via
> `mesolitica/Malaysian-whisper-large-v3-turbo-v3` (a Malaysian fine-tune
> of Whisper) with VAD pre-chunking and per-chunk language routing — not
> `mlx_whisper`. §3 and §6 describe the current design. §5.7 documents
> the migration. §5.1–5.6 remain as the historical iteration that
> preceded it — the constraints, guards, and lessons are still relevant
> even though the model changed.

---

## 1. Objective

Build a local, reproducible transcription pipeline for research interviews
about AI incident reporting in Malaysia. Output must be accurate enough
that a human researcher can read the transcript, follow the conversation,
and cite it — not perfect, but not misleading.

## 2. Requirements and constraints

- **Duration.** Interviews are 45 min – 3 h.
- **Speakers.** 2–6 per interview, needing individual identification.
- **Languages.** English and Malay, with frequent code-switching mid-sentence
  and mixed vocabulary within utterances (e.g. "So, kita ambil telco, kita
  ada meksis, telekom dan lain-lain").
- **Audio quality.** Not guaranteed clean; realistic recordings with
  ambient noise, uneven volume, and interviewee hesitation.
- **Local execution.** Runs on Apple Silicon via MPS for both Whisper and
  pyannote — no cloud calls for the audio itself.
- **Domain vocabulary.** Interviewees are from Malaysian regulators
  (MCMC, NSC, NACSA, CSM, BNM) and discuss specific legislation
  (CMA 1998), technical concepts (SMB port 445, ransomware, phishing),
  and Malay-language jargon (aduan, penyedia rangkaian).

## 3. System design

The pipeline lives in three files and runs sequentially from `main.py`:

1. **Convert** (`preprocess.convert_to_wav`) — ffmpeg re-encodes the input
   to 16 kHz mono 16-bit PCM WAV, Whisper's native format.
2. **Enhance** (`preprocess.enhance_audio`) — pydub loudness
   normalisation, then `noisereduce` spectral gating (0.5 s of the start
   as a noise profile, `prop_decrease=0.8`).
3. **Diarize** (`transcribe.diarize`) — `pyannote/speaker-diarization-3.1`
   on MPS, constrained by `NUM_SPEAKERS` (exact) or `MIN_SPEAKERS` /
   `MAX_SPEAKERS` (range). Returns `{speaker, start, end}` turns.
4. **Transcribe** (`transcribe.transcribe`) — Silero VAD chunks the audio
   at silences. For each chunk we run a **one-step language probe** that
   reads the language-token logits at the first decoder position, apply
   an **English bias** (force `en` when `p(en) > 0.3`, else the top
   language), then transcribe the chunk with that language forced. The
   model is `mesolitica/Malaysian-whisper-large-v3-turbo-v3` via
   HuggingFace transformers on MPS (bfloat16). Segment timestamps are
   re-anchored to absolute time before concatenation.
5. **Merge** (`transcribe.merge_diarization_and_transcript`) — assigns
   each transcript segment to the speaker with maximum time-overlap, then
   concatenates consecutive same-speaker segments into single turns.
   Writes `final_transcript.txt`.

Two supporting utilities:

- `experiment.py` — multi-backend config-comparison harness. Supports
  `mlx_whisper` (historical baselines), `hf_transformers` (mesolitica via
  pipeline), and `hf_transformers_per_chunk_lang` (mesolitica + VAD +
  per-chunk language detect — mirrors the production `transcribe.py`).
  See `output/experiments/README.md` for the full config catalogue.
- `output/{RUN_NAME}/diarization.json` — cached diarization output so
  merge-only iterations skip the ~2 min diarize step.

---

## 4. Challenges encountered

Four problems dominated the work:

**Repetition loops.** Whisper's decoder sometimes gets stuck emitting the
same phrase over and over. On the sample the worst case was 27 s of
"aduan mempunyai pengaruh secara sekaligus?" repeated ~10 times
(`compression_ratio=8.51`). Loops eat real content — during those 27 s the
interviewee was actually explaining how the MCMC complaint portal
segregates cybersecurity reports, and none of it was captured.

**YouTube-style hallucinations.** During silences Whisper occasionally
emits stock phrases from its training data — "Terima kasih kerana
menonton" ("thank you for watching") and "Jangan lupa untuk beri komentar
di bawah ini" ("don't forget to comment below"). These are visually
plausible Malay but obviously wrong in context.

**Language flipping and wrong proper nouns.** With `language=None`
(auto-detect per 30 s window), Whisper sometimes translated English
utterances into Malay when the surrounding audio was Malay-heavy. Proper
nouns landed phonetically wrong ("Naksa" for the agency NACSA).

**Diarization / transcription boundary alignment.** pyannote produces
turn boundaries down to ~10 ms precision; Whisper produces segments
aligned to internal decoder decisions. The two rarely coincide, so a
naive merge strategy loses speaker labels wherever the boundaries
disagree.

---

## 5. What we tried

### 5.1 Whisper decoder configuration

The most productive early iteration was tuning
`mlx_whisper.transcribe` kwargs against a 6-min clip that contained a
code-switching passage and a known loop region. All configs share
`large-v3-mlx`, `word_timestamps=True`, and
`condition_on_previous_text=False` (which itself was an early fix to
stop cross-window feedback loops).

| Config | Change | Cross-segment loops | YouTube hallucinations |
|---|---|---|---|
| A — baseline | current settings | 3 catastrophic regions | yes |
| B — hallucination guard | `hallucination_silence_threshold=2.0` | 0 | 0 |
| C — rich prompt | B + expanded `initial_prompt` | 0 | 0 |
| D — turbo model | C but `whisper-large-v3-turbo` | 0 | spurious "Use" tokens |
| E — aggressive temp ladder | temps `(0.0, 0.4, 0.8, 1.0)` | 5 (worse) | 0 |
| F — extra fallback | 5-step ladder ending at 0.8 | 1 (worse than C) | 0 |
| G — tight CR | F + `compression_ratio_threshold=1.2` | 2 (worse than C) | 0 |

**Kept: config C.** Two ingredients did the work.

- **`hallucination_silence_threshold=2.0`.** When Whisper's word-level
  timestamps suggest an implausibly long silent stretch is being
  transcribed, this parameter tells the decoder to skip it. Single biggest
  fix — it killed the 55-segment blank-loop region in the baseline.
- **The rich `initial_prompt`.** A ~60-word block naming the interview
  topic, the organisations (MCMC, NSC, NACSA, CSM, BNM), the relevant
  legislation (CMA 1998), and the technical + Malay vocabulary. Whisper
  conditions its decoder on this text, which anchors proper-noun spelling
  and reduces language flipping.

**Traps that looked promising but weren't.**

- *Turbo model.* Faster, but introduced its own hallucination pattern
  (spurious "Use" tokens sprinkled through the output).
- *Wider temperature ladder.* Whisper keeps the *last* attempt when the
  fallback ladder is exhausted. Raising the top temperature just meant
  the "kept" output was drawn from a hotter distribution — more looping
  content, not less.
- *Tighter `compression_ratio_threshold`.* Only helps if fallback attempts
  actually escape the loop. On stubborn regions they don't, so tightening
  just triggers fallback more often without changing the kept output.

### 5.2 Configurable speaker count

Initial code hard-coded `min_speakers=2, max_speakers=3`, which
under-covered the stated 2–6 requirement. Refactored so `.env` supplies
either `NUM_SPEAKERS` (exact — strongest pyannote constraint, use when
the interview roster is known) or `MIN_SPEAKERS` / `MAX_SPEAKERS` (range,
when unsure). `transcribe.diarize()` forwards only whichever kwargs are
supplied to pyannote, avoiding the "both specified" edge case.

Impact on the sample transcript was neutral — pyannote returned 2
speakers regardless of ceiling, because the sample really is a 1-on-1
interview. The change was made to correctly handle future 3–6 speaker
interviews.

### 5.3 Merge algorithm bug fix

The original `get_speaker(start, end)` used **strict containment**:

```python
if seg["start"] <= start and seg["end"] >= end:
    return seg["speaker"]
return "UNKNOWN"
```

This labelled ~47 % of Whisper segments as UNKNOWN because Whisper
segments frequently straddle pyannote turn boundaries. Rewrote to
**maximum time-overlap** — pick the diarization turn that shares the most
time with the Whisper segment. UNKNOWN dropped to ~3 % on the sample,
and the reconstructed dialog structure (interviewer question → interviewee
answer) became legible for the first time.

### 5.4 Uncertainty flagging

> Removed in §5.7 when we switched to mesolitica's HF pipeline, which
> doesn't expose per-segment `compression_ratio`. Documented here for
> the pattern; a text-level replacement is listed in §8.1.

Any Whisper segment with `compression_ratio > 2.4` is prefixed with
`[UNCERTAIN — please review]` in the final transcript. The threshold was
first set to 6.0 (catches only the pathological tier), then lowered to
2.4 to align with Whisper's own default `compression_ratio_threshold` for
fallback triggering.

In practice the CR distribution on the sample is bimodal — real segments
sit below 2.2 and catastrophic loops above 8 — so lowering the threshold
didn't change which segments were tagged. Kept 2.4 as the more
principled default in case future recordings hit the middle range.

**Limitation.** Per-segment `compression_ratio` cannot detect short
in-segment loops (e.g. "mempunyai pengaruhan mempunyai pengaruhan…"
inside a 4 s segment, `cr=2.17`). The text is highly repetitive but so
short that gzip overhead dominates, keeping CR low. This is a fundamental
limitation of the metric, not a threshold-tuning problem.

### 5.5 Noise reduction tuning — rejected

Hypothesis: `noisereduce` at `prop_decrease=0.8` might be over-suppressing
speaker-hesitation cues that trigger the internal loops. Tried
`prop_decrease=0.5` on the full sample.

Result was clearly worse: peak `cr=17.03` (vs 8.51 baseline) and three
new loop regions appeared at 213 s, 577–605 s, and 1212 s — the exact
stretches where the 0.8 baseline transcribed cleanly (data breach /
ransomware, SMB port 445 discussion). More noise, not less, is what
protects Whisper on this audio. Reverted.

### 5.6 VAD pre-chunking — the biggest win

Reasoning: Whisper's own 30 s windows are arbitrary, and once the decoder
starts looping it will happily emit the same phrase across multiple
windows. If we pre-split the audio at natural silences, any decoder loop
is bounded to a single chunk.

Added Silero VAD (`min_silence=0.5 s`, `max_chunk=30 s`, `min_chunk=5 s`).
`transcribe()` now loads the full audio as a numpy array, iterates over
chunks, calls `mlx_whisper.transcribe` on each chunk (which accepts
`ndarray` directly), and offsets returned timestamps by the chunk start
before concatenating.

**Result on the sample.**

| Metric | Baseline (no VAD) | VAD chunking |
|---|---|---|
| Segments with `cr > 6.0` | 10 (244–271 s cross-segment loop) | 1 (1484 s single-segment loop) |
| Segments with `cr > 2.4` | 10 | 1 |
| Content recovered in former loop region | none | "From aduan perspective, aduan is safe." + follow-up |
| Whisper runtime | ~20 min | ~14 min |

The 244–271 s loop is fully eliminated — VAD split that region into three
chunks (228–251, 252–270, 273–280 s), so the loop had nowhere to
propagate. The remaining catastrophic loop at 1484 s is a 24 s
"iaitu iaitu iaitu…" run inside a single chunk that had no
VAD-detectable silence — the uncertainty tag catches it cleanly
(`cr=21.55`).

The runtime drop was an unexpected bonus. Shorter chunks mean less
wasted decoding on padding at the tail of Whisper's 30 s windows.

### 5.7 Model swap to mesolitica + per-chunk language routing (2026-08-06)

The reviewer tagged ~5 min of `final_transcript.txt` and found the
dominant residual failure was **English speech decoded as gibberish
Malay**, especially in the first 3 min of the recording where the
interviewer opens with short English questions. Also occasional
"hallucinated Malay" in silence pockets. The uncertainty tag caught
almost none of it because these weren't loops — they were confidently
wrong outputs with plausible `compression_ratio`.

**Root cause.** `whisper-large-v3` on this material was doing per-chunk
language auto-detection, and short English-only chunks (interviewer
questions, opening lines) were being classified as Malay. Once the
language head landed on Malay, the decoder dutifully emitted Malay
tokens for the English acoustics, producing formal Bahasa that had no
resemblance to what was said.

**Model swap.** Adopted `mesolitica/Malaysian-whisper-large-v3-turbo-v3`
— a fine-tune of `whisper-large-v3-turbo` trained explicitly on Malaysian
speech with Manglish (Malay-English) code-switching, plus Mandarin and
Tamil. No MLX conversion published, so runs via HF transformers + MPS
with bfloat16. Fine-tuned model handled code-switching within a single
utterance natively (e.g. "if let's say macam licensee ada incident they
have to report to us") which the base model kept splitting into two
translations.

**Iteration on the mesolitica pipeline** (details in
`output/experiments/README.md`):

| Config | Change | Result |
|---|---|---|
| `M_mesolitica_turbo` | first attempt, `stride_length_s=5` | Fixed loops and most mistranslation; **introduced** intra-turn duplication (same clause transcribed once in English and once in formal Malay from chunk overlaps). |
| `M2_stride0` | `stride_length_s=0` | Duplication gone (chars −15%). Opening 0-13s and interviewer stretches at 397-430s **still all-Malay**. |
| `M3_lang_en` | force `language="en"` globally | Fixed interviewer stretches, over-anglicized real Manglish content ("a series that is a if it is a series for telecommunications"). Rejected. |
| `M4_detect_lang` | **VAD + per-chunk language probe + `en_bias=0.3`** | Fixed all three previously-flagged regions. Language split ~52/48 en/ms on the sample. **Adopted.** |

**Why the English bias.** The failure mode is asymmetric — mesolitica's
Malay-biased language head loses English chunks, but rarely the
reverse. Setting a threshold "prefer English if `p(en) > 0.3` even when
Malay's raw probability is higher" biases against the observed failure
without regressing on Malay-dominant chunks. If future recordings shift
toward more Malay-dominant material, lower the threshold.

**What we lost.** Mesolitica pipeline output doesn't expose per-segment
`compression_ratio`, so the `[UNCERTAIN — please review]` tag from §5.4
is no longer emitted. M4 produced 0 loops on the 26 min sample, so this
hasn't bitten yet — but if loops resurface, a text-level detector
(repeated n-gram or low character-diversity) would replace it.

**Speaker-turn merging.** Mesolitica's pipeline emits word-level
segments (~800 for the 26 min sample). Without merging, the final
transcript would fragment into a wall of single-word turns.
`merge_diarization_and_transcript()` now concatenates consecutive
same-speaker segments into single turns (65 turns on the sample), which
reads as natural dialog.

---

## 6. Final design decisions

- **Model: `mesolitica/Malaysian-whisper-large-v3-turbo-v3`.** A
  Malaysian fine-tune trained on Manglish code-switching. Base Whisper
  and its own turbo variant were both worse on interviewer English
  stretches (see §5.7). No MLX conversion published, so we accept the
  transformers + MPS runtime cost — quality was the priority.
- **Per-chunk language detection with English bias
  (`en_bias=0.3`).** VAD pre-chunk the audio, probe the language-token
  logits, force the detected language during decode. Bypasses the
  pipeline auto-detect that biased short English chunks toward Malay.
- **VAD pre-chunking on by default.** No parameter to disable it — the
  quality delta is too large and the runtime cost is negligible. Still
  serves as loop insurance even though mesolitica loops less than the
  MLX baseline.
- **Speaker-turn merging in the final transcript.** Mesolitica's
  pipeline output is word-level; without merging, `final_transcript.txt`
  fragments into a wall of single-word entries. Consecutive same-speaker
  segments become one turn.
- **No `initial_prompt`.** The HF pipeline API doesn't take one the same
  way `mlx_whisper` did, and the fine-tuned model already handles the
  domain vocabulary. If proper-noun accuracy regresses on new material,
  revisit via `generate_kwargs`.

---

## 7. Known limitations

- **No automated loop detection.** Mesolitica pipeline output doesn't
  expose `compression_ratio`, so the `[UNCERTAIN]` tag from §5.4 is
  gone. M4 produced 0 loops on the 26 min sample and the reviewer saw
  none in the merged output, but if loops surface on longer/harder
  material there is currently no automated flag. A text-level detector
  (§8.1) would restore this.
- **`en_bias` is a single global knob.** Chosen for interviews where
  English is the more commonly mis-routed language. If the source
  material shifts (e.g. mostly Malay interviews with the interviewer
  code-switching to English rarely), the threshold likely needs to move
  in the other direction. Currently no per-speaker or per-recording
  override.
- **Non-determinism.** Whisper's decoder gives different output between
  runs on the same audio. Reproducibility of any specific transcription
  is limited — treat the transcript as one sample from a distribution,
  not a fixed truth.
- **Long chunks bypass some VAD benefit.** If a speech region has no
  silence ≥ 0.5 s, the chunk exceeds `max_chunk_s` (48.5 s max seen on
  the sample) and the model resumes its internal 30 s windowing inside
  the chunk. Loops inside such a chunk can span multiple internal
  windows.
- **Diarization treats overlap as one speaker.** pyannote's turns are
  disjoint; when two speakers talk over each other the transcript will
  attribute the audio to one of them.
- **Occasional proper-noun mishearings.** The end-to-end run on the
  MCMC sample produced "MCM Zero" (probably "MCMC role"),
  "Naval Security" (probably "Network Security"), "TEKO" (probably
  "telco"). Small enough to spot-fix in review, but a consistent
  pattern to watch for on new material.

## 8. Future work

Ordered by expected impact per unit of implementation effort:

1. **Text-level loop detector.** A simple repeated-n-gram check (e.g. any
   3-word phrase repeated 3+ times in a segment) would give us back the
   automated `[UNCERTAIN]` flag that we lost when we moved to
   mesolitica's pipeline output. Cheap.
2. **Forced mid-region VAD split with overlap.** When a speech region
   exceeds `max_chunk_s`, hard-cut at 25 s with a 2 s overlap into the
   next chunk. Fully caps chunk length; the overlap gives the decoder
   enough context to avoid boundary artefacts.
3. **Speaker labelling.** After a run, let the reviewer supply real names
   for `SPEAKER_00`, `SPEAKER_01`, … and re-emit the transcript. Trivial
   post-pass, big usability win for downstream research.
4. **Per-recording `en_bias` config.** Allow `.env` (or a per-run YAML)
   to override `en_bias`. Interviews vary in language dominance; a
   single global knob will not fit all future material.
5. **Gemini or GPT-4o audio backend as A/B.** Cloud multilingual ASR
   models handle Malay-English code-switching in ways worth benchmarking
   against mesolitica on the current sample. Deferred while local
   quality is acceptable.
6. **Multiple-run consensus for confusing regions.** For any chunk with
   suspicious output (short internal loop, or the text-level detector
   fires), re-transcribe with a different temperature seed on run N+1.
   Keep whichever the detector prefers. Attacks the non-determinism
   problem directly at ~1.1x average cost.

## 9. Lessons learned

- **Whisper parameters interact non-obviously.** Widening the temperature
  fallback ladder or tightening the CR threshold both *sound* like they
  should reduce loops, and both make things worse for a subtle reason
  (Whisper keeps the last fallback attempt, so a hotter last-attempt
  produces worse "kept" output).
- **Preprocessing that makes the audio "cleaner" isn't always better.**
  Lighter noise reduction produced worse transcripts, because the residual
  noise was acting as an anchor for the decoder in confusing regions.
- **Cheap safety nets beat clever guards.** The single most useful
  addition wasn't a decoder tweak — it was the `[UNCERTAIN]` tag, which
  turns "invisible hallucination" into "visibly flagged segment a human
  can double-check". Reviewer trust matters more than raw accuracy.
- **Test the merge, not just the transcription.** A silent merge bug can
  hide the effect of a diarization improvement completely — we would
  have concluded pyannote was broken if the max-overlap fix hadn't
  landed first.
- **Cache the expensive stages.** Diarization takes ~2 min and doesn't
  change between runs on the same clean WAV. Caching it made
  merge-parameter iteration go from "20 min per attempt" to "10 s per
  attempt", which changed how much iteration was feasible.
- **A domain-specific fine-tune beats parameter tuning.** Weeks of
  iterating on Whisper decoder parameters produced meaningful but
  bounded improvements. Swapping to the Malaysian fine-tune (§5.7)
  produced a bigger quality jump than every prior decoder-config change
  combined. Reach for a specialised model before assuming the general
  one just needs more tuning.
- **Language mis-detection is asymmetric.** The failure mode on this
  material was "English forced into Malay", not the reverse — so a
  cheap unidirectional bias (`en_bias`) fixed most of it without
  regressing the working direction. Symmetric fixes (dual-decode both
  languages, pick better logprob) would have been 2× the cost with no
  additional quality. Diagnose the *direction* of the failure before
  fixing.
- **Ask about "unworkable" before shipping.** Two rounds of "looks
  much better" ended with the reviewer saying the output wasn't usable.
  Post-review calibration is essential — statistical improvements on
  loops and duplication counts don't translate directly to reviewer
  trust if a single visible mistranslation region remains.
