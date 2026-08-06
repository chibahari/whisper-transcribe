# Multilingual Interview Transcription — Project Writeup

This document is a narrative record of the design and iteration behind this
transcription pipeline. It is written for a future engineer picking up the
project — everything a reviewer or successor needs to understand not just
*what* the code does, but *why* each decision was made, what alternatives
were tried, and what remains open.

For usage instructions see `README.md`. For per-session changelogs see
`CLAUDE.md`.

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
- **Local execution.** Runs on Apple Silicon via MLX for Whisper and MPS
  for pyannote — no cloud calls for the audio itself.
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
   at silences, then `mlx_whisper` transcribes each chunk with a domain
   prompt and a hallucination guard. Segment timestamps are re-anchored
   to absolute time before concatenation.
5. **Merge** (`transcribe.merge_diarization_and_transcript`) — assigns
   each Whisper segment to the speaker with maximum time-overlap, applies
   the `[UNCERTAIN — please review]` tag to high-`compression_ratio`
   segments, and writes `final_transcript.txt`.

Two supporting utilities:

- `experiment.py` — config-comparison harness. Runs named
  `mlx_whisper.transcribe` configs against a clip and writes per-config
  outputs plus a `summary.txt` with loop counts and timings.
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

---

## 6. Final design decisions

- **Model: `whisper-large-v3-mlx`.** Turbo trades accuracy for speed here
  in a way that we can't afford — its hallucination pattern is worse than
  large-v3's on this audio.
- **Auto language detection (`language=None`).** With code-switching in
  every other sentence, forcing a single language would systematically
  mistranscribe half the content.
- **VAD pre-chunking on by default.** No parameter to disable it — the
  quality delta is too large and the runtime cost is negative.
- **Rich domain prompt.** The prompt is a code artefact
  (`transcribe.TRANSCRIBE_PROMPT`), not a config knob, because it's
  interview-topic-specific. Different interview topics will need
  different prompts.
- **Uncertainty tag as reviewer safety net.** Not a "fix" — an admission
  that some hallucinations will slip through, and a way to make them
  visible for spot-checking.

---

## 7. Known limitations

- **Non-determinism.** Whisper's decoder gives different output between
  runs on the same audio, especially in confusing regions. A single run
  is not a reliable measurement — reproducibility of the transcript is
  limited.
- **Short in-segment loops.** Per-segment `compression_ratio` cannot
  detect them (see §5.4). VAD reduces but does not eliminate them.
- **Long chunks bypass the VAD benefit.** If a speech region has no
  silence ≥ 0.5 s, the chunk exceeds `max_chunk_s` (48.5 s max seen on
  the sample) and Whisper resumes its internal 30 s windowing. Loops
  inside that chunk can span multiple internal windows.
- **Stray hallucinations at chunk boundaries.** "Terima kasih." on
  chunk-trailing silence occasionally slips through — a single-segment
  event, low CR, invisible to the uncertainty tag.
- **Diarization treats overlap as one speaker.** pyannote's turns are
  disjoint; when two speakers talk over each other the transcript will
  attribute the audio to one of them.
- **Prompt is topic-specific.** A different interview series (e.g. on
  healthcare AI) would need its own vocabulary in `TRANSCRIBE_PROMPT` to
  see equivalent proper-noun accuracy.

## 8. Future work

Ordered by expected impact per unit of implementation effort:

1. **Text-level loop detector for the uncertainty tag.** A simple
   repeated-n-gram check (e.g. any 3-word phrase repeated 3+ times in a
   segment) would catch the class of short in-segment loops that CR
   misses. Cheap.
2. **Forced mid-region VAD split with overlap.** When a speech region
   exceeds `max_chunk_s`, hard-cut at 25 s with a 2 s overlap into the
   next chunk. Fully caps chunk length; the overlap gives Whisper enough
   context to avoid boundary artefacts.
3. **Domain prompt per RUN_NAME.** Move `TRANSCRIBE_PROMPT` out of
   `transcribe.py` into a per-interview file (e.g.
   `data/prompts/{RUN_NAME}.txt`), so the pipeline can serve multiple
   interview topics without code changes.
4. **Speaker labelling.** After a run, let the reviewer supply real names
   for `SPEAKER_00`, `SPEAKER_01`, … and re-emit the transcript. Trivial
   post-pass, big usability win for downstream research.
5. **Multiple-run consensus for confusing regions.** For any chunk that
   emits high-CR output on run N, re-transcribe with a different
   `temperature` seed on run N+1. Keep whichever has the lower CR. Non-cheap
   (2× runtime on affected chunks) but the most principled way to attack
   the non-determinism problem.

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
