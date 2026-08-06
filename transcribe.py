"""
Transcription pipeline for Malaysian AI incident-reporting interviews.

Uses the mesolitica/Malaysian-whisper-large-v3-turbo-v3 fine-tune via HF
transformers on MPS. Audio is pre-chunked on silences with Silero VAD; each
chunk gets a per-chunk language probe (with an English bias) so that short
English utterances don't get mis-routed to Malay by mesolitica's Malay-
biased language head.

Diarization uses pyannote/speaker-diarization-3.1. The merge step aligns
speaker labels to transcript segments by max time-overlap, then joins
consecutive same-speaker segments into single turns in the final output.
"""

import json
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
from transformers.models.whisper import tokenization_whisper

SAMPLE_RATE = 16000
MODEL_ID = "mesolitica/Malaysian-whisper-large-v3-turbo-v3"

# Mesolitica registers a custom `transcribeprecise` task token. Keeping the
# tokenizer's known-task list in sync avoids errors when the model loads.
if "transcribeprecise" not in getattr(tokenization_whisper, "TASK_IDS", []):
    tokenization_whisper.TASK_IDS = list(
        getattr(tokenization_whisper, "TASK_IDS", ["translate", "transcribe"])
    ) + ["transcribeprecise"]


def diarize(
    wav_path: str,
    hf_token: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict]:
    """
    Returns list of {speaker, start, end} dicts.
    Requires a HuggingFace token — model is gated, accept terms at:
    https://hf.co/pyannote/speaker-diarization-3.1

    If `num_speakers` is provided, pyannote clusters to exactly that count
    (strongest constraint — use when the interview roster is known).
    Otherwise `min_speakers` / `max_speakers` bound the search.
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    print("Pyannote pipeline loaded.")
    pipeline.to(torch.device("mps"))

    speaker_kwargs: dict = {}
    if num_speakers is not None:
        speaker_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            speaker_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            speaker_kwargs["max_speakers"] = max_speakers

    with ProgressHook() as hook:
        diarization = pipeline(wav_path, hook=hook, **speaker_kwargs)

    segments = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
        })
    return segments


def vad_chunks(
    wav_path: str,
    max_chunk_s: float = 30.0,
    min_chunk_s: float = 5.0,
    min_silence_s: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Split audio at natural silences using Silero VAD. Returns a list of
    (start_s, end_s) chunks. Each chunk ends at a silence >= min_silence_s
    unless capped by max_chunk_s. Bounds how far any decoder loop can
    propagate within a single Whisper call.
    """
    model = load_silero_vad()
    wav = read_audio(wav_path, sampling_rate=SAMPLE_RATE)
    speech = get_speech_timestamps(
        wav, model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        min_silence_duration_ms=int(min_silence_s * 1000),
    )
    if not speech:
        return []

    chunks: list[tuple[float, float]] = []
    chunk_start = speech[0]["start"]
    chunk_end = speech[0]["end"]
    for i, region in enumerate(speech):
        chunk_end = region["end"]
        length = chunk_end - chunk_start
        next_gap = (speech[i + 1]["start"] - region["end"]) if i + 1 < len(speech) else float("inf")
        should_close = length >= max_chunk_s or (
            next_gap >= min_silence_s and length >= min_chunk_s
        )
        if should_close:
            chunks.append((chunk_start, chunk_end))
            if i + 1 < len(speech):
                chunk_start = speech[i + 1]["start"]
    if not chunks or chunks[-1][1] != chunk_end:
        chunks.append((chunk_start, chunk_end))
    return chunks


_pipeline_cache = None


def _get_pipeline():
    """Load the mesolitica pipeline once per process; MPS + bfloat16."""
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    print(f"Loading {MODEL_ID} on {device} ({dtype})…")
    _pipeline_cache = hf_pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        torch_dtype=dtype,
        device=device,
    )
    return _pipeline_cache


def _detect_chunk_language(
    pipe,
    chunk_audio: np.ndarray,
    en_bias: float = 0.3,
    candidate_langs: tuple[str, ...] = ("en", "ms", "id", "zh", "ta"),
) -> str:
    """Read language-token logits at the first decoder position and pick a
    language. If p(en) > en_bias, prefer English — this counters the model's
    Malay bias on short English utterances (interviewer questions, opening
    lines) that otherwise get decoded as formal Malay."""
    inputs = pipe.feature_extractor(
        chunk_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    dtype = next(pipe.model.parameters()).dtype
    input_features = inputs.input_features.to(pipe.model.device, dtype)

    decoder_start = pipe.model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start]], device=pipe.model.device)
    with torch.no_grad():
        outputs = pipe.model(input_features, decoder_input_ids=decoder_input_ids)
    lang_probs_all = torch.softmax(outputs.logits[0, -1].float(), dim=-1)

    tok = pipe.tokenizer
    probs: dict[str, float] = {}
    for code in candidate_langs:
        tok_id = tok.convert_tokens_to_ids(f"<|{code}|>")
        if tok_id is not None and tok_id != tok.unk_token_id:
            probs[code] = float(lang_probs_all[tok_id])

    if not probs:
        return "en"
    if probs.get("en", 0.0) > en_bias:
        return "en"
    return max(probs, key=probs.get)


def transcribe(
    wav_path: str,
    output_dir: str = "output",
    en_bias: float = 0.3,
) -> dict:
    """VAD-chunk audio, detect language per chunk (with English bias), then
    transcribe each chunk with the detected language forced. Returns a dict
    with `text`, `segments` (each with start/end/text/_lang), and `language`
    (always None — language varies per chunk)."""
    Path(output_dir).mkdir(exist_ok=True)
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    chunks = vad_chunks(wav_path)
    print(f"VAD produced {len(chunks)} chunks")

    pipe = _get_pipeline()

    all_segments: list[dict] = []
    lang_counts: dict[str, int] = {}
    for chunk_start, chunk_end in tqdm(chunks, desc="Transcribing chunks"):
        s0, s1 = int(chunk_start * SAMPLE_RATE), int(chunk_end * SAMPLE_RATE)
        chunk_audio = np.ascontiguousarray(audio[s0:s1])

        lang = _detect_chunk_language(pipe, chunk_audio, en_bias=en_bias)
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        out = pipe(
            {"array": chunk_audio, "sampling_rate": SAMPLE_RATE},
            return_timestamps=True,
            generate_kwargs={"task": "transcribe", "language": lang},
        )
        for ch in out.get("chunks", []) or []:
            ts = ch.get("timestamp") or (None, None)
            cs, ce = ts
            if cs is None:
                cs = 0.0
            if ce is None:
                ce = cs + max(1.0, len(ch.get("text", "")) * 0.05)
            all_segments.append({
                "id": len(all_segments),
                "start": float(cs) + chunk_start,
                "end": float(ce) + chunk_start,
                "text": ch.get("text", ""),
                "_lang": lang,
            })

    print(f"Language distribution across VAD chunks: {lang_counts}")

    combined = {
        "text": " ".join(s["text"].strip() for s in all_segments),
        "segments": all_segments,
        "language": None,
        "_lang_counts": lang_counts,
    }

    out_path = Path(output_dir) / "raw_transcript.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    return combined


def merge_diarization_and_transcript(
    segments: list[dict],
    transcript: dict,
    output_path: str = "output/final_transcript.txt",
) -> str:
    """Align speaker labels to transcript segments by max time-overlap, then
    concatenate consecutive same-speaker segments into a single turn."""

    def get_speaker(start: float, end: float) -> str:
        best_overlap = 0.0
        best_speaker = "UNKNOWN"
        for seg in segments:
            overlap = min(end, seg["end"]) - max(start, seg["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]
        return best_speaker

    turns = []
    cur_speaker: str | None = None
    cur_text: list[str] = []
    cur_start: float | None = None
    cur_end: float | None = None

    for segment in transcript["segments"]:
        speaker = get_speaker(segment["start"], segment["end"])
        text = segment["text"].strip()
        if not text:
            continue
        if speaker != cur_speaker:
            if cur_text:
                turns.append((cur_speaker, cur_start, cur_end, " ".join(cur_text)))
            cur_speaker = speaker
            cur_text = [text]
            cur_start = segment["start"]
            cur_end = segment["end"]
        else:
            cur_text.append(text)
            cur_end = segment["end"]
    if cur_text:
        turns.append((cur_speaker, cur_start, cur_end, " ".join(cur_text)))

    lines = []
    for speaker, start, end, text in turns:
        lines.append(f"[{speaker}] ({start:.1f}s–{end:.1f}s)\n{text}")

    output = "\n\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    return output
