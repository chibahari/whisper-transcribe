import mlx_whisper
import json
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

SAMPLE_RATE = 16000

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
        token=hf_token
    )
    print("Pipeline loaded from huggingface.")
    # Use MPS on apple silicon
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
        diarization = pipeline(
            wav_path,
            hook=hook,
            **speaker_kwargs,
        )
    segments = []
    # pyannote.audio 4.0 returns DiarizeOutput; use speaker_diarization attribute
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })
    return segments

TRANSCRIBE_PROMPT = (
    "Temu bual penyelidikan tentang pelaporan insiden AI di Malaysia. "
    "Interviewees are from MCMC, NSC, NACSA, CSM, BNM. "
    "Discussion covers CMA 1998 Section 263, cybersecurity incidents, "
    "data breach, ransomware, phishing, SMB port 445, telco licensees, "
    "postal operators, courier services, SII entities, aduan (complaints), "
    "penyedia rangkaian, broadcasting, incident reporting. "
    "This interview contains both Malay and English with frequent code switching."
)


def vad_chunks(
    wav_path: str,
    max_chunk_s: float = 30.0,
    min_chunk_s: float = 5.0,
    min_silence_s: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Split audio at natural silences using Silero VAD. Returns a list of
    (start_s, end_s) chunks. Each chunk ends at a silence >= min_silence_s
    unless capped by max_chunk_s. This bounds how far a Whisper decoder loop
    can propagate within a single call.
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


def transcribe(wav_path: str, output_dir: str = "output") -> dict:
    """
    Transcribe using mlx-whisper with large-v3 for best multilingual accuracy.
    Audio is pre-chunked on silences via Silero VAD, then each chunk is
    transcribed independently. This bounds loops to a single chunk instead
    of letting them accumulate across Whisper's arbitrary 30 s windows.
    language=None lets Whisper auto-detect per chunk.
    """
    Path(output_dir).mkdir(exist_ok=True)

    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    chunks = vad_chunks(wav_path)
    print(f"VAD produced {len(chunks)} chunks")

    all_segments: list[dict] = []
    detected_language: str | None = None
    for chunk_start, chunk_end in tqdm(chunks, desc="Transcribing chunks"):
        start_sample = int(chunk_start * SAMPLE_RATE)
        end_sample = int(chunk_end * SAMPLE_RATE)
        chunk_audio = np.ascontiguousarray(audio[start_sample:end_sample])

        result = mlx_whisper.transcribe(
            chunk_audio,
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            language=None,
            word_timestamps=True,
            verbose=False,
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4, 0.6),
            no_speech_threshold=0.6,
            compression_ratio_threshold=1.35,
            logprob_threshold=-1.0,
            hallucination_silence_threshold=2.0,
            task="transcribe",
            initial_prompt=TRANSCRIBE_PROMPT,
        )
        detected_language = detected_language or result.get("language")

        for seg in result["segments"]:
            seg["start"] += chunk_start
            seg["end"] += chunk_start
            for word in seg.get("words") or []:
                word["start"] += chunk_start
                word["end"] += chunk_start
            all_segments.append(seg)

    combined = {
        "text": " ".join(s["text"].strip() for s in all_segments),
        "segments": all_segments,
        "language": detected_language,
    }

    out_path = Path(output_dir) / "raw_transcript.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    return combined

UNCERTAIN_CR_THRESHOLD = 2.4
UNCERTAIN_TAG = "[UNCERTAIN — please review]"


def merge_diarization_and_transcript(
    segments: list[dict],      # from diarize()
    transcript: dict,          # from transcribe()
    output_path: str = "output/final_transcript.txt"
) -> str:
    """Align speaker labels with transcript words by timestamp overlap."""

    def get_speaker(start: float, end: float) -> str:
        # Max-overlap: pick the diarization turn that shares the most time
        # with this Whisper segment. Strict containment would leave every
        # segment that straddles a turn boundary as UNKNOWN.
        best_overlap = 0.0
        best_speaker = "UNKNOWN"
        for seg in segments:
            overlap = min(end, seg["end"]) - max(start, seg["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]
        return best_speaker

    lines = []
    current_speaker = None
    current_text = []

    for segment in transcript["segments"]:
        speaker = get_speaker(segment["start"], segment["end"])
        text = segment["text"].strip()
        if segment.get("compression_ratio", 0) > UNCERTAIN_CR_THRESHOLD:
            text = f"{UNCERTAIN_TAG} {text}"

        if speaker != current_speaker:
            if current_text:
                lines.append(f"[{current_speaker}]\n" + " ".join(current_text))
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)

    if current_text:
        lines.append(f"[{current_speaker}]\n" + " ".join(current_text))

    output = "\n\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    return output