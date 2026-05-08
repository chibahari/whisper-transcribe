import mlx_whisper
import json
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch

def diarise(wav_path: str, hf_token: str) -> list[dict]:
    """
    Returns list of {speaker, start, end} dicts.
    Requires a HuggingFace token — model is gated, accept terms at:
    https://hf.co/pyannote/speaker-diarization-3.1
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    print("Pipeline loaded from huggingface.")
    # Use MPS on M4
    pipeline.to(torch.device("mps"))

    with ProgressHook() as hook:
        diarisation = pipeline(wav_path, hook=hook)
    segments = []
    # pyannote.audio 4.0 returns DiarizeOutput; use speaker_diarization attribute
    for turn, speaker in diarisation.speaker_diarization:
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })
    return segments

def transcribe(wav_path: str, output_dir: str = "output") -> dict:
    """
    Transcribe using mlx-whisper with large-v3 for best multilingual accuracy.
    language=None lets Whisper auto-detect per segment.
    """
    Path(output_dir).mkdir(exist_ok=True)

    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language=None,
        word_timestamps=True,
        verbose=True,
        condition_on_previous_text=False,  # ← CHANGE: breaks the feedback loop
        temperature=(0.0, 0.2, 0.4, 0.6), # ← CHANGE: tuple triggers fallback on failure
        no_speech_threshold=0.6,           # ← ADD: skip segments that are likely silence
        compression_ratio_threshold=1.35,  # ← ADD: flag suspiciously repetitive output 
        logprob_threshold=-1.0,            # ← ADD: skip low-confidence segments
    )

    # Save raw output
    out_path = Path(output_dir) / "raw_transcript.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

def merge_diarisation_and_transcript(
    segments: list[dict],      # from diarise()
    transcript: dict,          # from transcribe()
    output_path: str = "output/final_transcript.txt"
) -> str:
    """Align speaker labels with transcript words by timestamp overlap."""

    def get_speaker(start: float, end: float) -> str:
        for seg in segments:
            if seg["start"] <= start and seg["end"] >= end:
                return seg["speaker"]
        return "UNKNOWN"

    lines = []
    current_speaker = None
    current_text = []

    for segment in transcript["segments"]:
        speaker = get_speaker(segment["start"], segment["end"])
        text = segment["text"].strip()

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