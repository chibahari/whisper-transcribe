import mlx_whisper
import json
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch

def diarize(
    wav_path: str,
    hf_token: str,
    min_speakers: int,
    max_speakers: int) -> list[dict]:
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
    # Use MPS on apple silicon
    pipeline.to(torch.device("mps"))

    with ProgressHook() as hook:
        diarization = pipeline(
            wav_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hook=hook
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
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6),
        no_speech_threshold=0.6,
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
        # Skip silent runs longer than this when word timestamps suggest a
        # hallucination — the single biggest fix for the repetition loops.
        hallucination_silence_threshold=2.0,
        task="transcribe",
        initial_prompt=TRANSCRIBE_PROMPT,
    )

    # Save raw output
    out_path = Path(output_dir) / "raw_transcript.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

def merge_diarization_and_transcript(
    segments: list[dict],      # from diarize()
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