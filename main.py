import os
from pathlib import Path
from dotenv import load_dotenv
from preprocess import convert_to_wav, enhance_audio
from transcribe import transcribe, diarize, merge_diarization_and_transcript

load_dotenv()
INPUT_FILE = os.environ.get("INPUT_FILE")
HF_TOKEN = os.environ.get("HF_TOKEN")
RUN_NAME = os.environ.get("RUN_NAME")

if not HF_TOKEN:
    print("ERROR: Huggingface token not found.")

if not INPUT_FILE:
    print("ERROR: Input file not found.")

if not RUN_NAME:
    print("ERROR: RUN_NAME not set in .env")

output_dir = Path(f"output/{RUN_NAME}")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Convert & enhance
wav_path = convert_to_wav(INPUT_FILE, f"tmp/{RUN_NAME}_raw.wav")
clean_wav = enhance_audio(wav_path, str(output_dir / "clean.wav"), normalised_path=f"tmp/{RUN_NAME}_normalised.wav")

# 2. diarize
speaker_segments = diarize(
    clean_wav,
    HF_TOKEN,
    min_speakers=2,
    max_speakers=3
)

# 3. Transcribe
transcript = transcribe(clean_wav, output_dir=str(output_dir))

# 4. Merge
final = merge_diarization_and_transcript(speaker_segments, transcript, output_path=str(output_dir / "final_transcript.txt"))
print(f"Done! Transcript saved to {output_dir}/final_transcript.txt")
