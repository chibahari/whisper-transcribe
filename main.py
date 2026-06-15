import os
from dotenv import load_dotenv
from preprocess import convert_to_wav, enhance_audio
from transcribe import transcribe, diarize, merge_diarization_and_transcript

load_dotenv()
INPUT_FILE = os.environ.get("INPUT_FILE")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: Huggingface token not found.")

if not INPUT_FILE:
    print("ERROR: Input file not found.")

# 1. Convert & enhance
wav_path = convert_to_wav(INPUT_FILE, "tmp/1_raw.wav")
clean_wav = enhance_audio(wav_path, "data/intermediate/2_clean.wav")

# 2. diarize
speaker_segments = diarize(
    clean_wav,
    HF_TOKEN,
    min_speakers=2,
    max_speakers=3
)

# 3. Transcribe
transcript = transcribe(clean_wav)

# 4. Merge
final = merge_diarization_and_transcript(speaker_segments, transcript)
print("Done! Transcript saved to output/final_transcript.txt")