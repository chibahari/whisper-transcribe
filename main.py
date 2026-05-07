import os
from dotenv import load_dotenv
from preprocess import convert_to_wav, enhance_audio
from transcribe import transcribe, diarise, merge_diarisation_and_transcript

load_dotenv()
INPUT_FILE = "data/raw/AIIR sample - EN.mp3"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: Hugginface token not found.")

# 1. Convert & enhance
wav_path = convert_to_wav(INPUT_FILE, "tmp/1_raw.wav")
clean_wav = enhance_audio(wav_path, "data/intermediate/2_clean.wav")

# 2. Diarise
speaker_segments = diarise(clean_wav, HF_TOKEN)

# 3. Transcribe
transcript = transcribe(clean_wav)

# 4. Merge
final = merge_diarisation_and_transcript(speaker_segments, transcript)
print("Done! Transcript saved to output/final_transcript.txt")