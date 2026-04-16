import subprocess
from pathlib import Path
import noisereduce as nr
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects

def m4a_to_wav(input_path: str, output_path: str) -> str:
    """Convert m4a to 16kHz mono WAV. This is the native file format for Whisper."""
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",   # 16kHz sample rate (Whisper expects this)
        "-ac", "1",        # mono
        "-c:a", "pcm_s16le",
        output_path
    ], check=True)
    print('Converted WAV file at', output_path)
    return output_path

def enhance_audio(wav_path: str, output_path: str) -> str:
    """Normalise volume and reduce background noise."""
    # Normalise with pydub
    audio = AudioSegment.from_wav(wav_path)
    normalized = effects.normalize(audio)
    normalized.export("/tmp/normalized.wav", format="wav")

    # Noise reduction with noisereduce
    data, rate = sf.read("/tmp/normalized.wav")
    # Use the first 0.5s as a noise profile (assumes ambient noise at start)
    noise_sample = data[:int(rate * 0.5)]
    reduced = nr.reduce_noise(y=data, sr=rate, y_noise=noise_sample, prop_decrease=0.8)
    sf.write(output_path, reduced, rate)

    return output_path