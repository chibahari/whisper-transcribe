import subprocess
from pathlib import Path
import noisereduce as nr
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
from tqdm import tqdm

def convert_to_wav(input_path: str, output_path: str) -> str:
    """Convert audio to 16kHz mono WAV. This is the native file format for Whisper."""
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

def enhance_audio(wav_path: str, output_path: str, normalised_path: str = "tmp/1_5_normalised.wav") -> str:
    """Normalise volume and reduce background noise."""
    with tqdm(total=5, desc="Enhancing audio", unit="step") as pbar:
        # 1. Load audio
        audio = AudioSegment.from_wav(wav_path)
        pbar.update(1)
        pbar.set_postfix_str("Normalising")

        # 2. Normalise with pydub
        normalised = effects.normalize(audio)
        normalised.export(normalised_path, format="wav")
        pbar.update(1)
        pbar.set_postfix_str("Loading for noise reduction")

        # 3. Read for noise reduction
        data, rate = sf.read(normalised_path)
        pbar.update(1)
        pbar.set_postfix_str("Reducing noise")

        # 4. Noise reduction with noisereduce
        # Use the first 0.5s as a noise profile (assumes ambient noise at start)
        noise_sample = data[:int(rate * 0.5)]
        reduced = nr.reduce_noise(y=data, sr=rate, y_noise=noise_sample, prop_decrease=0.8)
        pbar.update(1)
        pbar.set_postfix_str("Saving")

        # 5. Write to output
        sf.write(output_path, reduced, rate)
        pbar.update(1)
        pbar.set_postfix_str("Done")

    print("Audio enhancement complete.")
    return output_path