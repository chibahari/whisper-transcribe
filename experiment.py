"""
Config comparison harness for the AIIR interviews.

Each config picks a backend (`mlx_whisper` or `hf_transformers`) plus its
kwargs. Results are written to output/experiments/{clip_stem}/{config_name}.json
plus a summary row appended to output/experiments/{clip_stem}/summary.txt.
The mesolitica configs use HF transformers on MPS because there is no MLX
conversion of the Malaysian fine-tune published.
"""
import argparse
import json
import time
from pathlib import Path

import mlx_whisper

MODEL_LARGE_V3 = "mlx-community/whisper-large-v3-mlx"
MODEL_TURBO = "mlx-community/whisper-large-v3-turbo"

PROMPT_BASIC = (
    "Temu bual ini mengandungi Bahasa Melayu dan Bahasa Inggeris. "
    "This interview contains both Malay and English with frequent code switching."
)

# Rich prompt: gives Whisper the proper nouns and domain vocabulary so it is
# less likely to substitute or slide into a hallucinated Malay phrase.
PROMPT_RICH = (
    "Temu bual penyelidikan tentang pelaporan insiden AI di Malaysia. "
    "Interviewees are from MCMC, NSC, NACSA, CSM, BNM. "
    "Discussion covers CMA 1998 Section 263, cybersecurity incidents, "
    "data breach, ransomware, phishing, SMB port 445, telco licensees, "
    "postal operators, courier services, SII entities, aduan (complaints), "
    "penyedia rangkaian, broadcasting, incident reporting. "
    "This interview contains both Malay and English with frequent code switching."
)

MODEL_MESOLITICA_TURBO = "mesolitica/Malaysian-whisper-large-v3-turbo-v3"

# Backend sentinels — the "_backend" key selects which transcribe function to
# call. Everything else in the config is passed through to that backend as
# kwargs (minus underscore-prefixed keys, which are meta).
BACKEND_MLX = "mlx_whisper"
BACKEND_HF = "hf_transformers"
BACKEND_HF_PER_CHUNK_LANG = "hf_transformers_per_chunk_lang"

CONFIGS = {
    "A_baseline": dict(
        path_or_hf_repo=MODEL_LARGE_V3,
        language=None,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6),
        no_speech_threshold=0.6,
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
        task="transcribe",
        initial_prompt=PROMPT_BASIC,
    ),
    "B_hallu_guard": dict(
        path_or_hf_repo=MODEL_LARGE_V3,
        language=None,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6),
        no_speech_threshold=0.6,
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
        hallucination_silence_threshold=2.0,
        task="transcribe",
        initial_prompt=PROMPT_BASIC,
    ),
    "C_rich_prompt": dict(
        path_or_hf_repo=MODEL_LARGE_V3,
        language=None,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6),
        no_speech_threshold=0.6,
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
        hallucination_silence_threshold=2.0,
        task="transcribe",
        initial_prompt=PROMPT_RICH,
    ),
    "D_turbo": dict(
        path_or_hf_repo=MODEL_TURBO,
        language=None,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6),
        no_speech_threshold=0.6,
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
        hallucination_silence_threshold=2.0,
        task="transcribe",
        initial_prompt=PROMPT_RICH,
    ),
    "E_aggressive_temp": dict(
        # Wider temperature ladder to escape failure modes faster + all guards
        path_or_hf_repo=MODEL_LARGE_V3,
        language=None,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=(0.0, 0.4, 0.8, 1.0),
        no_speech_threshold=0.6,
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
        hallucination_silence_threshold=2.0,
        task="transcribe",
        initial_prompt=PROMPT_RICH,
    ),
    "F_extra_fallback": dict(
        # Extend the fallback ladder to 5 steps ending at 0.8 (not 1.0) so
        # persistent single-segment loops get one more escape attempt.
        path_or_hf_repo=MODEL_LARGE_V3,
        language=None,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
        no_speech_threshold=0.6,
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
        hallucination_silence_threshold=2.0,
        task="transcribe",
        initial_prompt=PROMPT_RICH,
    ),
    "G_tight_cr": dict(
        # Same as C but tighter compression_ratio_threshold to trigger fallback
        # earlier on repetitive output.
        path_or_hf_repo=MODEL_LARGE_V3,
        language=None,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
        no_speech_threshold=0.6,
        compression_ratio_threshold=1.2,
        logprob_threshold=-1.0,
        hallucination_silence_threshold=2.0,
        task="transcribe",
        initial_prompt=PROMPT_RICH,
    ),
    "M_mesolitica_turbo": dict(
        # Malaysian fine-tune of whisper-large-v3-turbo. Trained on Malaysian
        # speech with explicit Manglish (Malay-English) code-switching. No MLX
        # conversion published, so runs via HF transformers on MPS.
        _backend=BACKEND_HF,
        model_id=MODEL_MESOLITICA_TURBO,
        # language=None lets the model's own head decide per chunk; the
        # fine-tune should handle code-switching without forcing a language.
        language=None,
        task="transcribe",
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
    ),
    "M2_stride0": dict(
        # Same as M but with no stride overlap between chunks. Observed
        # duplication pattern (English clause immediately followed by a formal
        # Malay translation of the same clause) suggests adjacent chunks
        # decoded the overlap region in different languages and both survived.
        _backend=BACKEND_HF,
        model_id=MODEL_MESOLITICA_TURBO,
        language=None,
        task="transcribe",
        chunk_length_s=30,
        stride_length_s=0,
        return_timestamps=True,
    ),
    "M3_lang_en": dict(
        # Force language="en" globally. The interviewer speaks predominantly
        # English but their utterances kept getting decoded as formal Malay
        # (both when isolated and when they open the recording). Forcing
        # English tests whether we lose accuracy on the interviewee's Malay
        # in exchange for fixing the interviewer.
        _backend=BACKEND_HF,
        model_id=MODEL_MESOLITICA_TURBO,
        language="en",
        task="transcribe",
        chunk_length_s=30,
        stride_length_s=0,
        return_timestamps=True,
    ),
    "M4_detect_lang": dict(
        # VAD-pre-chunk the audio, run a per-chunk language probe, then force
        # the detected language during transcription — with an English bias
        # (if p(en) > en_bias, force en regardless of top language). This
        # counters mesolitica's Malay bias on short English interviewer
        # utterances, which is the failure mode M2_stride0 didn't fix.
        _backend=BACKEND_HF_PER_CHUNK_LANG,
        model_id=MODEL_MESOLITICA_TURBO,
        task="transcribe",
        en_bias=0.3,
        return_timestamps=True,
    ),
}


_hf_pipeline_cache: dict = {}


def _get_hf_pipeline(model_id: str):
    """Load HF Whisper pipeline once per model_id; MPS + bfloat16."""
    if model_id in _hf_pipeline_cache:
        return _hf_pipeline_cache[model_id]

    import torch
    from transformers import pipeline
    from transformers.models.whisper import tokenization_whisper

    # Model card registers a custom `transcribeprecise` task token. Even if
    # we don't use it, keeping the token list in sync avoids errors at load.
    if "transcribeprecise" not in getattr(tokenization_whisper, "TASK_IDS", []):
        tokenization_whisper.TASK_IDS = list(
            getattr(tokenization_whisper, "TASK_IDS", ["translate", "transcribe"])
        ) + ["transcribeprecise"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    print(f"    loading {model_id} on {device} ({dtype})…")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=dtype,
        device=device,
    )
    _hf_pipeline_cache[model_id] = pipe
    return pipe


def transcribe_hf(clip: Path, kwargs: dict) -> dict:
    """Transcribe via HF transformers pipeline. Returns the same schema as
    mlx_whisper.transcribe (text/segments/language) so downstream analyze()
    keeps working."""
    kwargs = dict(kwargs)
    model_id = kwargs.pop("model_id")
    language = kwargs.pop("language", None)
    task = kwargs.pop("task", "transcribe")
    chunk_length_s = kwargs.pop("chunk_length_s", 30)
    stride_length_s = kwargs.pop("stride_length_s", 5)
    return_timestamps = kwargs.pop("return_timestamps", True)

    pipe = _get_hf_pipeline(model_id)

    generate_kwargs = {"task": task}
    if language is not None:
        generate_kwargs["language"] = language

    out = pipe(
        str(clip),
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        return_timestamps=return_timestamps,
        generate_kwargs=generate_kwargs,
    )

    # Convert pipeline chunks -> whisper-style segments so analyze() works
    segments = []
    for i, ch in enumerate(out.get("chunks", []) or []):
        ts = ch.get("timestamp") or (None, None)
        start, end = ts
        if start is None:
            start = 0.0
        if end is None:
            # Some final chunks omit end; approximate from next chunk or text
            end = start + max(1.0, len(ch.get("text", "")) * 0.05)
        segments.append({
            "id": i,
            "start": float(start),
            "end": float(end),
            "text": ch.get("text", ""),
            # HF pipeline doesn't expose compression_ratio; analyze() falls
            # back to 0 for missing fields.
        })

    return {
        "text": out.get("text", ""),
        "segments": segments,
        "language": None,  # HF pipeline doesn't return detected language
    }


def detect_chunk_language(pipe, chunk_audio, sample_rate: int, en_bias: float,
                          candidate_langs=("en", "ms", "id", "zh", "ta")) -> tuple[str, dict]:
    """One-step language probe: run the model for a single decoder position and
    read the language-token logits directly. Returns (chosen_lang, probs_dict)."""
    import torch

    inputs = pipe.feature_extractor(chunk_audio, sampling_rate=sample_rate,
                                    return_tensors="pt")
    dtype = next(pipe.model.parameters()).dtype
    input_features = inputs.input_features.to(pipe.model.device, dtype)

    decoder_start = pipe.model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start]], device=pipe.model.device)
    with torch.no_grad():
        outputs = pipe.model(input_features, decoder_input_ids=decoder_input_ids)
    lang_logits = outputs.logits[0, -1].float()
    probs_all = torch.softmax(lang_logits, dim=-1)

    tok = pipe.tokenizer
    probs = {}
    for code in candidate_langs:
        tok_id = tok.convert_tokens_to_ids(f"<|{code}|>")
        if tok_id is not None and tok_id != tok.unk_token_id:
            probs[code] = float(probs_all[tok_id])

    if not probs:
        return "en", {}
    top = max(probs, key=probs.get)
    if probs.get("en", 0.0) > en_bias:
        return "en", probs
    return top, probs


def transcribe_hf_per_chunk_lang(clip: Path, kwargs: dict) -> dict:
    """VAD-chunk audio, detect language per chunk (with English bias), then
    transcribe each chunk with the detected language forced. Bypasses the
    pipeline's whole-chunk auto-detect, which mis-classifies short English
    interviewer utterances as Malay on this dataset."""
    import numpy as np
    import soundfile as sf
    from transcribe import vad_chunks, SAMPLE_RATE

    kwargs = dict(kwargs)
    model_id = kwargs.pop("model_id")
    task = kwargs.pop("task", "transcribe")
    en_bias = kwargs.pop("en_bias", 0.3)
    return_timestamps = kwargs.pop("return_timestamps", True)

    pipe = _get_hf_pipeline(model_id)

    audio, sr = sf.read(str(clip), dtype="float32")
    assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    chunks = vad_chunks(str(clip))
    print(f"    VAD produced {len(chunks)} chunks")

    all_segments = []
    lang_counts = {}
    for i, (start, end) in enumerate(chunks):
        s0, s1 = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        chunk_audio = np.ascontiguousarray(audio[s0:s1])

        lang, probs = detect_chunk_language(pipe, chunk_audio, SAMPLE_RATE, en_bias)
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        out = pipe(
            {"array": chunk_audio, "sampling_rate": SAMPLE_RATE},
            return_timestamps=return_timestamps,
            generate_kwargs={"task": task, "language": lang},
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
                "start": float(cs) + start,
                "end": float(ce) + start,
                "text": ch.get("text", ""),
                "_lang": lang,
            })

    print(f"    lang distribution: {lang_counts}")
    return {
        "text": " ".join(s["text"].strip() for s in all_segments),
        "segments": all_segments,
        "language": None,
    }


def analyze(result: dict) -> dict:
    segs = result["segments"]
    if not segs:
        return {"segments": 0}

    # Detect exact-repetition loops
    prev = None
    run = 1
    loop_runs = []
    for s in segs:
        t = s["text"].strip()
        if t == prev and t:
            run += 1
        else:
            if run >= 3:
                loop_runs.append(run)
            run = 1
            prev = t
    if run >= 3:
        loop_runs.append(run)

    high_cr = sum(1 for s in segs if s.get("compression_ratio", 0) > 2.4)
    very_high_cr = sum(1 for s in segs if s.get("compression_ratio", 0) > 3.0)

    # Rough English proportion via ASCII ratio (Malay uses same ASCII, so this
    # is only a weak signal, but useful for comparing configs on same audio.)
    total_chars = sum(len(s["text"]) for s in segs)

    return {
        "segments": len(segs),
        "duration_s": round(segs[-1]["end"], 1),
        "detected_language": result.get("language"),
        "loop_count": len(loop_runs),
        "worst_loop_len": max(loop_runs) if loop_runs else 0,
        "high_cr_segments": high_cr,
        "very_high_cr_segments": very_high_cr,
        "total_chars": total_chars,
    }


def run(clip: Path, configs: list[str], out_root: Path) -> None:
    stem = clip.stem
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    for name in configs:
        if name not in CONFIGS:
            print(f"skip unknown config {name}")
            continue

        out_json = out_dir / f"{name}.json"
        if out_json.exists():
            print(f"[{name}] already exists at {out_json}, skipping")
            with open(out_json) as f:
                result = json.load(f)
            stats = result.get("_stats") or analyze(result)
        else:
            kwargs = dict(CONFIGS[name])
            backend = kwargs.pop("_backend", BACKEND_MLX)
            print(f"\n=== [{name}] running on {clip.name} ===")
            model_label = kwargs.get("path_or_hf_repo") or kwargs.get("model_id") or "?"
            print(f"    backend: {backend}  model: {model_label}")
            t0 = time.time()
            if backend == BACKEND_MLX:
                result = mlx_whisper.transcribe(str(clip), verbose=False, **kwargs)
            elif backend == BACKEND_HF:
                result = transcribe_hf(clip, kwargs)
            elif backend == BACKEND_HF_PER_CHUNK_LANG:
                result = transcribe_hf_per_chunk_lang(clip, kwargs)
            else:
                raise ValueError(f"unknown backend {backend}")
            elapsed = time.time() - t0
            stats = analyze(result)
            stats["elapsed_s"] = round(elapsed, 1)
            result["_stats"] = stats
            result["_config"] = {k: v for k, v in kwargs.items() if k != "initial_prompt"}
            result["_backend"] = backend
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # Also write a plain text view for eyeballing
            with open(out_dir / f"{name}.txt", "w", encoding="utf-8") as f:
                for s in result["segments"]:
                    f.write(f"[{s['start']:7.1f}s] {s['text'].strip()}\n")

        line = f"{name:22s} loops={stats['loop_count']:2d}  worst={stats['worst_loop_len']:3d}  high_cr={stats['high_cr_segments']:3d}  vhigh_cr={stats['very_high_cr_segments']:3d}  segs={stats['segments']:4d}  chars={stats['total_chars']:6d}  lang={stats.get('detected_language')}  t={stats.get('elapsed_s','?')}s"
        print(line)
        summary_lines.append(line)

    with open(out_dir / "summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSummary written to {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True, help="Path to audio clip")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    p.add_argument("--out-root", default="output/experiments")
    args = p.parse_args()
    run(Path(args.clip), args.configs, Path(args.out_root))
