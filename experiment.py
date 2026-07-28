"""
Config comparison harness for mlx_whisper on the AIIR interviews.

Each config is a dict of kwargs passed to mlx_whisper.transcribe. Results are
written to output/experiments/{clip_stem}/{config_name}.json plus a summary
row appended to output/experiments/{clip_stem}/summary.txt.
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
            kwargs = CONFIGS[name]
            print(f"\n=== [{name}] running on {clip.name} ===")
            print(f"    model: {kwargs['path_or_hf_repo']}")
            t0 = time.time()
            result = mlx_whisper.transcribe(str(clip), verbose=False, **kwargs)
            elapsed = time.time() - t0
            stats = analyze(result)
            stats["elapsed_s"] = round(elapsed, 1)
            result["_stats"] = stats
            result["_config"] = {k: v for k, v in kwargs.items() if k != "initial_prompt"}
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
