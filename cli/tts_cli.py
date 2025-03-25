import sys
import os
import torch
import numpy as np
import soundfile as sf
import logging
from datetime import datetime
import time
import librosa
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import EMO_MAP

from semantic_text_splitter import TextSplitter
import random

# Global cache for reuse
_cached_model_instance = None


def generate_tts_audio(
    text,
    model_dir=None,
    device="cuda:0",
    prompt_speech_path=None,
    prompt_text=None,
    gender=None,
    pitch=None,
    speed=None,
    emotion=None,
    save_dir="example/results",
    segmentation_threshold=None,
    seed=None,
    model=None,
    skip_model_init=False
):

    """
    Generates TTS audio from input text, splitting into segments if necessary.

    Args:
        text (str): Input text for speech synthesis.
        model_dir (str): Path to the model directory.
        device (str): Device identifier (e.g., "cuda:0" or "cpu").
        prompt_speech_path (str, optional): Path to prompt audio for cloning.
        prompt_text (str, optional): Transcript of prompt audio.
        gender (str, optional): Gender parameter ("male"/"female").
        pitch (str, optional): Pitch parameter (e.g., "moderate").
        speed (str, optional): Speed parameter (e.g., "moderate").
        emotion (str, optional): Emotion tag (e.g., "HAPPY", "SAD", "ANGRY").
        save_dir (str): Directory where generated audio will be saved.
        segmentation_threshold (int): Maximum number of words per segment.
        seed (int, optional): Seed value for deterministic voice generation.

    Returns:
        str: The unique file path where the generated audio is saved.
    """
    # ============================== OPTIONS REFERENCE ==============================
    # ✔ Gender options: "male", "female"
    # ✔ Pitch options: "very_low", "low", "moderate", "high", "very_high"
    # ✔ Speed options: same as pitch
    # ✔ Emotion options: list from token_parser.py EMO_MAP keys
    # ✔ Seed: any integer (e.g., 1337, 42, 123456) = same voice (mostly)
    # ==============================================================================

    if model_dir is None:
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "Spark-TTS-0.5B"))

    global _cached_model_instance

    if not skip_model_init or model is None:
        if _cached_model_instance is None:
            logging.info("Initializing TTS model...")
            if not prompt_speech_path:
                logging.info(f"Using Gender: {gender or 'default'}, Pitch: {pitch or 'default'}, Speed: {speed or 'default'}, Emotion: {emotion or 'none'}, Seed: {seed or 'random'}")
            model = SparkTTS(model_dir, torch.device(device))
            _cached_model_instance = model
        else:
            model = _cached_model_instance

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Seed set to: {seed}")
    # Set seed for reproducibility
    else:
        seed = random.randint(0, 4294967295)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            #torch.use_deterministic_algorithms(True)
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Seed set to: {seed}")

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    # Standardize quotes to basic ASCII double quotes
    text = re.sub(r'[""„‟«»❝❞〝〞〟＂“”]', '"', text)  # Convert fancy double quotes
    text = re.sub(r'[''‚‛‹›❛❜`´’]', "'", text)  # Convert fancy single quotes/apostrophes

    # Handle other common Unicode punctuation
    text = re.sub(r'[—–]', '-', text)  # Em dash, en dash to hyphen
    text = re.sub(r'…', '...', text)  # Ellipsis
    text = re.sub(r'[•‣⁃*]', '', text)  # Bullets to none

    logging.info(f"Splitting into segments... Threshold: {segmentation_threshold} characters")
    #segments = [' '.join(words[i:i + segmentation_threshold]) for i in range(0, len(words), segmentation_threshold)]
    splitter = TextSplitter(segmentation_threshold)
    segments = splitter.chunks(text)
    logging.info(f"Number of segments: {len(segments)}")
    MAX_SILENCE_THRESHOLD = 5.0  # 10 seconds
    MAX_RETRY_ATTEMPTS = 3

    wavs = []
    for seg in segments:
        logging.info(f"Processing one segment:\n{seg}")

        retry_count = 0
        while retry_count < MAX_RETRY_ATTEMPTS:
            with torch.no_grad():
                wav = model.inference(
                    seg,
                    prompt_speech_path,
                    prompt_text=prompt_text,
                    gender=gender,
                    pitch=pitch,
                    speed=speed,
                    seed=seed,
                    emotion=emotion
                )

            # Get both the trimmed audio and the amount of silence trimmed
            trimmed_wav, seconds_trimmed = trim_trailing_silence_librosa(wav, sample_rate=16000)

            # If silence is acceptable, or we've tried too many times, use this result
            if seconds_trimmed < MAX_SILENCE_THRESHOLD or retry_count == MAX_RETRY_ATTEMPTS - 1:
                wavs.append(trimmed_wav)
                break
            else:
                logging.warning(f"Too much silence detected ({seconds_trimmed:.2f}s > {MAX_SILENCE_THRESHOLD}s). Retrying segment... (This usually means your clone clip is bad.)")
                retry_count += 1
                # You might want to vary the seed for different results
                seed = random.randint(0, 4294967295)
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

        delay_time = random.uniform(0.30, 0.5)  # Random delay between 300-500ms
        silence_samples = int(16000 * delay_time)  # 16000 is sample rate
        silence = np.zeros(silence_samples)
        wavs.append(silence)
        logging.info(f"Processed one segment{' after ' + str(retry_count) + ' retries' if retry_count > 0 else ''}.")

    final_wav = np.concatenate(wavs, axis=0)

    sf.write(save_path, final_wav, samplerate=16000)
    logging.info(f"Audio saved at: {save_path}")
    return save_path

def trim_trailing_silence_librosa(wav_data, sample_rate=16000, top_db=30, frame_length=1024, hop_length=512):
    """
    Trims trailing silence using librosa's effects.trim and returns both trimmed audio and seconds trimmed

    Args:
        wav_data: numpy array of audio samples
        sample_rate: audio sample rate
        top_db: threshold in dB for silence detection
        frame_length: analysis frame length
        hop_length: analysis frame hop length

    Returns:
        Tuple of (trimmed audio as numpy array, seconds trimmed)
    """
    try:
        # Trim starting and trailing silence
        y_trimmed, _ = librosa.effects.trim(
            wav_data,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )

        seconds_trimmed = (len(wav_data) - len(y_trimmed)) / sample_rate
        logging.info(f"Trimmed {seconds_trimmed:.2f}s of silence using librosa")

        return y_trimmed, seconds_trimmed
    except Exception as e:
        logging.warning(f"Error trimming silence with librosa: {e}")
        return wav_data, 0.0

# Example CLI usage
if __name__ == "__main__":
    import argparse


    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_audio", type=str, help="Path to audio file for voice cloning")
    parser.add_argument("--prompt_text", type=str, help="Transcript text for the prompt audio (optional)")
    parser.add_argument("--text", type=str, help="Text to generate", required=False)
    parser.add_argument("--text_file", type=str, help="Path to .txt file with input text")
    parser.add_argument("--gender", type=str, choices=["male", "female"], default=None)
    parser.add_argument("--pitch", type=str, choices=["very_low", "low", "moderate", "high", "very_high"], default="moderate")
    parser.add_argument("--speed", type=str, choices=["very_low", "low", "moderate", "high", "very_high"], default="moderate")
    parser.add_argument("--emotion", type=str, choices=list(EMO_MAP.keys()), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seg_threshold", type=int, default=400)
    args = parser.parse_args()


    # ---------------- Argument Validation Block ---------------- NEW! SPECIAL!!!EXTRA SPICY!!!
    if not args.prompt_audio and not args.gender:
        print("❌ Error: You must provide either --gender (male/female) or --prompt_audio for voice cloning.")
        print("   Example 1: python tts_cli.py --text \"Hello there.\" --gender female")
        print("   Example 2: python tts_cli.py --text \"Hello there.\" --prompt_audio sample.wav")
        sys.exit(1)

    # --------------- Emotions ------------
    if args.emotion:
        logging.warning("⚠ Emotion input is experimental — model may not reflect emotion changes reliably or at all.")



    # Allow loading text from a file if provided
    if args.text_file:
        if os.path.exists(args.text_file):
            with open(args.text_file, "r", encoding="utf-8") as f:
                args.text = f.read().strip()
        else:
            raise FileNotFoundError(f"Text file not found: {args.text_file}")

    # If Not Provided Text or Text File
    if not args.text:
        raise ValueError("You must provide either --text or --text_file.")

    # Voice Cloning Mode Overrides
    if args.prompt_audio:
        # Normalize path + validate
        args.prompt_audio = os.path.abspath(args.prompt_audio)
        if not os.path.exists(args.prompt_audio):
            logging.error(f"❌ Prompt audio file not found: {args.prompt_audio}")
            sys.exit(1)

        # Log cloning info
        logging.info("🔊 Voice cloning mode enabled")
        logging.info(f"🎧 Cloning from: {args.prompt_audio}")

        # Bonus: Log audio info
        try:
            info = sf.info(args.prompt_audio)
            logging.info(f"📏 Prompt duration: {info.duration:.2f} seconds | Sample Rate: {info.samplerate}")
        except Exception as e:
            logging.warning(f"⚠️ Could not read prompt audio info: {e}")

        # Override pitch/speed/gender
        if args.gender or args.pitch or args.speed:
            print("[!] Warning: Voice cloning mode detected — ignoring gender/pitch/speed settings.")
        args.gender = None
        args.pitch = None
        args.speed = None

    # Start timing
    start_time = time.time()

    output_file = generate_tts_audio(
        text=args.text,
        gender=args.gender,
        pitch=args.pitch,
        speed=args.speed,
        emotion=args.emotion,
        seed=args.seed,
        prompt_speech_path=args.prompt_audio,
        prompt_text=args.prompt_text,
        segmentation_threshold=args.seg_threshold,
    )

    # End timing
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Generated audio file: {output_file}")
    print(f"⏱ Generation time: {elapsed:.2f} seconds")


