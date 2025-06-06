import sys
import os
import shutil
import subprocess
import platform
import torch
import numpy as np
import soundfile as sf
import logging
from datetime import datetime
import time
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import argparse
import uuid
from semantic_text_splitter import TextSplitter
import random
import json
import librosa
import re
import demoji

# Import Chatterbox TTS
from chatterbox.tts import ChatterboxTTS

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global cache for reuse
_cached_model_instance = None
# Global voice parameters set at server startup
GLOBAL_VOICE_PARAMS = {
    "seed": None,
    "prompt_speech_path": None,
    "cfg_weight": 0.5,
    "exaggeration": 0.5,
    "temperature": 0.8
}

app = Flask(__name__)
CORS(app)

def is_ffmpeg_available():
    """Check if ffmpeg is available on the system path, works on Windows and Linux."""
    # Check if we're on Windows
    is_windows = platform.system().lower() == "windows"

    # On Windows, we should also check for ffmpeg.exe specifically
    ffmpeg_commands = ['ffmpeg.exe', 'ffmpeg'] if is_windows else ['ffmpeg']

    # Method 1: Using shutil.which (works on both platforms)
    for cmd in ffmpeg_commands:
        if shutil.which(cmd) is not None:
            return True

    # Method 2: Fallback to subprocess
    for cmd in ffmpeg_commands:
        try:
            # On Windows, shell=True might be needed in some environments
            subprocess.run(
                [cmd, '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=5,
                shell=is_windows
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            continue

    return False


ffmpeg_available = is_ffmpeg_available()
logging.info(f"FFmpeg is {'available' if ffmpeg_available else 'not available'}")

# Your existing generate_tts_audio function
def generate_tts_audio(
    text,
    device="cuda:0",
    prompt_speech_path=None,
    save_dir="example/results",
    segmentation_threshold=None,
    seed=None,
    model=None,
    skip_model_init=False,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8
):
    """
    Generates TTS audio from input text using Chatterbox, splitting into segments if necessary.
    Args:
        text (str): Input text for speech synthesis.
        device (str): Device identifier (e.g., "cuda:0" or "cpu").
        prompt_speech_path (str, optional): Path to prompt audio for voice cloning.
        save_dir (str): Directory where generated audio will be saved.
        segmentation_threshold (int): Maximum number of characters per segment.
        seed (int, optional): Seed value for deterministic voice generation.
        exaggeration (float): Exaggeration level for generation (0.0-1.0).
        cfg_weight (float): CFG weight for generation (0.0-1.0).
        temperature (float): Temperature for generation (0.0-1.0).
    Returns:
        str: The unique file path where the generated audio is saved.
    """
    # ============================== OPTIONS REFERENCE ==============================
    # ✔ Audio prompt: path to reference audio file for voice cloning
    # ✔ Exaggeration: 0.0-1.0 (default: 0.5)
    # ✔ CFG Weight: 0.0-1.0 (default: 0.5)  
    # ✔ Temperature: 0.0-1.0 (default: 0.8)
    # ✔ Seed: any integer for deterministic generation
    # ==============================================================================
    
    global _cached_model_instance
    if not skip_model_init or model is None:
        if _cached_model_instance is None:
            logging.info("Initializing Chatterbox TTS model...")
            if prompt_speech_path:
                logging.info(f"Using voice cloning with prompt: {prompt_speech_path}")
            else:
                logging.info("Using default Chatterbox voice")
            model = ChatterboxTTS.from_pretrained(device=device)
            _cached_model_instance = model
        else:
            model = _cached_model_instance
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Seed set to: {seed}")
    else:
        seed = random.randint(0, 4294967295)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Seed set to: {seed}")
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    # Standardize quotes to basic ASCII double quotes
    text = re.sub(r'[""„‟«»❝❞〝〞〟＂“”＂]', '"', text)  # Convert fancy double quotes
    text = re.sub(r'[''‚‛‹›❛❜`´’‘]', "'", text)  # Convert fancy single quotes/apostrophes

    # Handle other common Unicode punctuation
    text = re.sub(r'[–]', '-', text)  # En dash to hyphen
    text = re.sub(r'…', '...', text)  # Ellipsis
    text = re.sub(r'[•‣⁃*]', '', text)  # Bullets to none

    text = demoji.replace(text, "")

    if not args.allow_allcaps:
        # Convert all-caps words to lowercase (model chokes on all caps)
        # NOTE FROM BOT: Find out if this is the case for Chatterbox
        def lowercase_all_caps(match):
            word = match.group(0)
            if word.isupper() and len(word) > 1:
                return word.lower()
            return word

        text = re.sub(r'\b[A-Z][A-Z]+\b', lowercase_all_caps, text)
    
    logging.info(f"Splitting into segments... Threshold: {segmentation_threshold} characters")
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
                wav = model.generate(
                    seg,
                    audio_prompt_path=prompt_speech_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature
                )
            # NOTE FROM BOT: Workaround for a tensor shape error that occurred with Chatterbox. There may be a more "correct" fix, idk, i don't Torch. It works.
            # Convert to numpy and ensure it's 1D
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            
            # Ensure wav is 1D
            if wav.ndim > 1:
                wav = wav.squeeze()
            if wav.ndim > 1:
                wav = wav[0]  # Take first channel if still multi-dimensional
            
            # Get both the trimmed audio and the amount of silence trimmed
            trimmed_wav, seconds_trimmed = trim_trailing_silence_librosa(wav, sample_rate=model.sr)

            # If silence is acceptable, or we've tried too many times, use this result
            if seconds_trimmed < MAX_SILENCE_THRESHOLD or retry_count == MAX_RETRY_ATTEMPTS - 1:
                # Ensure trimmed_wav is 1D
                if trimmed_wav.ndim > 1:
                    trimmed_wav = trimmed_wav.squeeze()
                if trimmed_wav.ndim > 1:
                    trimmed_wav = trimmed_wav[0]
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
        silence_samples = int(model.sr * delay_time)  # Use model's sample rate
        silence = np.zeros(silence_samples, dtype=np.float32)  # Ensure 1D array
        wavs.append(silence)
        logging.info(f"Processed one segment{' after ' + str(retry_count) + ' retries' if retry_count > 0 else ''}.")
    
    final_wav = np.concatenate(wavs, axis=0)
    sf.write(save_path, final_wav, samplerate=model.sr)
    logging.info(f"Audio saved at: {save_path}")
    
    return save_path

# Format converter for the audio file
def convert_wav_to_mp3(wav_path):
    """Convert WAV file to MP3 format"""
    try:
        from pydub import AudioSegment
        mp3_path = wav_path.replace('.wav', '.mp3')
        sound = AudioSegment.from_wav(wav_path)
        sound.export(mp3_path, format="mp3", parameters=["-q:a", "0"])
        return mp3_path
    except ImportError:
        logging.warning("pydub not installed. Returning WAV file instead.")
        return wav_path

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

# OpenAI compatible TTS endpoint
@app.route('/v1/audio/speech', methods=['POST'])
def tts():
    try:
        # Parse the request
        data = request.json
        client_ip = request.remote_addr

        # Print request details to console
        print("\n==== TTS REQUEST ====")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Client IP: {client_ip}")
        print("Request data:")
        print(json.dumps(data, indent=2))
        print("====================\n")
        
        # Extract parameters similar to OpenAI's API
        model = data.get('model', 'tts-1')  # Ignored but included for compatibility
        input_text = data.get('input')
        response_format = data.get('response_format', 'mp3')
        speed = data.get('speed', 1.0)  # OpenAI speed parameter (0.25 to 4.0)
        
        if not input_text:
            return jsonify({"error": "Input text is required"}), 400
        
        # Validate speed parameter (OpenAI compatible range)
        if speed < 0.25 or speed > 4.0:
            return jsonify({"error": "Speed must be between 0.25 and 4.0"}), 400
        
        # Map OpenAI speed to Chatterbox cfg_weight
        # OpenAI speed: 0.25 (very slow) to 4.0 (very fast)
        # Chatterbox cfg_weight: 0.0 to 1.0 (higher = faster pace according to Resemble)
        # Linear mapping: speed 0.25->0.0, speed 1.0->0.5, speed 4.0->1.0
        cfg_weight = min(1.0, max(0.0, (speed - 0.25) / 3.75))
        
        logging.info(f"Speed parameter: {speed} -> CFG weight: {cfg_weight:.3f}")
            
        # Create a temp directory for outputs
        temp_dir = tempfile.mkdtemp()
        
        # Generate the audio using global voice parameters and speed mapping
        output_file = generate_tts_audio(
            text=input_text,
            device=inference_device if 'inference_device' in locals() else 'cuda:0',
            prompt_speech_path=GLOBAL_VOICE_PARAMS["prompt_speech_path"],
            seed=GLOBAL_VOICE_PARAMS["seed"],
            segmentation_threshold=400,
            save_dir=temp_dir,
            cfg_weight=cfg_weight  # Use the mapped cfg_weight from speed
        )
        
        # Convert to mp3 if needed
        if response_format == 'mp3' and ffmpeg_available:
            output_file = convert_wav_to_mp3(output_file)
            mimetype = "audio/mpeg"
        else:
            mimetype = "audio/wav"
        
        # Return the audio file
        return send_file(output_file, mimetype=mimetype)
        
    except Exception as e:
        logging.error(f"Error in TTS endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """Mimics OpenAI's models endpoint"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai",
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai",
            }
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    voice_info = {
        "using_prompt": GLOBAL_VOICE_PARAMS["prompt_speech_path"] is not None
    }
    
    return jsonify({
        "status": "ok", 
        "message": "Chatterbox TTS server is running", 
        "voice_config": voice_info
    }), 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenAI-compatible TTS Server for Chatterbox')
    parser.add_argument('--port', type=int, default=9991, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--device', type=str, default='default', help='Device to use for model inference. Using CPU is untested.')
    
    # Voice configuration arguments
    parser.add_argument('--prompt_audio', type=str, help='Path to audio file for voice cloning')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seg_threshold", type=int, default=400, help="Character limit for a single segment of text.")
    parser.add_argument("--allow_allcaps", action='store_true', help="Allow words that have 2 or more capital letters to stay capital letters. Normally these are filtered out so that the model can pronounce ALLCAPS words correctly. Useful when the text to be read has many acronyms like API or GUI or EDU.")
    
    # Chatterbox-specific parameters
    parser.add_argument("--cfg_weight", type=float, default=0.5, help="Default CFG weight for generation (0.0-1.0). Higher values = faster pace. Can be overridden by API speed parameter.")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Default exaggeration level (0.0-1.0)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Default temperature for generation (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Validate Chatterbox parameters
    if not (0.0 <= args.cfg_weight <= 1.0):
        logging.error("CFG weight must be between 0.0 and 1.0")
        sys.exit(1)
    if not (0.0 <= args.exaggeration <= 1.0):
        logging.error("Exaggeration must be between 0.0 and 1.0")
        sys.exit(1)
    if not (0.0 <= args.temperature <= 1.0):
        logging.error("Temperature must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Set global voice parameters  
    GLOBAL_VOICE_PARAMS["seed"] = args.seed
    GLOBAL_VOICE_PARAMS["prompt_speech_path"] = os.path.abspath(args.prompt_audio) if args.prompt_audio else None
    GLOBAL_VOICE_PARAMS["cfg_weight"] = args.cfg_weight
    GLOBAL_VOICE_PARAMS["exaggeration"] = args.exaggeration
    GLOBAL_VOICE_PARAMS["temperature"] = args.temperature
    
    # Log voice configuration
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
    else:
        logging.info("🔊 Using default Chatterbox voice")
        if args.seed:
            logging.info(f"🎲 Fixed seed: {args.seed}")

    inference_device = None
    if args.device == "default":
        # Automatically detect the best available device
        if torch.cuda.is_available():
            inference_device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            inference_device = "mps"
        else:
            inference_device = "cpu"
    else:
        inference_device = args.device
    
    logging.info(f"Using device: {inference_device}")
    
    # Preload the Chatterbox model on startup
    try:
        logging.info("Preloading Chatterbox TTS model...")
        _cached_model_instance = ChatterboxTTS.from_pretrained(device=inference_device)
        logging.info("Chatterbox model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to preload Chatterbox model: {e}")
        sys.exit(1)
            
    logging.info(f"Starting OpenAI-compatible TTS server on http://{args.host}:{args.port}/v1/audio/speech")
    app.run(host=args.host, port=args.port, debug=False)
