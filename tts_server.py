import sys
import os
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

# Import your existing TTS code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import EMO_MAP

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global cache for reuse
_cached_model_instance = None
# Global voice parameters set at server startup
GLOBAL_VOICE_PARAMS = {
    "gender": None,
    "pitch": None,
    "speed": None,
    "emotion": None,
    "seed": None,
    "prompt_speech_path": None,
    "prompt_text": None
}

app = Flask(__name__)
CORS(app)

# Your existing generate_tts_audio function
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
        #data = request.json

        # Log the incoming request
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
        
        if not input_text:
            return jsonify({"error": "Input text is required"}), 400
            
        # Create a temp directory for outputs
        temp_dir = tempfile.mkdtemp()
        
        # Generate the audio using global voice parameters
        output_file = generate_tts_audio(
            text=input_text,
            gender=GLOBAL_VOICE_PARAMS["gender"],
            pitch=GLOBAL_VOICE_PARAMS["pitch"],
            speed=GLOBAL_VOICE_PARAMS["speed"],
            emotion=GLOBAL_VOICE_PARAMS["emotion"],
            seed=GLOBAL_VOICE_PARAMS["seed"],
            prompt_speech_path=GLOBAL_VOICE_PARAMS["prompt_speech_path"],
            prompt_text=GLOBAL_VOICE_PARAMS["prompt_text"],
            segmentation_threshold=400,
            save_dir=temp_dir
        )
        
        # Convert to mp3 if needed
        if response_format == 'mp3':
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

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    voice_info = {
        "gender": GLOBAL_VOICE_PARAMS["gender"],
        "pitch": GLOBAL_VOICE_PARAMS["pitch"],
        "speed": GLOBAL_VOICE_PARAMS["speed"],
        "emotion": GLOBAL_VOICE_PARAMS["emotion"],
        "using_prompt": GLOBAL_VOICE_PARAMS["prompt_speech_path"] is not None
    }
    
    return jsonify({
        "status": "ok", 
        "message": "SparkTTS server is running", 
        "voice_config": voice_info
    }), 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenAI-compatible TTS Server for SparkTTS')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for model inference')
    parser.add_argument('--model_dir', type=str, help='Path to the SparkTTS model directory')
    
    # Voice configuration arguments
    parser.add_argument('--prompt_audio', type=str, help='Path to audio file for voice cloning')
    parser.add_argument('--prompt_text', type=str, help='Transcript text for the prompt audio (optional)')
    parser.add_argument('--gender', type=str, choices=["male", "female"], help='Gender parameter')
    parser.add_argument('--pitch', type=str, choices=["very_low", "low", "moderate", "high", "very_high"], 
                        default="moderate", help='Pitch parameter')
    parser.add_argument('--speed', type=str, choices=["very_low", "low", "moderate", "high", "very_high"], 
                        default="moderate", help='Speed parameter')
    parser.add_argument('--emotion', type=str, choices=list(EMO_MAP.keys()), help='Emotion tag')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seg_threshold", type=int, default=400)
    
    args = parser.parse_args()
    
    # Validate voice configuration
    #if not args.prompt_audio and not args.gender:
    #    logging.error("❌ Error: You must provide either --gender (male/female) or --prompt_audio for voice cloning.")
    #    sys.exit(1)
    
    # Set global voice parameters
    GLOBAL_VOICE_PARAMS["gender"] = args.gender if not args.prompt_audio else None
    GLOBAL_VOICE_PARAMS["pitch"] = args.pitch if not args.prompt_audio else None
    GLOBAL_VOICE_PARAMS["speed"] = args.speed if not args.prompt_audio else None
    GLOBAL_VOICE_PARAMS["emotion"] = args.emotion
    GLOBAL_VOICE_PARAMS["seed"] = args.seed
    GLOBAL_VOICE_PARAMS["prompt_speech_path"] = os.path.abspath(args.prompt_audio) if args.prompt_audio else None
    GLOBAL_VOICE_PARAMS["prompt_text"] = args.prompt_text
    
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

        # Override pitch/speed/gender
        if args.gender or args.pitch or args.speed:
            print("[!] Warning: Voice cloning mode detected — ignoring gender/pitch/speed settings.")
        args.gender = None
        args.pitch = None
        args.speed = None
    else:
        logging.info(f"🔊 Using configured voice: Gender={args.gender}, Pitch={args.pitch}, Speed={args.speed}")
        if args.emotion:
            logging.info(f"😊 Emotion: {args.emotion}")
        if args.seed:
            logging.info(f"🎲 Fixed seed: {args.seed}")
    
    # Preload the model on startup if model_dir is provided
    if args.model_dir:
        try:
            logging.info(f"Preloading SparkTTS model from {args.model_dir}")
            _cached_model_instance = SparkTTS(args.model_dir, torch.device(args.device))
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error preloading model: {e}")
            sys.exit(1)
            
    logging.info(f"Starting OpenAI-compatible TTS server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
