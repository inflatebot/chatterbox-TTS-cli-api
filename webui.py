# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
import platform

from datetime import datetime
from chatterbox.tts import ChatterboxTTS


def initialize_model(device=0):
    """Load the Chatterbox model once at the beginning."""
    logging.info("Loading Chatterbox TTS model...")

    # Determine appropriate device based on platform and availability
    if platform.system() == "Darwin":
        # macOS with MPS support (Apple Silicon)
        device_str = "mps"
        logging.info(f"Using MPS device: {device_str}")
    elif torch.cuda.is_available():
        # System with CUDA support
        device_str = f"cuda:{device}"
        logging.info(f"Using CUDA device: {device_str}")
    else:
        # Fall back to CPU
        device_str = "cpu"
        logging.info("GPU acceleration not available, using CPU")

    model = ChatterboxTTS.from_pretrained(device=device_str)
    return model


def run_tts(
    text,
    model,
    prompt_speech=None,
    save_dir="example/results",
):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Saving audio to: {save_dir}")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.generate(
            text,
            audio_prompt_path=prompt_speech
        )

        # Convert to numpy if it's a tensor
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        
        # Ensure it's 1D
        if wav.ndim > 1:
            wav = wav.squeeze()
        if wav.ndim > 1:
            wav = wav[0]

        sf.write(save_path, wav, samplerate=model.sr)

    logging.info(f"Audio saved at: {save_path}")

    return save_path


def build_ui(device=0):

    # Initialize model
    model = initialize_model(device=device)

    # Define callback function for voice cloning
    def voice_clone(text, prompt_wav_upload, prompt_wav_record):
        """
        Gradio callback to clone voice using text and optional prompt speech.
        - text: The input text to be synthesised.
        - prompt_wav_upload/prompt_wav_record: Audio files used as reference.
        """
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record

        audio_output_path = run_tts(
            text,
            model,
            prompt_speech=prompt_speech
        )
        return audio_output_path

    with gr.Blocks() as demo:
        # Use HTML for centered title
        gr.HTML('<h1 style="text-align: center;">Chatterbox TTS</h1>')
        
        # Voice Clone Tab (main functionality)
        gr.Markdown(
            "### Upload reference audio or recording for voice cloning"
        )

        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources="upload",
                type="filepath",
                label="Choose the prompt audio file, ensuring the sampling rate is no lower than 16kHz.",
            )
            prompt_wav_record = gr.Audio(
                sources="microphone",
                type="filepath",
                label="Record the prompt audio file.",
            )

        with gr.Row():
            text_input = gr.Textbox(
                label="Text to synthesize", 
                lines=3, 
                placeholder="Enter text here",
                value="Hello, this is a test of the Chatterbox voice cloning system."
            )

        audio_output = gr.Audio(
            label="Generated Audio", autoplay=True, streaming=True
        )

        generate_button = gr.Button("Generate Voice Clone", variant="primary")

        generate_button.click(
            voice_clone,
            inputs=[
                text_input,
                prompt_wav_upload,
                prompt_wav_record,
            ],
            outputs=[audio_output],
        )
        
        # Add information about Chatterbox
        gr.Markdown("""
        ### About Chatterbox TTS
        - **Voice Cloning**: Upload or record a reference audio to clone any voice
        - **High Quality**: Advanced neural TTS with natural prosody
        - **Fast Generation**: Optimized for real-time synthesis
        - Leave audio fields empty to use the default Chatterbox voice
        """)

    return demo


def parse_arguments():
    """
    Parse command-line arguments such as device ID and server settings.
    """
    parser = argparse.ArgumentParser(description="Chatterbox TTS Gradio server.")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU device to use (e.g., 0 for cuda:0)."
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server host/IP for Gradio app."
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port for Gradio app."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Build the Gradio demo by specifying the GPU device
    demo = build_ui(
        device=args.device
    )

    # Launch Gradio with the specified server name and port
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port
    )