# Spark-TTS-cli-api

This is a humanized fork of [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) that adds an OpenAI compatible text to speech API. This means you can actually use the amazing capability of Spark-TTS in real programs, instead of just a tech demo.

This fork is capable of processing an unlimited amount of text at once. It uses intelligent semantic text splitting to overcome the limitations of the original repo. It has quality of life improvements that make it suitable as a hands-off TTS provider. It will trim out excessive silence, and retry failed segments.

Generally, this smooths over all the bumps usually encountered with local text to speech models.

These improvements allow spark text to speech to be used as a seamless drop in replacement for OpenAI. This means it will work with programs like [SillyTavern](https://github.com/SillyTavern/SillyTavern) and others. It can be easily used to read stories, roleplay, chat, and so on.

It takes about 8.5 GB vram, small enough to fit and run at about 2x realtime on an RTX 3060. Anything larger will have an easy time running this model at much faster than real time.

This was built off the work of the original model authors, and especially the foundational work of [@AcTePuKc's script](https://github.com/SparkAudio/Spark-TTS/issues/10). This would not have been possible without [@AcTePuKc](https://github.com/AcTePuKc)'s inference script.

This project is fully functional, but not feature complete. I plan to add the ability to choose custom voices at inference time with the API.

Right now, it is designed to be used locally by one user, processing one request at a time. Multi-user (concurrent requests) generation works, but it is not as stable as it could be.

The program is a work in progress, but right now, pretty great!

## To-Do List

- [ ] Add the ability to have multiple voice cloning sources loaded and selectable by API.
- [ ] Switch to a more production stable HTTP hosting solution.

---

## Install
**Clone and Install**

(Windows instructions not updated for this repo, but close enough.)
If you're on Windows, please refer to the [Windows Installation Guide](https://github.com/SparkAudio/Spark-TTS/issues/5).  
*(Thanks to [@AcTePuKc](https://github.com/AcTePuKc) for the detailed Windows instructions!)*

Linux install:

Clone the repo
``` sh
git clone https://github.com/dogarrowtype/Spark-TTS-cli-api
cd Spark-TTS
```

- Needs `python3.11` or `python3.12`

Install with conda or pip:

Conda install method:
- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:
``` sh
conda create -n sparktts -y python=3.12
conda activate sparktts
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
#pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

Pip install method:
``` sh
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
#pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

**Model Download**

Download via huggingface-cli command:
```sh
huggingface-cli download SparkAudio/Spark-TTS-0.5B --local-dir "pretrained_models/Spark-TTS-0.5B"
```

Download via python code and huggingface_hub (requires huggingface_cli to be installed):
```python
from huggingface_hub import snapshot_download

snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B")
```

Or download via git clone:
```sh
mkdir -p pretrained_models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B
```

**Basic Usage**

To start the server on port 9991:
``` sh
python ./tts_server.py --model_dir "pretrained_models/Spark-TTS-0.5B/" --prompt_audio "voice_samples/female2.wav" --port 9991
```

Alternatively, you can run a command line interface that will read text input：

``` sh
python ./cli/tts_cli.py --text "Hello, nice to meet you." --prompt_audio "voice_samples/female2.wav"
```

Or have it **read text from a file**:
``` sh
python ./cli/tts_cli.py --text_file "[path to your .txt file]" --prompt_audio "voice_samples/female2.wav"
```

This will save the output to `examples/results`

**Switching the voice**

Several working (and consistent) voices are provided in `voice_samples`. To switch to another voice, simply change out `female2.wav` for another voice.

**Provided voice descriptions**
- female1.wav: melancholy, rainy
- female2.wav: confident, expressive
- female3.wav: anime VA, cutesy, kinda sad, amazed
- female4.wav: clean, light, a touch of mystery
- male1.wav: cute, super expressive, adventurous
- male2.wav: determined, focused
- male3.wav: surprised, best for short bursts
- male4.wav: deep, rich, noir, serious


**Voice cloning**

To clone another voice, provide a 5 to 20 second voice clip of clear audio, and hope for the best. A single long sentence can work great.

Cloned voices seem to work better when using a second generation clone. Meaning, clone a voice, try generating a few different times, then use the result you like as a clone for future generations.

Simply put:
- Generate a few samples with your desired clone voice
- Find one that has the correct accent/sound
- Use that perfect result as clone input to get consistent long generations

**Random voices**
If you don't specify a clone source, you must specify `--gender`. You'll get a random voice for each chunk. This is useful to find new voices, and use those as clone sources. However, for normal usage, I don't suggest leaving it on random as the output will be chaotic.

---

## SillyTavern usage

SillyTavern Settings
![Image 1](src/figures/sillytavern-settings.png) 
Make SillyTavern TTS settings (in the "Extensions" menu at the top) match this screenshot. "Narrate by paragraphs (when not streaming)" is very important to reduce latency.

---

## Original Readme (partial)

### Overview

Spark-TTS is an advanced text-to-speech system that uses the power of large language models (LLM) for highly accurate and natural-sounding voice synthesis. It is designed to be efficient, flexible, and powerful for both research and production use.

### Key Features

- **Simplicity and Efficiency**: Built entirely on Qwen2.5, Spark-TTS eliminates the need for additional generation models like flow matching. Instead of relying on separate models to generate acoustic features, it directly reconstructs audio from the code predicted by the LLM. This approach streamlines the process, improving efficiency and reducing complexity.
- **High-Quality Voice Cloning**: Supports zero-shot voice cloning, which means it can replicate a speaker's voice even without specific training data for that voice. This is ideal for cross-lingual and code-switching scenarios, allowing for seamless transitions between languages and voices without requiring separate training for each one.
- **Bilingual Support**: Supports both Chinese and English, and is capable of zero-shot voice cloning for cross-lingual and code-switching scenarios, enabling the model to synthesize speech in multiple languages with high naturalness and accuracy.
- **Controllable Speech Generation**: Supports creating virtual speakers by adjusting parameters such as gender, pitch, and speaking rate.

---


## ⚠️ Usage Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities.

- Ensure compliance with local laws and regulations when using this model and uphold ethical standards.

- The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.
