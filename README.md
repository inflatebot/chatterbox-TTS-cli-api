This is a dirty hack of dogarrowtype's [Spark-TTS-cli-api](https://github.com/dogarrowtype/Spark-TTS-cli-api) which drops in ResembleAI's [Chatterbox](https://github.com/resemble-ai/chatterbox) in its place. It should more or less work exactly the same, and you can read its readme here (TODO: put a hotlink here), and should for setup instructions.

### Some notes:

- I used `uv` to install `Spark-TTS-cli-api` while working on this fork, and it works great. But the original guidance should work as well. (I just can't conscionably recommend `conda` in a world where `uv` exists.)
```sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run tts_server.py
```
- When I can be bothered, I'd like to convert this to a Python project and set it up for the `uv tool` interface. But that would be a major break away from upstream, and I'd rather that dogarrowtype (or you!!) take this fork as an indication that it'd be pretty easy to generalize this code, AllTalk-style (but hopefully in a more ergonomic manner than AllTalk!)

- Claude 4.0 Sonnet did the actual conversion. It all works for me, but be mindful that something could be subtly broken. 

- Manual downloading of the model like in dogarrowtype's server wasn't necessary; the model was downloaded on its own.