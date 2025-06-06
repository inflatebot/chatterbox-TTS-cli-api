This is a dirty hack of dogarrowtype's [Spark-TTS-cli-api](https://github.com/dogarrowtype/Spark-TTS-cli-api) which drops in ResembleAI's [Chatterbox](https://github.com/resemble-ai/chatterbox) in its place. It should more or less work exactly the same, so check its readme for installation instructions.

### Some notes:

- I used `uv` to install `Spark-TTS-cli-api` while working on this fork, and it works great. But the original guidance should work as well. (I just can't conscionably recommend `conda` in a world where `uv` exists.)
```sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run tts_server.py
```
- I've considered converting this to `pyproject.toml` and setting it up for the `uv tool` interface. But that would be a major break away from upstream, and I'd rather that dogarrowtype (or you!!) take this fork as an indication that it'd be pretty easy to generalize this code, AllTalk-style (but, affectionately, *hopefully in a more ergonomic manner than AllTalk!*)

- Claude 4.0 Sonnet did the actual conversion. It all works for me, but be mindful that something could be subtly broken. 

- Manual downloading of the model like in dogarrowtype's server wasn't necessary; the model was downloaded on its own to HuggingFace's model cache. If you're comfortable using huggingface-cli to delete the Chatterbox model should the need arise, you can skip that step here.

- Chatterbox is a much simpler model than SparkTTS. It lacks the fine-grained control, but since the point of both of these projects is use of an API, and the OpenAI schema doesn't allow any ergonomic way of utilizing that control anyway, this should be fine. Essentially, if you have a voice that you're cloning, and don't need all the blinkenlights of Spark, Chatterbox is a great alternative (and, subjectively, I like its results better; hence all the hoopla I just went through.)

- The "CFG Weight" parameter for Chatterbox is mapped to the OpenAI "Speed" parameter, since CFG Weight has the effect of altering speaker pace. It's not a 1:1 though, so don't expect it to reliably make the speaker faster or slower. Yes, this is a hack. I don't have any better ideas.