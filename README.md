# llamaindex-with-whisper

Extract audio from a video and transcribe with Whisper, indexing with Llma-index, and summarize with GPT-4.

## Usage

### Extract audio from the video file and transcribe

```console
python3 ./transcriptin.py -f ./sample.mp4
```

Then, transcripted documents will be in ./data/documents

### Vectorize the transcribed data

```console
python3 ./transcriptin.py -i
```

Then, index data will be in ./data/indexes/index.json

### Execute query

```console
python3 ./transcriptin.py
```

```console
Input query: <INPUT_YOUR_QUERY_ABOUT_TRANSCRIPTED_TEXT>
```

### Response

We can get a streaming answer like the ChatGPT.

```console
==========
Query:
<QUERY_YOU_INPUTED>
Answer:
<ANSWER_FROM_AI>
==========

node.node.id_='876f8bdb-xxxx-xxxx-xxxx-xxxxxxxxxxxx', node.score=0.8484xxxxxxxxxxxxxx
----------

Cosine Similarity:
0.84xxxxxxxxxxxxxx

Reference text:
<THE_PART_AI_REFERRED_TO>
```

#### When you exit the console, input 'exit'.

```console
Input query: exit
```

## Setup

### Recommended System Requirements

- Python 3.10 or higher.

### Setup venv environment

To create a venv environment and activate:

```console
python3 -m venv .venv
source .venv/bin/activate
```

To deactivate:

```console
deactivate
```

### Setup Python environment

```console
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

The main libraries installed are as follows:

```console
pip freeze | grep -e "openai" -e "pydub" -e "llama-index" -e "sentence_transformers" -e "tiktoken"

llama-index==0.8.11.post3
openai==0.27.9
pydub==0.25.1
tiktoken==0.4.0
```

### Requirement OpenAI API Key

Set your API Key to environment variables or shell dotfile like '.zshenv':

```console
export OPENAI_API_KEY= 'YOUR_OPENAI_API_KEY'
```

## Reference

- [OpenAIのWhisper APIの25MB制限に合うような調整を検討する](https://dev.classmethod.jp/articles/openai-api-whisper-about-data-limit/)
- [Converting Speech to Text with the OpenAI Whisper API](https://www.datacamp.com/tutorial/converting-speech-to-text-with-the-openAI-whisper-API)
- [Speech to text](https://platform.openai.com/docs/guides/speech-to-text)
- [Llamaindex を用いた社内文書の ChatGPT QA ツールをチューニングする](https://recruit.gmo.jp/engineer/jisedai/blog/llamaindex-chatgpt-tuning/)