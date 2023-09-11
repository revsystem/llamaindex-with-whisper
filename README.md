# llamaindex-with-whisper

Extract audio from a video and transcribe with Whisper, indexing with Llma-index, and summarize with GPT-4.

## Usage

### Extract audio from the video file and transcribe

```console
python3 ./transcriptin.py -f ./sample.mp4
```

Then, transcripted documents will be in ./data/documents

### Execute query

```console
python3 ./transcriptin.py
```

Then, index data will be in ./data/indexes/index.json if you don't have any indexes.

```console
Input query: <INPUT_YOUR_QUERY_ABOUT_TRANSCRIPTED_TEXT>
```

### Response

We can get a answer.

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

llama-index==0.8.22
openai==0.28.0
pydub==0.25.1
tiktoken==0.4.0
```

### Requirement OpenAI API Key

Set your API Key to environment variables or shell dotfile like '.zshenv':

```console
export OPENAI_API_KEY= 'YOUR_OPENAI_API_KEY'
```

## Reference

- [Router Query Engine](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/RouterQueryEngine.html)
- [Experimenting LlamaIndex RouterQueryEngine with Document Management](https://betterprogramming.pub/experimenting-llamaindex-routerqueryengine-with-document-management-19b17f2e3a32)
- [LlamaIndex の RouterQueryEngine を試す](https://note.com/npaka/n/n0a068497ac96)

## Licensing

This software includes the work that is distributed in the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
Original source code is [wenqiglantz/DevSecOpsKB-LlamaIndex-LangChain-OpenAI](https://github.com/wenqiglantz/DevSecOpsKB-LlamaIndex-LangChain-OpenAI/tree/main/DevSecOpsKB-router-query-engine-document-management)
