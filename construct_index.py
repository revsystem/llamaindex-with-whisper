import logging
import os

import tiktoken
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    LangchainEmbedding,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.indices.list.base import ListRetrieverMode

# from langchain.chat_models import ChatOpenAI
from llama_index.llms import OpenAI
from llama_index.logger.base import LlamaLogger
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import Prompt
from llama_index.text_splitter import TokenTextSplitter

from constants import FILEPATH_CACHE_INDEX, FOLDERPATH_DOCUMENTS


def get_service_context() -> ServiceContext:
    logging.info("Get Service Context.")
    # for debug
    llama_debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([llama_debug_handler])

    # Customize the LLM

    # Language model for obtaining textual responses (Completion)
    # llm_predictor = LLMPredictor(
    #     llm=ChatOpenAI(
    #         model="gpt-4",
    #         temperature=0.8,
    #         max_tokens=6000,
    #     )
    # )

    llm = OpenAI(
        model="gpt-4-0613",
        temperature=0,
        max_tokens=3000,
    )

    llm_predictor = LLMPredictor(llm=llm)

    # Customize behavior of splitting into chunks
    text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=256,
        chunk_overlap=20,
        backup_separators=["\n\n", "\n", "。", "。 ", "、", "、", " ", ""],
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
    )

    # Split text into chunks and create nodes
    node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)

    # Customize Embedded Models
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    )

    # Split text to meet token count limit on LLM side
    prompt_helper = PromptHelper(
        context_window=3900,  # default
        num_output=256,  # default
        chunk_overlap_ratio=0.1,  # default
        chunk_size_limit=None,  # default
    )

    service_context = ServiceContext.from_defaults(
        callback_manager=callback_manager,
        llama_logger=LlamaLogger(),
        node_parser=node_parser,
        embed_model=embed_model,
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
    )

    return service_context


def construct_index() -> VectorStoreIndex:
    directory_reader = SimpleDirectoryReader(
        FOLDERPATH_DOCUMENTS,
    )
    documents = directory_reader.load_data()
    os.makedirs(
        FILEPATH_CACHE_INDEX,
        exist_ok=True,
    )

    storage_context = StorageContext.from_defaults()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    index.storage_context.persist(FILEPATH_CACHE_INDEX)

    return index


def query_with_index(user_input: str):
    logging.info("Loading exist documents.")
    storage_context = StorageContext.from_defaults(persist_dir=FILEPATH_CACHE_INDEX)

    index = load_index_from_storage(storage_context)

    # set Q&A Prompt
    qa_prompt_file = "qa_prompt_ja.txt"
    with open(qa_prompt_file, "r", encoding="utf-8") as file:
        qa_prompt_tmpl = file.read()

    qa_prompt = Prompt(qa_prompt_tmpl)

    # make QueryEngine, query with index
    query_engine = index.as_query_engine(
        # Use customized prompts
        text_qa_template=qa_prompt,
        # Select nodes using embedded vectors; enabled by default for VectorStoreIndex.
        retriever_mode=ListRetrieverMode.EMBEDDING,
        # Select nodes in order of similarity to the query
        similarity_top_k=5,
        # Output responses in stream format.
        streaming=True,
    )
    response = query_engine.query(user_input)

    return response
