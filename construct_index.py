import logging
import os
from typing import Callable, List

import nest_asyncio
import nltk
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    LangchainEmbedding,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    set_global_service_context,
)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.indices.loading import load_index_from_storage
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor

# from llama_index.llms import OpenAI
from llama_index.logger.base import LlamaLogger
from llama_index.node_parser import SentenceWindowNodeParser, SimpleNodeParser

# from llama_index.prompts import Prompt
from llama_index.query_engine.router_query_engine import RouterQueryEngine

# from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.text_splitter import TokenTextSplitter
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.utils import get_cache_dir

from constants import FILEPATH_CACHE_INDEX, FOLDERPATH_DOCUMENTS, VARIABLES_FILE
from prompt_tmpl import (
    CHAT_TEXT_QA_PROMPT,
    CHAT_TREE_SUMMARIZE_PROMPT,
    DEFAULT_CHOICE_SELECT_PROMPT,
    SINGLE_PYD_SELECT_PROMPT_TMPL,
    SUMMARY_QUERY,
)

# enable asynchronous processing
nest_asyncio.apply()


# Tokenizer for Japanese
def split_by_sentence_tokenizer() -> Callable[[str], List[str]]:
    cache_dir = get_cache_dir()
    nltk_data_dir = os.environ.get("NLTK_DATA", cache_dir)

    # update nltk path for nltk so that it finds the data
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    sent_detector = nltk.RegexpTokenizer("[^　！？。]*[！？。.\n]")
    return sent_detector.tokenize


def get_service_context() -> ServiceContext:
    logging.info("Get Service Context.")
    # for debug
    llama_debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([llama_debug_handler])

    # Customize the LLM

    # Language model for obtaining textual responses (Completion)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0613",
        temperature=0.8,
        max_tokens=1024,
    )

    # llm = OpenAI(
    #     model="gpt-4-0613",
    #     temperature=0,
    #     max_tokens=1024,
    # )

    llm_predictor = LLMPredictor(llm=llm)

    # Customize behavior of splitting into chunks
    # text_splitter = TokenTextSplitter(
    #     separator=" ",
    #     chunk_size=256,
    #     chunk_overlap=20,
    #     backup_separators=["\n\n", "\n", "。", "。 ", "、", "、 "],
    #     tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
    # )

    # Split text into chunks and create nodes
    # node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
    node_parser = SentenceWindowNodeParser.from_defaults(
        sentence_splitter=split_by_sentence_tokenizer(),
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

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


def load_variables():
    global list_id, vector_id

    logging.info("check if the variable file exists.")
    if os.path.isfile(VARIABLES_FILE):
        with open(VARIABLES_FILE, "r", encoding="utf-8") as file:
            logging.info("read the values from the file")
            values = file.read().split(",")
            list_id = values[0]
            vector_id = values[1]


def save_variables():
    global list_id, vector_id

    # write the values to the file.
    with open(VARIABLES_FILE, "w", encoding="utf-8") as file:
        file.write(f"{list_id},{vector_id}")


def construct_index():
    global list_id, vector_id

    directory_reader = SimpleDirectoryReader(
        FOLDERPATH_DOCUMENTS,
        filename_as_id=True,
    )

    service_context = get_service_context()
    set_global_service_context(service_context)
    documents = directory_reader.load_data()
    os.makedirs(
        FILEPATH_CACHE_INDEX,
        exist_ok=True,
    )

    try:
        storage_context = StorageContext.from_defaults(persist_dir=FILEPATH_CACHE_INDEX)
        summary_index = load_index_from_storage(
            storage_context=storage_context,
            index_id=list_id,
        )
        vector_index = load_index_from_storage(
            storage_context=storage_context,
            index_id=vector_id,
        )
        logging.debug(
            "vector_index: %s", vector_index.storage_context.docstore.docs.items()
        )
        logging.info("list_index and vector_index loaded")

    except FileNotFoundError:
        logging.info("storage context not found. Add nodes to docstore")
        storage_context = StorageContext.from_defaults()

        # define response_synthesizer

        # construct list_index and vector_index from storage_context.
        summary_index = DocumentSummaryIndex.from_documents(
            documents,
            storage_context=storage_context,
            response_synthesizer=get_response_synthesizer(
                response_mode="tree_summarize",
                use_async=True,
                text_qa_template=CHAT_TEXT_QA_PROMPT,  # QAプロンプト
                summary_template=CHAT_TREE_SUMMARIZE_PROMPT,  # TreeSummarizeプロンプト
            ),
            summary_query=SUMMARY_QUERY,
        )

        vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )

        # persist both indexes to disk
        summary_index.storage_context.persist(persist_dir=FILEPATH_CACHE_INDEX)
        vector_index.storage_context.persist(persist_dir=FILEPATH_CACHE_INDEX)

        # update the global variables of list_id and vector_id
        list_id = summary_index.index_id
        vector_id = vector_index.index_id

        save_variables()

    # define list_query_engine and vector_query_engine
    list_query_engine = summary_index.as_query_engine(
        choice_select_prompt=DEFAULT_CHOICE_SELECT_PROMPT,  # ChoiceSelectプロンプト
        response_synthesizer=get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=True,
            summary_template=CHAT_TREE_SUMMARIZE_PROMPT,  # TreeSummarizeプロンプト
        ),
    )

    vector_query_engine = vector_index.as_query_engine(
        similarity_top_k=5,
        response_synthesizer=get_response_synthesizer(
            response_mode="compact",
            text_qa_template=CHAT_TEXT_QA_PROMPT,
        ),
        node_processors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
        text_qa_prompt=CHAT_TEXT_QA_PROMPT,
    )

    # build list_tool and vector_tool
    list_tool = QueryEngineTool.from_defaults(
        query_engine=list_query_engine,
        description="テキストの要約に役立ちます｡",
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="テキストから特定のコンテキストを取得するのに役立ちます。",
    )

    # construct RouteQueryEngine
    query_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(
            prompt_template_str=SINGLE_PYD_SELECT_PROMPT_TMPL,
        ),
        query_engine_tools=[
            list_tool,
            vector_tool,
        ],
    )

    # run refresh_ref_docs function to check for document updates
    list_refreshed_docs = summary_index.refresh_ref_docs(
        documents, update_kwargs={"delete_kwargs": {"delete_from_docstore": True}}
    )
    print(list_refreshed_docs)
    print("Number of newly inserted/refreshed docs: ", sum(list_refreshed_docs))

    summary_index.storage_context.persist(persist_dir=FILEPATH_CACHE_INDEX)
    logging.info("list_index refreshed and persisted to storage.")

    vector_refreshed_docs = vector_index.refresh_ref_docs(
        documents, update_kwargs={"delete_kwargs": {"delete_from_docstore": True}}
    )
    print(vector_refreshed_docs)
    print("Number of newly inserted/refreshed docs: ", sum(vector_refreshed_docs))

    vector_index.storage_context.persist(persist_dir=FILEPATH_CACHE_INDEX)
    logging.info("vector_index refreshed and persisted to storage.")

    return query_engine


def query_with_index(user_input: str):
    query_engine = construct_index()
    response = query_engine.query(user_input)

    return response
