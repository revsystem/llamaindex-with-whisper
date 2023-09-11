from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate

# QAシステムプロンプト
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "あなたは世界中で信頼されているQAシステムです。\n"
        "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
        "従うべきいくつかのルール:\n"
        "1. 回答内で指定されたコンテキストを直接参照しないでください。\n"
        "2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、またはそれに類するような記述は避けてください。"
    ),
    role=MessageRole.SYSTEM,
)

# QAプロンプトテンプレートメッセージ
TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "コンテキスト情報は以下のとおりです。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "事前知識ではなくコンテキスト情報を考慮して、クエリに答えます。\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

# TreeSummarizeプロンプトメッセージ
TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "複数のソースからのコンテキスト情報を以下に示します。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "予備知識ではなく、複数のソースからの情報を考慮して、質問に答えます。\n"
            "疑問がある場合は、「情報無し」と答えてください。\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

# SinglePydSelectプロンプトテンプレートの定義
SINGLE_PYD_SELECT_PROMPT_TMPL = (
    "いくつかの選択肢を番号付きリストで示します。(1〜{num_choices})\n"
    "リスト内の各項目は要約に対応します。\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "予備知識ではなく上記の選択肢のみを使用して、"
    "以下の質問に最も関連する番号とその理由を答えてください。\n"
    "'{query_str}'\n"
)

# チャットQAプロンプト
CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

# TreeSummarizeプロンプト
CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
    message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS
)
