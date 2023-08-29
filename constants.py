"""Set of constants."""
import os
import openai

FOLDERPATH_DOCUMENTS = os.path.join("data", "documents")
FOLDERPATH_INDEX = os.path.join("data", "indexes")
FILEPATH_CACHE_INDEX = os.path.join(FOLDERPATH_INDEX, "index.json")

# target file size = 25MB
DEFAULT_TARGET_FILE_SIZE = 25000000

openai.api_key = os.getenv("OPENAI_API_KEY")
