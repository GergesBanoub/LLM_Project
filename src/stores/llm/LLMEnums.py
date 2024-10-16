from enum import Enum

class LLMEnums(Enum):
    OPENAI="OPENAI"
    COHERE ="COHERE"
    OLAMA="OLAMA"
class OpenAIEnums(Enum):
    SYSTEM="system"
    USER="user"
    ASSISTANT="assistant"
class CoHereEnums(Enum):
    SYSTEM="SYSTEM"
    USER="USER"
    ASSISTANT="CHATBOT"
    DOCUMENT="search_document"
    QUERY="search_query"
class DocumnetTypeEnum(Enum):
    DOCUMENT="document"
    QUERY="query"

class LLaMAEnums(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

