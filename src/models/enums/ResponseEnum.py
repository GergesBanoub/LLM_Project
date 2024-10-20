from enum import Enum

class ResponseSignal(Enum):
    FILE_VALIDATED_SUUCESS = "File Validate Successfully"
    FilE_TYPE_NOT_SUPPORTED ="File Type Not Supported"
    FILE_SIZE_EXCEEDED = "File Size Exceeded"
    FILE_UPLOAD_SUCCESS ="File Upload Success"
    FILE_UPLOAD_FAILED ="File Upload Failed"
    PROCESSING_SUCCESS ="processing_success"
    PROCESSING_Failed ="processing_failed"
    NO_FILES_ERROR="Not_Found_Files"
    PROJECT_NOT_FOUND_ERROR = "project_not_found"
    INSERT_INTO_VECTORDB_ERROR = "insert_into_vectordb_error"
    INSERT_INTO_VECTORDB_SUCCESS = "insert_into_vectordb_success"
    VECTORDB_COLLECTION_RETRIEVED = "vectordb_collection_retrieved"
    VECTORDB_SEARCH_ERROR = "vectordb_search_error"
    VECTORDB_SEARCH_SUCCESS = "vectordb_search_success"
    RAG_ANSWER_ERROR = "rag_answer_error"
    RAG_ANSWER_SUCCESS = "rag_answer_success"
