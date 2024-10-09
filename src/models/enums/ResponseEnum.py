from enum import Enum

class ResponseSignal(Enum):
    FILE_VALIDATED_SUUCESS = "File Validate Successfully"
    FilE_TYPE_NOT_SUPPORTED ="File Type Not Supported"
    FILE_SIZE_EXCEEDED = "File Size Exceeded"
    FILE_UPLOAD_SUCCESS ="File Upload Success"
    FILE_UPLOAD_FAILED ="File Upload Failed"
    PROCESSING_SUCCESS ="processing_success"
    PROCESSING_Failed ="processing_failed"