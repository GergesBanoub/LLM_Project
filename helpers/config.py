from pydantic_settings  import BaseSettings ,SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY: str
    FILE_ALLOWED_TYPES:list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE : int
    
    class Config:  # Fix the typo (use Config with uppercase C)
        env_file = ".env"  # Path to your .env file

def get_settings():
    return Settings()

 
