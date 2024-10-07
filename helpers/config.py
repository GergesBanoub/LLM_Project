from pydantic_settings  import BaseSettings ,SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY: str
    
    class Config:  # Fix the typo (use Config with uppercase C)
        env_file = "/home/mini_rag_app/LLM_Project/.env"  # Path to your .env file

def get_settings():
    return Settings()

 
