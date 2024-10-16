from ..LLMInterface import LLMInterface
import logging
from openai import OpenAI  # Corrected class name
from ..LLMEnums import OpenAIEnums

class OpenAIProvider(LLMInterface):
    def __init__(self, api_key: str, api_url: str = None, 
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        self.api_key = api_key
        self.api_url = api_url
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None
        self.client = OpenAI(api_key=self.api_key, api_url=self.api_url)
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
    def process_text(self,text:str):
        return text[:self.default_input_max_characters].strip()
    def is_arabic(self, text):
    # Basic check for Arabic text
        return any('\u0600' <= char <= '\u06FF' for char in text)

    


    def generate_text(self, prompt: str, max_output_tokens: int = None, 
                      chat_history: list = [], temperature: float = None):
        
        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_generation_max_output_tokens
        temperature = temperature if temperature is not None else self.default_generation_temperature
        
        chat_history.append(self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value))

        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=chat_history,
            max_tokens=max_output_tokens,
            temperature=temperature  # Corrected spelling
        )

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("Error while generating text with OpenAI")
            return None

        return response.choices[0].message["content"]

    def embed_text(self, text: str, document_type: str = None):  # Corrected spelling
        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None
        
        response = self.client.embedding.create(model=self.embedding_model_id, input=text)

        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error("Error while embedding text with OpenAI")
            return None  # Changed to return None to indicate an error

        return response.data[0].embedding  # Fixed return statement

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}
    
    

