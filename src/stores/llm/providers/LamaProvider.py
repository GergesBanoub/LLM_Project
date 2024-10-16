from ..LLMInterface import LLMInterface
import logging
from ..LLMEnums import LLaMAEnums  # Assuming this exists as in the original context
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.cuda.amp import autocast

class LLaMAProvider(LLMInterface):
    def __init__(self, model_name: str,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 200,
                 default_generation_temperature: float = 0.7):
        self.model_name = model_name
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model = None
        self.embedding_size = None

        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Check if a GPU is available and move the model to the correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        """
        Set the embedding model.
        """
        self.logger.info(f"Successfully set embedding model: {model_id} with size: {embedding_size}")
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size  # Keep track of the embedding size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def is_arabic(self, text):
        # Basic check for Arabic text
        return any('\u0600' <= char <= '\u06FF' for char in text)



    def generate_text(self, prompt: str, max_output_tokens: int = None, 
                  chat_history: list = [], temperature: float = None):

        if not self.generation_model_id:
            self.logger.error("Generation model for LLaMA was not set")
            return None
    
        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_generation_max_output_tokens
        temperature = temperature if temperature is not None else self.default_generation_temperature
    
        if self.is_arabic(prompt):
            prompt = "<|ar|> " + prompt
        else:
            prompt = "<|en|> " + prompt
    
        chat_history.append(self.construct_prompt(prompt=prompt, role=LLaMAEnums.USER.value))
    
        input_ids = self.tokenizer.encode(chat_history[-1]["content"], return_tensors='pt').to(self.device)
    
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.int64, device=self.device) if self.tokenizer.pad_token_id is not None else torch.ones(input_ids.shape, dtype=torch.int64, device=self.device)
    
        with autocast():  # Use mixed precision
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_output_tokens,
                top_k=50,
                top_p=0.9,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
    
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text[len(chat_history[-1]["content"]):]


    def embed_text(self, text: str, document_type: str = None):
        if not self.model:
            self.logger.error("Embedding model was not set")
            return None

        try:
            # Tokenize the input text
            input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(input_ids)

            # Get the last hidden states (or the logits depending on the model output)
            # Since CausalLM does not have last_hidden_state, use outputs.logits
            embeddings = outputs.logits[:, -1, :]  # Take the last token's logits as the embedding

            return embeddings.squeeze().cpu().numpy()  # Move tensor back to CPU and convert to numpy array
        except Exception as e:
            self.logger.error(f"An error occurred while generating embeddings: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}
