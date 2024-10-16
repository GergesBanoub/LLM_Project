import logging
from .LamaProvider import LLaMAProvider  # Replace with your actual module path

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Replace 'your_model_name' with the name of your LLaMA model
    model_name = "/home/dm-desktop/OLama/Llama-3.1-8B"  # e.g., 'meta-llama/Llama-2-7b-chat-hf'
    embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"  # Replace with your desired embedding model
    generation_model_id = "your-generation-model-id"  # Replace with your actual generation model ID

    try:
        # Initialize LLaMAProvider
        llama_provider = LLaMAProvider(model_name=model_name)
        
        # Set the generation model ID
        llama_provider.set_generation_model(generation_model_id)  # Set the generation model
        
        # Set the embedding model
        llama_provider.set_embedding_model(model_id=embedding_model_id, embedding_size=384)  # Set the embedding model with an example size
        
        logging.info("LLaMAProvider initialized successfully.")

        # Test text generation
        prompt = "ماهي اهداف استراتيجيه 20230 بالمملكه العربيه السعودية"
        logging.info(f"Generating text for prompt: '{prompt}'")
        generated_text = llama_provider.generate_text(prompt)
        logging.info("Text generation completed.")
        print("Generated Text:")
        print(generated_text)

        # Test embedding
        text_to_embed = "AI can improve diagnosis accuracy."
        logging.info(f"Embedding text: '{text_to_embed}'")

        # Generate the embeddةing
        embedding = llama_provider.embed_text(text_to_embed)

        if embedding is not None:
            logging.info("Embedding completed successfully.")
            print("Embedding Shape:", len(embedding))  # Print the length of the embedding vector
            print("Embedding Vector:", embedding)  # Optionally print the embedding vector
        else:
            logging.error("Embedding returned None.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()