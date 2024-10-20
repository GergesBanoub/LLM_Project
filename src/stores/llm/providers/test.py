from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "/mnt/c/LLama/Llama-3.1-8B"  # Specify your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "what is the data governance in arabic please"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

try:
    output = model.generate(
        input_ids=input_ids,
        max_length=50,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
except Exception as e:
    print(f"Error during generation: {e}")
