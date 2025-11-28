from transformers import AutoTokenizer, GPTNeoForCausalLM

model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Save locally
tokenizer.save_pretrained("./gpt-neo-1.3B")
model.save_pretrained("./gpt-neo-1.3B")