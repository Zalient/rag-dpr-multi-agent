import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import os
import sys

def main():
    # --- 1. Define Scratch Path (as per supervisor) ---
    # Get username from environment, default to your user ID
    username = os.environ.get('USER')
    if not username:
        print("FATAL: Could not get $USER from environment.")
        sys.exit(1)
        
    scratch_path = f'/local/scratch/users/{username}'
    output_file = os.path.join(scratch_path, 'iridis_test_SUCCESS.txt')
    
    print(f"--- Iridis Test Job ---")
    print(f"User: {username}")
    print(f"Saving success log to: {output_file}")

    try:
        os.makedirs(scratch_path, exist_ok=True)
        print(f"Successfully ensured scratch directory exists.")
    except Exception as e:
        print(f"\n*** CRITICAL ERROR: Could not create scratch directory. ***")
        print(f"Error: {e}")
        sys.exit(1)
    
    # --- 2. Check GPU (Critical Test) ---
    print(f"\nPyTorch Version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("\n*** CRITICAL ERROR: CUDA IS NOT AVAILABLE. ***")
        print("Check your conda env installation on Iridis.")
        sys.exit(1)
    
    print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")

    # --- 3. Load 1B Model (as per supervisor) ---
    model_name = "EleutherAI/gpt-neo-1.3B"
    print(f"\nLoading model: {model_name}...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    except Exception as e:
        print(f"*** ERROR: Failed to download/load model. ***")
        print(f"Error: {e}")
        sys.exit(1)
        
    print("Model loaded successfully.")

    # --- 4. Move to GPU (Tests PyTorch-CUDA Link) ---
    model.to('cuda')
    print("Model successfully moved to 'cuda' (GPU).")

    # --- 5. Run Test Inference ---
    print("Running test inference...")
    prompt = "Hello, I am a 1.3B parameter model on Iridis X. My purpose is to"
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    output = model.generate(inputs.input_ids, max_length=50, do_sample=True, top_k=50, top_p=0.9)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("\n--- Test Generation Output ---")
    print(generated_text)
    print("------------------------------")

    # --- 6. Write Success File to Scratch (as per supervisor) ---
    try:
        with open(output_file, 'w') as f:
            f.write(f"SUCCESS: Test job completed.\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Generated Text: {generated_text}\n")
        print(f"\nSuccessfully wrote success file to scratch drive.")
    except Exception as e:
        print(f"\n*** ERROR: Failed to write to scratch drive at {scratch_path} ***")
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Test Job Finished Successfully ---")

if __name__ == "__main__":
    main()