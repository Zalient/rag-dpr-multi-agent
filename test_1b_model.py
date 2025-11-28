import time
import torch
import os
import sys
import logging
from transformers import GPTNeoForCausalLM, AutoTokenizer

# Configure the Logger - format: Time, Level (INFO/ERROR), Message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        sys.exit(1)
    logger.info(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")

    # Path to 1B model in scratch
    scratch_model_dir = "/iridisfs/scratch/zc3g23/gpt-neo-1.3B"

    if not os.path.exists(scratch_model_dir):
        logger.error(f"Model directory does not exist: {scratch_model_dir}")
        sys.exit(1)
    logger.info(f"Target model directory: {scratch_model_dir}")
    logger.info(f"Files found: {os.listdir(scratch_model_dir)}")

    # Load 1B model
    logger.info("Attempting to load tokenizer and model")
    try:
        tokenizer = AutoTokenizer.from_pretrained(scratch_model_dir, local_files_only=True)
        model = GPTNeoForCausalLM.from_pretrained(scratch_model_dir, local_files_only=True)
        logger.info("Tokenizer and model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model/tokenizer", exc_info=True)
        sys.exit(1)

    # Move to GPU
    logger.info("Moving model to GPU")
    model.to("cuda")
    logger.info("Model moved to GPU")

    # Run test generation
    prompt = "Hello, I am a 1.3B parameter model on Iridis X. My purpose is to"
    logger.info(f"Running test generation with prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    start_time = time.time()
    output = model.generate(inputs.input_ids, max_length=50, do_sample=True, top_k=50, top_p=0.9)
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Test generation completed in {duration:.3f}")

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Log generated text output
    logger.info("\n" + "="*30 + "\n" + generated_text + "\n" + "="*30)

    # Save output to /home
    log_file = "/iridisfs/home/zc3g23/rag-dpr-multi-agent/model_output.txt"
    try:
        with open(log_file, "w") as f:
            f.write(generated_text)
        logger.info(f"Generated text saved to: {log_file}")
    except Exception as e:
        logger.error(f"Failed to save output to file: {log_file}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()