import os
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

BASE_DIR = "/iridisfs/scratch/zc3g23"
INPUT_FILE = os.path.join(BASE_DIR, "datasets/cross_encoder_mbert_top10_results_nq.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "datasets/mbert_generated_answers_nq.json")
PREDICTOR_PATH = os.path.join(BASE_DIR, "models/llama_3_1_8b_instruct")

def main():
    with open(INPUT_FILE, 'r') as f:
        tasks = json.load(f)

    print("Loading LLM model from: {PREDICTOR_PATH} with vLLM...")
    # Load tokenizer separately to apply the chat template
    tokenizer = AutoTokenizer.from_pretrained(PREDICTOR_PATH, local_files_only=True)
    
    # Initialise vLLM
    llm = LLM(
        model=PREDICTOR_PATH, 
        quantization="bitsandbytes", 
        load_format="bitsandbytes", 
        enforce_eager=True, 
        max_model_len=4096
    )

    sys_prompt = "You are a concise expert. Answer the question using ONLY the provided context. If the answer is not in the context, say 'I do not know'. DO NOT provide explanations. DO NOT use outside knowledge."

    print("Formatting prompts...")
    prompts = []
    for task in tasks:
        # Join the list of top 10 contexts into a single formatted string
        joined_context = "\n\n".join(task['context_list'])
        
        prompt_messages = [
            {"role": "system", "content": sys_prompt}, 
            {"role": "user", "content": f"Context:\n{joined_context}\n\nQuestion: {task['question']}"}
        ]
        prompts.append(tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))

    print(f"Generating 3 answers per question for {len(prompts)} tasks...")
    
    # t=1.0 and n=3 to generate 3 diverse answers per prompt
    sampling_params = SamplingParams(temperature=1.0, max_tokens=256, n=3)

    outputs = llm.generate(prompts, sampling_params)

    # Map the generated texts back to the original tasks list
    for i, output in enumerate(outputs):
        generated_variants = [completion.text.strip() for completion in output.outputs]
        tasks[i]['initial_answers'] = generated_variants

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(tasks, f, indent=4)
        
    print(f"Saved initial answers to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()