import os
import json
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = "/iridisfs/scratch/zc3g23"
INPUT_FILE = os.path.join(BASE_DIR, "datasets/mbert_generated_answers_nq.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "datasets/mbert_judged_answers_nq.json")
JUDGE_PATH = os.path.join(BASE_DIR, "models/deepseek_r1_distill_qwen_32b")

SAVE_EVERY = 25
DEBATER_TOKENS = 256 
CHIEF_TOKENS = 256    
CORRECTOR_TOKENS = 256

# ChatEval framework - multiple personas
SYSTEM_PROMPTS = {
    "Scientist": "You are Scientist, one of the referees. You are a professional engaged in systematic study who possesses a strong background in the scientific method, critical thinking, and problem-solving. Evaluate the candidate answers strictly against the provided context. Determine which response is the most factually accurate. 'I do not know' is the correct answer if the context lacks the information. DO NOT use outside knowledge. Output concise critique.",
    "Critic": "You are Critic, one of the referees. You will check fluent writing, clear sentences, and good wording. Your job is to rigorously question the candidates to make sure they are well-considered and strictly supported by the context. Offer a preference or alternative if responses are at the same level. DO NOT use outside knowledge. Output concise critique.",
    "GeneralPublic": "You are General Public, one of the referees. You are interested in the story and looking for updates. Please think critically by yourself and note that it is your responsibility to choose which of the candidate answers is better based strictly on the provided context. DO NOT use outside knowledge. Output concise critique.",
    "NewsAuthor": "You are News Author, one of the referees. You will focus heavily on consistency with the original article (the context). Help determine which response is the most faithful to the text without hallucinating outside knowledge. DO NOT use outside knowledge. Output concise critique.",
    "Summariser": "You are the Summariser. Read the question, context, candidate answers, and the critiques from the 4 referees. Form a consensus. Select the best answer from the candidates. Determine if this best answer is strictly supported by the context or if it is a Hallucination. Output EXACT JSON: {\"best_index\": <int>, \"verdict\": \"Supported\" or \"Hallucination\"}",
    "Corrector": "You are a Corrector. The previous best answer was flagged as a hallucination. Review the context, question, and the critiques. Generate a NEW, corrected answer using ONLY the context. If the answer is not in the context, strictly say 'I do not know'. DO NOT provide explanations."
}

def extract_json(text):
    # Extracts JSON, ignoring Deepseek"s <think> tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "best_index" not in data: data["best_index"] = 0
            return data
        except:
            pass
    return {"best_index": 0, "verdict": "Error"}

def strip_think_tags(text):
    # Removes DeepSeek reasoning steps from the final generated answers
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def run_council_batch(tasks_chunk, llm, tokenizer, target_key):
    # Processes a chunk of tasks through the 4 personas and the Summariser
    debater_prompts = []
    
    for task in tasks_chunk:
        ans_data = task[target_key]
        
        # Use top 5 limit so max_model_len is not exceeded
        joined_context = "\n\n".join(task["context_list"][:5])
        
        # Format dynamically depending on num of candidates
        if isinstance(ans_data, list):
            ans_text = "\n".join([f"Candidate {idx}: {a}" for idx, a in enumerate(ans_data)])
        else:
            ans_text = f"Candidate 0: {ans_data}"

        for role in ["Scientist", "Critic", "GeneralPublic", "NewsAuthor"]:
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPTS[role]}, 
                {"role": "user", "content": f"Context:\n{joined_context}\n\nQuestion: {task["question"]}\n\n{ans_text}\n\nEvaluate the answer(s) strictly based on the context."}
            ]
            debater_prompts.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
    
    # Generate all 4 critiques per task in one massive batch
    debater_params = SamplingParams(temperature=0.3, max_tokens=DEBATER_TOKENS)
    debater_outputs = llm.generate(debater_prompts, debater_params, use_tqdm=False)

    chief_prompts = []
    for i, task in enumerate(tasks_chunk):
        ans_data = task[target_key]
        
        # Use top 5 limit
        joined_context = "\n\n".join(task["context_list"][:5])
        
        if isinstance(ans_data, list):
            ans_text = "\n".join([f"Candidate {idx}: {a}" for idx, a in enumerate(ans_data)])
        else:
            ans_text = f"Candidate 0: {ans_data}"

        c1 = debater_outputs[i * 4].outputs.text.strip()
        c2 = debater_outputs[i * 4 + 1].outputs.text.strip()
        c3 = debater_outputs[i * 4 + 2].outputs.text.strip()
        c4 = debater_outputs[i * 4 + 3].outputs.text.strip()
        
        critiques_key = "critiques_1" if target_key == "initial_answers" else "critiques_2"
        task[critiques_key] = [c1, c2, c3, c4]

        final_prompt = [
            {"role": "system", "content": SYSTEM_PROMPTS["Summariser"]}, 
            {"role": "user", "content": f"Context:\n{joined_context}\n\nQuestion: {task["question"]}\n\n{ans_text}\n\nCritiques:\n1. Scientist: {c1}\n2. Critic: {c2}\n3. Gen Public: {c3}\n4. News Author: {c4}\n\nVerdict JSON:"}
        ]
        chief_prompts.append(tokenizer.apply_chat_template(final_prompt, tokenize=False, add_generation_prompt=True))

    # Generate all Summariser verdicts
    chief_params = SamplingParams(temperature=0.1, max_tokens=CHIEF_TOKENS)
    chief_outputs = llm.generate(chief_prompts, chief_params, use_tqdm=False)

    # Extract and assign verdicts
    for i, task in enumerate(tasks_chunk):
        verdict_key = "verdict_1" if target_key == "initial_answers" else "verdict_2"
        task[verdict_key] = extract_json(chief_outputs[i].outputs.text)

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from {OUTPUT_FILE}")
        with open(OUTPUT_FILE, "r") as f: tasks = json.load(f)
    else:
        print(f"Starting fresh from {INPUT_FILE}")
        with open(INPUT_FILE, "r") as f: tasks = json.load(f)

    print("Loading DeepSeek-32B Judge Engine...")
    llm = LLM(
        model=JUDGE_PATH, 
        quantization="bitsandbytes", 
        load_format="bitsandbytes", 
        enforce_eager=True, 
        max_model_len=2048,           
        gpu_memory_utilization=0.95,  
        max_num_seqs=2,               
        max_num_batched_tokens=2048
    )
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_PATH, local_files_only=True)

    # Initial evaluation of 3 candidates stage
    tasks_needing_eval1 = [t for t in tasks if "verdict_1" not in t]
    
    if tasks_needing_eval1:
        print(f"\nJudging {len(tasks_needing_eval1)} tasks (Multi-Agent Consensus)")
        chunks = list(chunk_list(tasks_needing_eval1, SAVE_EVERY))
        
        for chunk in tqdm(chunks, desc="Round 1 Eval (Batched)"):
            run_council_batch(chunk, llm, tokenizer, "initial_answers")
            
            for task in chunk:
                v_data = task["verdict_1"]
                best_idx = v_data.get("best_index", 0)
                
                # Safety check in case the LLM outputs an invalid index
                if not isinstance(best_idx, int) or best_idx < 0 or best_idx >= len(task.get("initial_answers", [""])):
                    best_idx = 0
                
                # Collapse the 3 candidates down to the single best answer chosen by the council
                task["initial_answer"] = task["initial_answers"][best_idx]
                task["needs_correction"] = "Hallucination" in v_data.get("verdict", "")
            
            with open(OUTPUT_FILE, "w") as f: json.dump(tasks, f, indent=4)
    
    # Correction stage
    tasks_needing_correction = [t for t in tasks if t.get("needs_correction") and "corrected_answer" not in t]
    
    if tasks_needing_correction:
        print(f"\nGenerating corrections for {len(tasks_needing_correction)} tasks")
        corrector_params = SamplingParams(temperature=0.3, max_tokens=CORRECTOR_TOKENS)
        chunks = list(chunk_list(tasks_needing_correction, SAVE_EVERY))
        
        for chunk in tqdm(chunks, desc="Correcting (Batched)"):
            prompts = []
            for task in chunk:
                c1, c2, c3, c4 = task["critiques_1"]
                
                # Top 5 limit used
                joined_context = "\n\n".join(task["context_list"][:5]) 
                
                u_prompt = f"Context:\n{joined_context}\n\nQuestion: {task["question"]}\nOld Answer: {task["initial_answer"]}\n\nCritiques:\n1. {c1}\n2. {c2}\n3. {c3}\n4. {c4}\n\nGenerate corrected answer."
                prompt = [{"role": "system", "content": SYSTEM_PROMPTS["Corrector"]}, {"role": "user", "content": u_prompt}]
                prompts.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
            
            outputs = llm.generate(prompts, corrector_params, use_tqdm=False)
            
            for i, task in enumerate(chunk):
                raw_correction = outputs[i].outputs.text
                task["corrected_answer"] = strip_think_tags(raw_correction)
            
            with open(OUTPUT_FILE, "w") as f: json.dump(tasks, f, indent=4)

    tasks_needing_eval2 = [t for t in tasks if "corrected_answer" in t and "verdict_2" not in t]
    
    if tasks_needing_eval2:
        print(f"\nJudging {len(tasks_needing_eval2)} corrected answers")
        chunks = list(chunk_list(tasks_needing_eval2, SAVE_EVERY))
        
        for chunk in tqdm(chunks, desc="Round 2 Eval (Batched)"):
            run_council_batch(chunk, llm, tokenizer, "corrected_answer")
            
            for task in chunk:
                task["final_answer"] = task["corrected_answer"]
            
            with open(OUTPUT_FILE, "w") as f: json.dump(tasks, f, indent=4)

    for task in tasks:
        if not task.get("needs_correction"):
            task["final_answer"] = task.get("initial_answer", "")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(tasks, f, indent=4)

if __name__ == "__main__":
    main()