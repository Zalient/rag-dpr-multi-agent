import os
import json
import re
import string
from tqdm import tqdm

# Config
BASE_DIR = "/iridisfs/scratch/zc3g23"
INPUT_JSON = os.path.join(BASE_DIR, "datasets/bi_encoder_mbert_top100_results_nq.json")

# Text Normalisation
def normalise_text(text):
    if not isinstance(text, str): return ""
    # Lowercase text
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # Remove stop words
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def main():
    print(f"--- Bi-Encoder Evaluation ---")
    print(f"Loading results from {INPUT_JSON}...")

    if not os.path.exists(INPUT_JSON):
        print(f"[ERROR]: File not found at {INPUT_JSON}")
        return

    # Load the JSON data
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    correct_k = {1: 0, 5: 0, 20: 0, 100: 0}
    total_processed = 0

    print(f"Evaluating {len(data)} queries...")

    # Iterate through each query in the JSON
    for item in tqdm(data):
        raw_ans = item["answers"]
        
        # Robust answer extraction
        # TriviaQA
        if isinstance(raw_ans, dict) and "aliases" in raw_ans:
            actual_answers = raw_ans["aliases"]
        # SQuADv2
        elif isinstance(raw_ans, dict) and "text" in raw_ans:
            actual_answers = raw_ans["text"]
        # NQ
        elif isinstance(raw_ans, list):
            actual_answers = raw_ans
        else:
            actual_answers = [str(raw_ans)]
            
        # Skip unanswerable queries
        if not actual_answers or not any(a.strip() for a in actual_answers): 
            continue
            
        norm_ans_list = [normalise_text(a) for a in actual_answers if a]
        found_matches = {k: False for k in correct_k}
        
        # Check the candidates for the answer
        candidates = item.get("candidates", [])
        
        for rank, cand in enumerate(candidates):
            norm_ctx_text = normalise_text(cand["text"])
            
            matched = False
            for ans in norm_ans_list:
                if not ans: continue
                # Exact whole word match
                if re.search(r"\b" + re.escape(ans) + r"\b", norm_ctx_text):
                    matched = True
                    break
            
            # If this specific candidate matched one of the answers then record its rank and stop looking through the candidates list because no need
            # (would need to if needed to calculate P@10 to see what percentage were correct) 
            if matched:
                for k in correct_k:
                    if rank < k: found_matches[k] = True
                break 

        # Calculate final metrics
        for k in correct_k:
            if found_matches[k]: correct_k[k] += 1
        total_processed += 1

    print(f"\n--- Bi-Encoder Results (N={total_processed}) ---")
    for k in sorted(correct_k.keys()):
        print(f"Recall@{k}: {correct_k[k] / total_processed:.2%}")

if __name__ == "__main__":
    main()