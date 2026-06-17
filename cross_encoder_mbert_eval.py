import os
import json
import re
import string
from tqdm import tqdm

# Config
BASE_DIR = "/iridisfs/scratch/zc3g23"
INPUT_JSON = os.path.join(BASE_DIR, "datasets/cross_encoder_mbert_top10_results_nq.json")

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
    print(f"--- Cross-Encoder Evaluation ---")
    print(f"Loading results from {INPUT_JSON}...")

    if not os.path.exists(INPUT_JSON):
        print(f"[ERROR]: File not found at {INPUT_JSON}")
        return
    
    # Load the JSON data
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    # Metrics accumulators
    mrr_10 = 0.0
    p_1 = 0.0
    recall_10 = 0.0
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
            
        norm_ans_list = [normalise_text(a) for a in actual_answers if a]
        
        # Check the candidates for the answer
        candidates = item.get("candidates", [])
        
        # Track the highest rank where a match is found
        matched_rank = -1
        
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
                matched_rank = rank
                break # Stop at the first match for MRR and Recall
                
        # Calculate final metrics
        if matched_rank != -1:
            recall_10 += 1.0
            mrr_10 += 1.0 / (matched_rank + 1)
            if matched_rank == 0:
                p_1 += 1.0

        total_processed += 1

    print(f"\n--- Cross-Encoder Results (N={total_processed}) ---")
    print(f"Recall@1: {p_1 / total_processed:.2%}")
    print(f"Recall@10: {recall_10 / total_processed:.2%}")
    print(f"MRR@10: {mrr_10 / total_processed:.2f}")

if __name__ == "__main__":
    main()