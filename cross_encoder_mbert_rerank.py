import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Config
torch.set_float32_matmul_precision("high")
BASE_DIR = "/iridisfs/scratch/zc3g23"

BASE_MODEL_PATH = os.path.join(BASE_DIR, "models/modernbert_base")
CROSS_ENCODER_PATH = os.path.join(BASE_DIR, "models/cross_encoder_modernbert_averaged")
CANDIDATES_FILE = os.path.join(BASE_DIR, "datasets/bi_encoder_mbert_top100_results_nq.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "datasets/cross_encoder_mbert_top10_results_nq.json")

BATCH_SIZE = 64
TOP_K_TO_KEEP = 10  # Keep only top 10

def extract_score(tuple_item):
    _, score_val = tuple_item
    return score_val

def main():
    device = torch.device("cuda")

    print(f"Loading tokeniser from: {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    
    print(f"Loading model from: {CROSS_ENCODER_PATH}...")
    model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_PATH, local_files_only=True).to(device)
    model.eval()

    print(f"Loading pre-filtered candidates from: {CANDIDATES_FILE}...")
    if not os.path.exists(CANDIDATES_FILE):
        print(f"[ERROR]: Could not find {CANDIDATES_FILE}")
        return
        
    with open(CANDIDATES_FILE, "r") as f:
        data = json.load(f)

    processed_tasks = []

    print(f"Reranking {len(data)} queries...")
    for item in tqdm(data):
        query = item.get("question").strip()
        candidates = item.get("candidates")
        
        if not candidates: 
            continue
            
        candidates = candidates[:100]
        
        # Score pairs
        pairs = []
        for c in candidates:
            pairs.append([query, c.get("text")])
            
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(pairs), BATCH_SIZE):
                batch = pairs[i:i+BATCH_SIZE]
                inputs = tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(device)
                
                with torch.amp.autocast("cuda"):
                    logits = model(**inputs).logits
                    
                if logits.shape[-1] > 1:
                    batch_scores = logits[:, 1]
                else:
                    batch_scores = logits.squeeze(-1)
                
                scores.extend(batch_scores.cpu().tolist())

        # Ranking
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=extract_score, reverse=True)
        
        # Take the top K
        top_k = scored_candidates[:TOP_K_TO_KEEP]

        context_list = []
        candidate_list = []
        
        for cand_dict, score_val in top_k:
            text_val = cand_dict.get("text")
            context_list.append(text_val)
            candidate_list.append({"text": text_val, "score": float(score_val)})

        task_payload = {
            "question": query,
            "answers": item.get("answers"),
            "context_list": context_list, 
            "candidates": candidate_list 
        }
        
        processed_tasks.append(task_payload)
        item.clear()

    # Save results
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(processed_tasks, f, indent=4)

if __name__ == "__main__":
    main()