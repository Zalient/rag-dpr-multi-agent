import os
import json
import re
import string
from collections import Counter

BASE_DIR = "/iridisfs/scratch/zc3g23"
INPUT_FILE = os.path.join(BASE_DIR, "datasets/mbert_generated_answers_nq.json")

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

def check_match(prediction, actual_answers):
    # Checks if the generated answers matches any of the actual answers via substring
    if isinstance(actual_answers, str):
        actual_answers = [actual_answers] # Convert to list if it is a single string
        
    norm_pred = normalise_text(prediction)
    
    for truth in actual_answers:
        norm_truth = normalise_text(truth)
        # Using substring matching because LLM models often output full sentences
        if norm_truth in norm_pred:
            return True
    return False

def check_exact_match(prediction, actual_answers):
    # Checks if the generated answer is an exact match to the actual answer after normalisation applied
    if isinstance(actual_answers, str):
        actual_answers = [actual_answers]
        
    norm_pred = normalise_text(prediction)
    
    for truth in actual_answers:
        norm_truth = normalise_text(truth)
        if norm_pred == norm_truth:
            return True
    return False

def calculate_f1(prediction, actual_answers):
    # Calculates the macro F1 score between the generated answer and actual answers
    if isinstance(actual_answers, str):
        actual_answers = [actual_answers]
        
    def _compute_f1(pred_str, truth_str):
        pred_toks = pred_str.split()
        truth_toks = truth_str.split()
        
        # Only gets a score of 1 if both are empty
        if len(pred_toks) == 0 or len(truth_toks) == 0:
            return int(pred_toks == truth_toks)
            
        common = Counter(pred_toks) & Counter(truth_toks)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
            
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(truth_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    norm_pred = normalise_text(prediction)
    # Return the highest F1 score among all actual answers
    return max(_compute_f1(norm_pred, normalise_text(truth)) for truth in actual_answers)

def get_majority_vote(candidates):
    # Find the most common answer among the three generated ones, most of the time this will not apply (t=1)
    # Normalise candidates so slight variations count as the same vote
    norm_candidates = [normalise_text(c) for c in candidates]
    counts = Counter(norm_candidates)
    
    top_ans, top_count = counts.most_common(1)[0]
    
    # If 2 or more agents agree, we have a majority
    if top_count >= 2:
        # Return the original unnormalised string that won
        for c in candidates:
            if normalise_text(c) == top_ans:
                return c
                
    # If a majority cannot be found (i.e. they are all different, very common at t=1) then fallback to candidate 0
    return candidates[0]

def main():
    print(f"Loading generated answers from {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        tasks = json.load(f)

    # Counters for our metrics
    top_1_correct = 0
    pass_at_3_correct = 0
    majority_vote_substring = 0
    majority_vote_em = 0
    majority_vote_f1_total = 0.0
    
    # Only counting tasks that are actually answerable
    valid_tasks_count = 0

    for task in tasks:
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
            
        candidates = task.get("initial_answers", [])
        if not candidates:
            continue
            
        # Only count it as a task if it had actual answers and candidate answers
        valid_tasks_count += 1

        # Pass@3
        if any(check_match(c, actual_answers) for c in candidates):
            pass_at_3_correct += 1
            
        # Evaluate the majority vote answer
        majority_ans = get_majority_vote(candidates)
        
        # Substring match
        if check_match(majority_ans, actual_answers):
            majority_vote_substring += 1
            
        # Exact Match (EM)
        if check_exact_match(majority_ans, actual_answers):
            majority_vote_em += 1
            
        # Macro F1 score
        majority_vote_f1_total += calculate_f1(majority_ans, actual_answers)

    print("--- Initial Answer Generation (Llama) ---")
    print(f"Pass@3: {(pass_at_3_correct / valid_tasks_count) * 100:.2f}%")
    print(f"Majority Vote (Substring): {(majority_vote_substring / valid_tasks_count) * 100:.2f}%")
    print(f"Majority Vote (Exact Match): {(majority_vote_em / valid_tasks_count) * 100:.2f}%")
    print(f"Majority Vote (Macro F1): {(majority_vote_f1_total / valid_tasks_count) * 100:.2f}%")
    
if __name__ == "__main__":
    main()