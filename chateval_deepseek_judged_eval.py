import os
import json
import re
import string
from collections import Counter

BASE_DIR = "/iridisfs/scratch/zc3g23"
INPUT_FILE = os.path.join(BASE_DIR, "datasets/mbert_judged_answers_nq.json")

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

def main():
    print(f"Loading judged answers from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r") as f:
            tasks = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR]: Could not find {INPUT_FILE}")
        return
    
    p2_substring = 0
    p2_em = 0
    p2_f1 = 0.0
    
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
            
        initial_ans = str(task.get("initial_answer", ""))
        final_ans = str(task.get("final_answer", ""))
        
        if not initial_ans and not final_ans:
            continue
            
        valid_tasks_count += 1

        # Evaluate DeepSeek"s final corrected output
        if check_match(final_ans, actual_answers): p2_substring += 1
        if check_exact_match(final_ans, actual_answers): p2_em += 1
        p2_f1 += calculate_f1(final_ans, actual_answers)

    if valid_tasks_count == 0:
        print("[ERROR]: No valid answerable tasks found in the dataset")
        return

    print("--- Final Answer Generation (DeepSeek) ---")
    print(f"Total Valid Questions Evaluated: {valid_tasks_count}")
    print(f"Substring Match: {(p2_substring / valid_tasks_count) * 100:.2f}%")
    print(f"Exact Match (EM): {(p2_em / valid_tasks_count) * 100:.2f}%")
    print(f"Macro F1: {(p2_f1 / valid_tasks_count) * 100:.2f}%")

if __name__ == "__main__":
    main()