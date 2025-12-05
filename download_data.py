import os
from datasets import load_dataset
from huggingface_hub import snapshot_download

BASE_DIR = "/iridisfs/scratch/zc3g23"
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASETS = [
    # Benchmark: SQuAD 2.0 (Unanswerable questions support) 
    ("squad_v2", "squad_v2", None),
    
    # Complex/Varied: TriviaQA (RC subset) 
    ("trivia_qa", "trivia_qa_rc", "rc"),
    
    # Training Data: Pre-formatted NQ for DPR (Triplets: Query, Pos, Neg) 
    ("sentence-transformers/natural-questions", "nq_dpr_train", None),
    
    # Evaluation Data: Open Domain NQ (Questions + Answers)
    ("nq_open", "nq_open_eval", None) 
]

MODELS = [
    # Base BERT for Custom DPR
    ("bert-base-uncased", "bert-base-uncased"),
    
    # The Judge (Baseline): Facebook's Pre-trained Question Encoder
    ("facebook/dpr-question_encoder-single-nq-base", "facebook_dpr_question_encoder"),
    
    # The Judge (Baseline): Facebook's Pre-trained Context Encoder
    ("facebook/dpr-ctx_encoder-single-nq-base", "facebook_dpr_ctx_encoder")
]

def download_data():
    for hf_id, local_name, config in DATASETS:
        save_path = os.path.join(DATA_DIR, local_name)
            
        print(f"\n--- Downloading {hf_id} ---")
        try:
            if config:
                ds = load_dataset(hf_id, config)
            else:
                ds = load_dataset(hf_id)
            
            ds.save_to_disk(save_path)
            print(f"[SUCCESS] Saved to {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed {hf_id}: {e}")

def download_models():
    for hf_id, local_name in MODELS:
        save_path = os.path.join(MODEL_DIR, local_name)
        print(f"\n--- Downloading {hf_id} ---")
        try:
            # snapshot_download downloads the raw files (config, bin, json) directly (more robust than AutoModel)
            snapshot_download(repo_id=hf_id, local_dir=save_path)
            print(f"[SUCCESS] Saved to {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed {hf_id}: {e}")

if __name__ == "__main__":
    download_data()
    download_models()