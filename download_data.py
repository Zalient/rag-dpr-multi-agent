import os
from datasets import load_dataset
from huggingface_hub import snapshot_download

BASE_DIR = "/iridisfs/scratch/zc3g23"
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASETS = [
    # Pre-formatted NQ for DPR (Triplets: Query, Pos, Neg) 
    ("tomaarsen/natural-questions-hard-negatives", "nq_triplets", "triplet-all"),
    
    # Open Domain NQ (Questions + Answers)
    ("nq_open", "nq_open_eval", None), 
    
    # SQuAD 2.0 (Unanswerable questions) 
    ("squad_v2", "squad_v2", None),
    
    # TriviaQA (RC subset) 
    ("trivia_qa", "trivia_qa_rc", "rc"),
    
    # Wikipedia context (2018 dump)
    ("wiki_dpr", "psgs_w100", "psgs_w100.nq.no_index")
]

MODELS = [
    # Baseline Bi-Encoder (Context)
    ("facebook/dpr-ctx_encoder-single-nq-base", "facebook_dpr_ctx_encoder"),

    # Baseline Bi-Encoder (Queries)
    ("facebook/dpr-question_encoder-single-nq-base", "facebook_dpr_question_encoder"),

    # ModernBERT Bi-Encoder/Cross-Encoder to be fine-tuned
    ("answerdotai/ModernBERT-base", "modernbert_base"),
    
    # Cross-Encoder Teacher
    ("BAAI/bge-reranker-v2-m3", "bge_reranker_v2_m3"),
    
    # Generation (The Student): Generates the final answers from retrieved passages
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama_3_1_8b_instruct"),

    # Refinement (The Professor): ChatEval: Adopts multiple personas (4) to critique and grade the generator's output
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek_r1_distill_qwen_32b")
]

def download_data():
    for hf_id, local_name, config in DATASETS:
        save_path = os.path.join(DATA_DIR, local_name)

        # Skip if already exists to save time/bandwidth
        if os.path.exists(save_path):
            print(f"[SKIP] {local_name} already exists")
            continue
            
        print(f"\n--- Downloading {hf_id} ---")
        try:
            if config:
                ds = load_dataset(hf_id, config, trust_remote_code=True)
            else:
                ds = load_dataset(hf_id, trust_remote_code=True)
            
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
            snapshot_download(
                repo_id=hf_id, 
                local_dir=save_path, 
                ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.tflite", "*.onnx", "coreml/*"])
            print(f"[SUCCESS] Saved to {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed {hf_id}: {e}")

if __name__ == "__main__":
    download_data()
    download_models()