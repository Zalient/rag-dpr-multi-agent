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
    
    # Training Data: Pre-formatted NQ for DPR (Triplets: Query, Pos, Neg) (M2)
    ("tomaarsen/natural-questions-hard-negatives", "nq_triplets", "triplet-all"),
    
    # Evaluation Data: Open Domain NQ (Questions + Answers) (M4)
    ("nq_open", "nq_open_eval", None), 
    
    # Evaluation Data: Wikipedia context (M4)
    ("wiki_dpr", "psgs_w100", "psgs_w100")
]

MODELS = [
    # Indexing: Creates the dense vector index from passages (M1)
    ("facebook/dpr-ctx_encoder-single-nq-base", "facebook_dpr_ctx_encoder"),

    # Retrieval: Encodes the question into a vector for searching the index (M1)
    ("facebook/dpr-question_encoder-single-nq-base", "facebook_dpr_question_encoder"),

    # Custom DPR Base: The generic BERT skeleton used to initialise and fine-tune the custom dual-encoder (M2)
    ("bert-base-uncased", "bert_base_uncased"),

    # Novel Modification: The newest encoder to replace BERT with and study performance improvements
    ("answerdotai/ModernBERT-base", "modernbert_base"),
    
    # Generation (The Student): Synthesises the final answer from retrieved passages (M3)
    # In the multi-agent workflow, this "weaker" model generates the candidate response for the stronger judge to critique
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama_3_1_8b_instruct"),

    # Refinement (The Professor): Acts as the "Judge" and "Filter" across the pipeline (M3)
    # 1. MAIN-RAG: Filters noisy documents from retrieval before they reach the generator
    # 2. ChatEval: Adopts multiple personas (4) to critique and grade the generator's output
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
                ds = load_dataset(path=hf_id, name=config, trust_remote_code=True)
            else:
                ds = load_dataset(path=hf_id, trust_remote_code=True)
            # trust_remote_code needed for wiki_dpr.py parsing script to execute to construct the dataset
            
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
                ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.tflite", "*.onnx"])
            print(f"[SUCCESS] Saved to {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed {hf_id}: {e}")

if __name__ == "__main__":
    download_data()
    download_models()
