import os
import json
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from datasets import concatenate_datasets
import glob

# Config
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")
BASE_DIR = "/iridisfs/scratch/zc3g23"

MODEL_PATH = os.path.join(BASE_DIR, "models/bi_encoder_modernbert_averaged") 
INDEX_FILE = os.path.join(BASE_DIR, "indexes/bi_encoder_mbert_index") 
CORPUS_DIR = os.path.join(BASE_DIR, "datasets/psgs_w100")
QA_EVAL_PATH = os.path.join(BASE_DIR, "datasets/nq_open_eval")
OUTPUT_FILE = os.path.join(BASE_DIR, "datasets/bi_encoder_mbert_top100_results_nq.json")

BATCH_SIZE = 128
EMBED_DIM = 768
NUM_WORKERS = 4
TOP_K = 100

def load_dataset_safe(path):
    try:
        from datasets import Dataset
        import pyarrow as pa
        # Memory map to only load specific rows into memory when needed
        with pa.memory_map(path, "r") as source:
            try:
                # If .arrow saved as stream then open it as stream and read sequentially
                table = pa.ipc.open_stream(source).read_all()
            except:
                # Otherwise it is saved as a file so look at start of file (source.seek(0)) and read
                source.seek(0)
                table = pa.ipc.open_file(source).read_all()
            # Clean the metadata so only raw data is left, means script works regardless of dataset input
            clean_schema = table.schema.with_metadata({})
            table = table.cast(clean_schema)
            # Wrap the arrow table into a Dataset object. Because arrow table memory mapped, the Dataset object is fast and lazy
            return Dataset(table)
    except Exception as e:
        print(f"[ERROR]: Error loading {path}: {e}")
        return None

def load_sharded_validation(folder_path):
    # Look inside the input folder and all its subfolders (**), looking for .arrow files
    all_files = sorted(glob.glob(os.path.join(folder_path, "**/*.arrow"), recursive=True))
    # Only want eval files, not train files
    eval_files = [f for f in all_files if "train" not in f.lower()]
    
    print(f"Filtered out training data. Found {len(eval_files)} evaluation shards")
    # Create Dataset object for each file
    datasets = [load_dataset_safe(f) for f in eval_files]
    # Merge them together into a single Dataset object
    return concatenate_datasets([d for d in datasets if d is not None])

def encode_batch(texts, tokenizer, model, device):
    # Doing inference so do not need gradients
    with torch.no_grad():
        # Padding to make sentences same length, truncation to stop long sentences, PyTorch tensors instead of Python lists
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        # Feed inputs into model to get last_hidden_state 
        outputs = model(**inputs)
        # Use CLS token (index 0)
        emb = outputs.last_hidden_state[:, 0, :]
        # Normalise
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        # Move result from GPU to CPU and convert into NumPy array
        return emb.cpu().numpy().astype("float32")

def main():
    # Get the GPU
    device = torch.device("cuda")
    
    # Load the model
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # Need the index to retrieve top IDs, but also need the text to save into the JSON
    print("Loading Corpus for Text Lookup...")
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "**/*.arrow"), recursive=True))
    datasets = [load_dataset_safe(f) for f in files]
    corpus = concatenate_datasets([d for d in datasets if d is not None])
    
    # Load index
    print(f"Loading FAISS Index from {INDEX_FILE}...")
    index = faiss.read_index(INDEX_FILE)

    print(f"Index Ready. Total Docs: {index.ntotal}")

    # Load Queries
    print("Loading Queries...")
    val_dataset = load_sharded_validation(QA_EVAL_PATH)
    
    # Detect the correct column names
    keys = val_dataset.column_names
    q_key = "question" if "question" in keys else "query"
    a_key = "answer" if "answer" in keys else "answers"
    
    queries = val_dataset[q_key]
    answers_list = val_dataset[a_key]

    # Search
    print(f"Searching for {len(queries)} queries...")
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(queries), BATCH_SIZE)):
        batch_queries = queries[i : i+BATCH_SIZE]
        batch_answers = answers_list[i : i+BATCH_SIZE]
        
        # Encode queries
        q_embs = encode_batch(batch_queries, tokenizer, model, device)
        
        # Ask FAISS for top k (100) most similar document vectors, returns distances/scores and indexes/IDs
        # Dimensions of D and I = num of queries x top k (i.e. for NQ 3610x100)
        D, I = index.search(q_embs, TOP_K)
        
        # Pair scores with IDs with zip
        for j, (distances, indices) in enumerate(zip(D, I)):
            query_text = batch_queries[j]
            ans_list = batch_answers[j]
            
            candidates = []
            # For each query look at all of its results, rank 0 being best match
            for rank, doc_idx in enumerate(indices):
                # FAISS could not find a match
                if doc_idx == -1: continue
                
                # Ignore IDs that do not exist
                if int(doc_idx) >= len(corpus): 
                    continue
                
                # Turn IDs into text
                # Pull the actual row from memory mapped dataset 
                row = corpus[int(doc_idx)]
                ctx_text = f"{row["title"]} {row["text"]}"
                
                # Each candidate saved as a dictionary containing the key info
                candidates.append({
                    "doc_id": str(doc_idx), # or row["id"]
                    "score": float(distances[rank]),
                    "text": ctx_text
                })
            
            # Once all the candidates have been formatted for every query, the final results list with all queries can be made
            results.append({
                "question": query_text,
                "answers": ans_list,
                "candidates": candidates
            })

    # Save to the output file
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()