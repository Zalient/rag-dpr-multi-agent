import os
import torch
import torch.nn.functional
import faiss
import numpy as np
import gc
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import concatenate_datasets
from transformers import AutoModel, AutoTokenizer
import torch.multiprocessing

# Config
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")
BASE_DIR = "/iridisfs/scratch/zc3g23"

MODEL_PATH = os.path.join(BASE_DIR, "models/bi_encoder_modernbert_averaged")
CORPUS_DIR = os.path.join(BASE_DIR, "datasets/psgs_w100")
INDEX_OUTPUT_PATH = os.path.join(BASE_DIR, "indexes/bi_encoder_mbert_index")

MAX_DOCS_PER_FRAGMENT = 2500000  # Save every 2.5M docs
BATCH_SIZE = 512       
EMBED_DIM = 768
NUM_WORKERS = 4

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
    files = sorted(glob.glob(os.path.join(folder_path, "**/*.arrow"), recursive=True))
    
    # Create Dataset object for each file
    datasets = [load_dataset_safe(f) for f in files]
    # Merge them together into a single Dataset object
    return concatenate_datasets([d for d in datasets if d is not None])
        
def encode_batch(batch, tokenizer, model, device):
    # Doing indexing so do not need gradients
    with torch.no_grad():
        # Padding to make sentences same length, truncation to stop long sentences, PyTorch tensors instead of Python lists
        inputs = tokenizer(text=batch["title"], text_pair=batch["text"], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
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
    
    # Load the corpus
    print("Loading Corpus...")
    corpus = load_sharded_dataset(CORPUS_DIR)
    if corpus is None: return

    # Load the model
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()


    # Initialise/Resume
    existing_parts = sorted(glob.glob(f"{INDEX_OUTPUT_PATH}_part*.index"))
    start_doc_idx = 0
    fragment_id = 0
    
    # Skip documents if resuming
    if existing_parts:
        fragment_id = len(existing_parts)
        start_doc_idx = fragment_id * MAX_DOCS_PER_FRAGMENT
        print(f"Resuming - Found {fragment_id} existing parts. Skipping first {start_doc_idx} docs")
    
    print("Initialising FAISS index using Flat IP...")
    index = faiss.IndexFlatIP(EMBED_DIM)

    # Start indexing from the resume/initialise point
    subset_corpus = corpus.select(range(start_doc_idx, len(corpus))) if start_doc_idx < len(corpus) else None
    
    if subset_corpus:
        # Create data loader to fetch data 
        dataloader = DataLoader(subset_corpus, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        # Buffer for embeddings before being committed to FAISS index
        buffer_embs = []
        # Counter to keep track of num of documents in current file before starting a new one
        docs_in_current_fragment = 0

        for batch in tqdm(dataloader, desc=f"Indexing (Part {fragment_id})"):
            embs = encode_batch(batch, tokenizer, model, device)
            buffer_embs.append(embs)
            
            # Irregularly update the FAISS index
            if len(buffer_embs) * BATCH_SIZE >= 100000:
                # Move embeddings to FAISS index
                index.add(np.concatenate(buffer_embs))
                docs_in_current_fragment += (len(buffer_embs) * BATCH_SIZE)
                buffer_embs = []
                # Garbage collector to clean up memory
                gc.collect()

            # Once num of documents hits limit, write to file
            if docs_in_current_fragment >= MAX_DOCS_PER_FRAGMENT:
                frag_path = f"{INDEX_OUTPUT_PATH}_part{fragment_id}.index"
                print(f"\n[SAVE] Fragment {fragment_id} reached {MAX_DOCS_PER_FRAGMENT} docs. Saving to {frag_path}")
                faiss.write_index(index, frag_path)
                
                # Clears the index from memory to make room for next 2.5M documents
                index.reset() 
                fragment_id += 1
                docs_in_current_fragment = 0

        # If there are still left over embeddings they get added to the index
        if buffer_embs:
            index.add(np.concatenate(buffer_embs))
        # If there are still embeddings in the index that have not been saved to a file yet then save them
        if index.ntotal > 0:
            final_frag_path = f"{INDEX_OUTPUT_PATH}_part{fragment_id}.index"
            faiss.write_index(index, final_frag_path)
            print(f"Final partial fragment saved: {final_frag_path}")

if __name__ == "__main__":
    main()