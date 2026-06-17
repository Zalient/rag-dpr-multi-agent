import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import csv
import warnings
from grad_cache import GradCache 
from dpr_data import DPRData 
from dpr_model import DPRModel, gradcache_loss

# Config
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")
BASE_DIR = "/iridisfs/scratch/zc3g23"

DATA_PATH = os.path.join(BASE_DIR, "datasets/nq_triplets") 
BASE_MODEL = os.path.join(BASE_DIR, "models/modernbert_base")
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints/bi_encoder_modernbert_tuned")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LAST_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint_last")
STATE_FILE = os.path.join(LAST_CHECKPOINT_DIR, "training_state.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.csv")

# Parameters
CHUNK_SIZE = 32            
ACCUMULATION_STEPS = 4    
GLOBAL_BATCH_SIZE = CHUNK_SIZE * ACCUMULATION_STEPS 

EPOCHS = 15
LEARNING_RATE = 1e-5
BETAS = (0.9, 0.98)
EPSILON = 1e-6
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT = 0.05
MAX_GRAD_NORM = 1.0

# Automated warmup
WARMUP_RATIO = 0.05 
MAX_Q_LEN = 64
MAX_CTX_LEN = 256
TEMP = 0.05
NUM_WORKERS = 4        

def train():
    device = torch.device("cuda")
    
    # Loads triplet data
    full_dataset = DPRData(DATA_PATH)
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    # Slice data into a training set and a small test set
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Loads the datasets, drop last to ensure compatability with GradCache batching
    train_loader = DataLoader(train_ds, batch_size=GLOBAL_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CHUNK_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Load config for the model
    config = AutoConfig.from_pretrained(BASE_MODEL, local_files_only=True)
    if hasattr(config, "reference_compile"): config.reference_compile = False
    
    model = DPRModel(BASE_MODEL, config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPSILON, weight_decay=WEIGHT_DECAY)
    
    # Warmup calculation to start lr at 0 and ramp it up slowly over first 5% of training. Prevents model from jolting and ruining pre-trained weights
    total_steps = len(train_loader) * EPOCHS
    auto_warmup_steps = int(total_steps * WARMUP_RATIO)
    print(f"Calculated total steps: {total_steps} | auto warmup steps: {auto_warmup_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=auto_warmup_steps, 
        num_training_steps=total_steps
    )

    gc = GradCache(
        # Use same model for query, positive, and negative
        models=[model, model, model], 
        # Chunk size is physical GPU limit (32)
        chunk_sizes=CHUNK_SIZE, 
        loss_fn=lambda q, p, n, **kwargs: gradcache_loss(q, p, n, temp=TEMP), 
        fp16=False
    )

    # Resume
    start_epoch = 0
    global_step = 0
    skipped_steps = 0  
    best_val_loss = float("inf")

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                start_epoch = state["epoch"]
                global_step = state["global_step"]
                best_val_loss = state.get("best_val_loss", float("inf"))
                skipped_steps = state.get("step_in_epoch", 0)
            
            print(f"Found state file. Attempting resume from epoch {start_epoch}, step {global_step}...")
            
            model.encoder = AutoModel.from_pretrained(LAST_CHECKPOINT_DIR, config=config, local_files_only=True).to(device)
            model.encoder.gradient_checkpointing_enable()
            
            if os.path.exists(os.path.join(LAST_CHECKPOINT_DIR, "optimizer.pt")):
                optimizer.load_state_dict(torch.load(os.path.join(LAST_CHECKPOINT_DIR, "optimizer.pt")))
                scheduler.load_state_dict(torch.load(os.path.join(LAST_CHECKPOINT_DIR, "scheduler.pt")))
            print(f"Successfully resumed")
        except Exception as e:
            print(f"Resume failed ({str(e)}). Starting from scratch.")
            start_epoch = 0
            global_step = 0
            skipped_steps = 0

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            csv.writer(f).writerow(["Epoch", "Global_Step", "Train_Loss", "Val_Loss"])

    model.train()
    current_val_loss = 0.0

    for epoch in range(start_epoch, EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, (queries, positives, negatives) in enumerate(loop):
            if epoch == start_epoch and step < skipped_steps:
                if step % 100 == 0:
                    loop.set_postfix({"status": "skipping", "target": skipped_steps})
                continue

            def q_forward(model_, inp):
                inp = {k: v.to(device) for k, v in inp.items()}
                return model_(**inp)

            def ctx_forward(model_, inp):
                inp = {k: v.to(device) for k, v in inp.items()}
                return model_(**inp)

            q_in = tokenizer(list(queries), padding=True, truncation=True, max_length=MAX_Q_LEN, return_tensors="pt")
            p_in = tokenizer(list(positives), padding=True, truncation=True, max_length=MAX_CTX_LEN, return_tensors="pt")
            n_in = tokenizer(list(negatives), padding=True, truncation=True, max_length=MAX_CTX_LEN, return_tensors="pt")

            optimizer.zero_grad()
            
            # Replaces the standard loss.backward() with GradCache logic
            loss = gc(
                q_in, p_in, n_in, 
                forward_fn=q_forward, 
                forward_fn_p=ctx_forward, 
                forward_fn_n=ctx_forward
            )
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1

            # Every 500 steps, validate and save a checkpoint (for averaging and resuming)
            if global_step > 0 and global_step % 500 == 0:
                model.eval()
                val_loss_accum = 0
                val_batches = 0
                with torch.no_grad():
                    for v_i, (vq, vp, vn) in enumerate(val_loader):
                        if v_i > 200: break 
                        
                        vq_in = tokenizer(list(vq), padding=True, truncation=True, max_length=MAX_Q_LEN, return_tensors="pt").to(device)
                        vp_in = tokenizer(list(vp), padding=True, truncation=True, max_length=MAX_CTX_LEN, return_tensors="pt").to(device)
                        vn_in = tokenizer(list(vn), padding=True, truncation=True, max_length=MAX_CTX_LEN, return_tensors="pt").to(device)
                        
                        vq_emb = model(**vq_in)
                        vp_emb = model(**vp_in)
                        vn_emb = model(**vn_in)
                        
                        val_loss_accum += gradcache_loss(vq_emb, vp_emb, vn_emb, temp=TEMP).item()
                        val_batches += 1
                
                current_val_loss = val_loss_accum / val_batches if val_batches else 0.0
                
                with open(LOG_FILE, "a") as f:
                    csv.writer(f).writerow([epoch+1, global_step, loss.item(), current_val_loss])
                
                # Overwrite last version
                model.save_pretrained(LAST_CHECKPOINT_DIR)
                tokenizer.save_pretrained(LAST_CHECKPOINT_DIR)
                state = {
                    "epoch": epoch,          
                    "step_in_epoch": step + 1, 
                    "global_step": global_step, 
                    "best_val_loss": best_val_loss
                }
                with open(STATE_FILE, "w") as f: json.dump(state, f)
                
                torch.save(optimizer.state_dict(), os.path.join(LAST_CHECKPOINT_DIR, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(LAST_CHECKPOINT_DIR, "scheduler.pt"))
                
                # Save checkpoint
                step_path = os.path.join(OUTPUT_DIR, f"checkpoint_ep{epoch+1}_step{global_step}")
                model.save_pretrained(step_path)
                tokenizer.save_pretrained(step_path)
                
                model.train()

            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "val_loss": f"{current_val_loss:.4f}" if current_val_loss > 0 else "init"
            })
        skipped_steps = 0

if __name__ == "__main__":
    train()