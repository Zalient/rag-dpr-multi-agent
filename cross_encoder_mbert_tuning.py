import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import json
import csv

# Config
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")
BASE_DIR = "/iridisfs/scratch/zc3g23"

DATA_PATH = os.path.join(BASE_DIR, "datasets/nq_triplets_scored") 
BASE_MODEL = os.path.join(BASE_DIR, "models/modernbert_base")

OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints/cross_encoder_modernbert_tuned")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LAST_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint_last")
STATE_FILE = os.path.join(LAST_CHECKPOINT_DIR, "training_state.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.csv")

# Parameters
MICRO_BATCH_SIZE = 16   
ACCUMULATION_STEPS = 8
EPOCHS = 10             
LEARNING_RATE = 1e-6    
BETAS = (0.9, 0.98)
EPSILON = 1e-6
WEIGHT_DECAY = 0.1
VALIDATION_SPLIT = 0.05
MAX_GRAD_NORM = 1.0
ATTN_DROPOUT = 0.1
HIDDEN_DROPOUT = 0.1

# Automated warmup
WARMUP_RATIO = 0.05       
MAX_SEQ_LEN = 512   
NUM_WORKERS = 4

class ScoredDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        print(f"Loading distilled triplets from {jsonl_path}...")
        with open(jsonl_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} training triplets")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item["query"], 
            item["positive"], 
            item["negative"], 
            float(item.get("teacher_pos_score", 1.0)), 
            float(item.get("teacher_neg_score", 0.0))
        )

def train():
    device = torch.device("cuda")
    # Load distilled data
    full_dataset = ScoredDataset(DATA_PATH)
    
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    # Slice data into a training set and a small test set
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Loads the datasets, drop last to ensure compatability with gradient accumulation
    train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Setup model
    config = AutoConfig.from_pretrained(BASE_MODEL, num_labels=1, local_files_only=True)
    config.attention_probs_dropout_prob = ATTN_DROPOUT
    config.hidden_dropout_prob = HIDDEN_DROPOUT

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config, local_files_only=True).to(device)
    model.gradient_checkpointing_enable() 
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPSILON, weight_decay=WEIGHT_DECAY)
    
    # Warmup calculation to start lr at 0 and ramp it up slowly over first 5% of training. Prevents model from jolting and ruining pre-trained weights
    total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
    auto_warmup_steps = int(total_steps * WARMUP_RATIO)
    print(f"Calculated total steps: {total_steps} | auto warmup steps: {auto_warmup_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=auto_warmup_steps, 
        num_training_steps=total_steps
    )
    
    loss_fct = nn.MSELoss()

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
            
            print(f"Found state file. Attempting resume from epoch {start_epoch}, etep {global_step}...")
            
            model = AutoModelForSequenceClassification.from_pretrained(LAST_CHECKPOINT_DIR, config=config, local_files_only=True).to(device)
            model.gradient_checkpointing_enable()
            
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
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, (queries, positives, negatives, t_pos_scores, t_neg_scores) in enumerate(loop):
            if epoch == start_epoch and step < skipped_steps:
                if step % 100 == 0:
                    loop.set_postfix({"status": "skipping", "target": skipped_steps})
                continue

            pos_pairs = [[q, p] for q, p in zip(queries, positives)]
            neg_pairs = [[q, n] for q, n in zip(queries, negatives)]

            pos_inputs = tokenizer(pos_pairs, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
            neg_inputs = tokenizer(neg_pairs, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)

            teacher_margins = (t_pos_scores - t_neg_scores).float().to(device)

            with torch.amp.autocast("cuda"):
                student_pos_scores = model(**pos_inputs).logits.squeeze(-1)
                student_neg_scores = model(**neg_inputs).logits.squeeze(-1)
                
                student_margins = student_pos_scores - student_neg_scores
                loss = loss_fct(student_margins, teacher_margins) / ACCUMULATION_STEPS

            loss.backward()
            total_loss += loss.item() * ACCUMULATION_STEPS

            if (step + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Every 500 steps, validate and save a checkpoint (for averaging and resuming)
                if global_step > 0 and global_step % 500 == 0:
                    model.eval()
                    val_loss_accum = 0
                    val_batches = 0
                    with torch.no_grad():
                        for v_i, (vq, vp, vn, vt_pos, vt_neg) in enumerate(val_loader):
                            if v_i > 100: break 
                            
                            v_pos_pairs = [[q, p] for q, p in zip(vq, vp)]
                            v_neg_pairs = [[q, n] for q, n in zip(vq, vn)]
                            
                            v_pos_inputs = tokenizer(v_pos_pairs, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
                            v_neg_inputs = tokenizer(v_neg_pairs, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
                            
                            v_teacher_margins = (vt_pos - vt_neg).float().to(device)
                            
                            with torch.amp.autocast("cuda"):
                                v_student_pos = model(**v_pos_inputs).logits.squeeze(-1)
                                v_student_neg = model(**v_neg_inputs).logits.squeeze(-1)
                                v_student_margins = v_student_pos - v_student_neg
                                val_loss_accum += loss_fct(v_student_margins, v_teacher_margins).item()
                                
                            val_batches += 1
                    
                    current_val_loss = val_loss_accum / val_batches if val_batches else 0.0
                    
                    with open(LOG_FILE, "a") as f:
                        csv.writer(f).writerow([epoch+1, global_step, loss.item()*ACCUMULATION_STEPS, current_val_loss])
                    
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
                "loss": f"{loss.item() * ACCUMULATION_STEPS:.4f}",
                "val_loss": f"{current_val_loss:.4f}" if current_val_loss > 0 else "init"
            })
        skipped_steps = 0

if __name__ == "__main__":
    train()