import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class ModernBERTBiEncoder(nn.Module):
    """
    Bi-Encoder architecture for ModernBERT
    """
    def __init__(self, model_path, config):
        super().__init__()
            
        self.encoder = AutoModel.from_pretrained(
            model_path, 
            config=config, 
            local_files_only=True
        )
        self.encoder.gradient_checkpointing_enable() 
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Expects input_ids and attention_mask to be tensors on the correct device already
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # CLS pooling (index 0) and L2 normalisation
        emb = out.last_hidden_state[:, 0, :]
        return F.normalize(emb, p=2, dim=1)
    
    def save_pretrained(self, path):
        self.encoder.save_pretrained(path)

def gradcache_loss(q, p, n, temp=0.05, **kwargs):
    cands = torch.cat([p, n], dim=0)
    scores = torch.matmul(q, cands.transpose(0, 1)) / temp
    targets = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(scores, targets)