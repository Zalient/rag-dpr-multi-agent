import os
from torch.utils.data import Dataset
from datasets import load_from_disk

class DPRData(Dataset):
    """
    Handles Tommy Aarsen's natural-questions-hard-negatives dataset: 
    huggingface.co/datasets/tomaarsen/natural-questions-hard-negatives
    """
    def __init__(self, data_path):
        print(f"Loading Arrow dataset from: {data_path}")
        self.dataset = load_from_disk(data_path)
        
        # Handle if the folder is a DatasetDict containing a "train" split
        if hasattr(self.dataset, "keys") and "train" in self.dataset.keys():
            self.dataset = self.dataset["train"]
        print(f"Dataset size: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract raw text
        query = item.get("query")
        pos_text = item.get("answer")
        neg_text = item.get("negative") # Using triplet-all instead of triplet-5

        # Return raw strings so the DataLoader can batch them dynamically
        return query, pos_text, neg_text