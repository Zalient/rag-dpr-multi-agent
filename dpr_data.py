from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_from_disk

class DPRData(Dataset):
    """
    Custom PyTorch Dataset for Dense Passage Retrieval (DPR) tasks.
    
    This handles loading NQ-style (Natural Questions) datasets with hard negatives and
    tokenizing triplets (Question, Positive, Negative) for dual-encoder training.
    """

    def __init__(self, data_path, tokenizer_path, max_q_len=64, max_ctx_len=256):
        """
        Initialises the dataset and tokenizer.

        Args:
            data_path (str): Path to the pre-saved Hugging Face dataset on disk.
            tokenizer_path (str): Path to the pre-trained tokenizer on disk.
            max_q_len (int, optional): Max token length for questions. Defaults to 64.
            max_ctx_len (int, optional): Max token length for context passages. 
            Defaults to 256 (512 causes OOM issues).
        """
        print(f"Loading dataset from: {data_path}")
        # Load the pre-saved dataset from disk (compute nodes have no internet access)
        self.dataset = load_from_disk(data_path)["train"]
        
        # Finds the first column starting with "negative"
        self.neg_key = next((key for key in self.dataset.column_names if key.startswith("negative")), None)
        print(f"Using negative column: '{self.neg_key}'")
        
        # Load pre-trained vectors from disk (local_files_only=True)
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path, 
            local_files_only=True
        )
        
        self.max_q_len = max_q_len
        self.max_ctx_len = max_ctx_len
        print(f"Dataset loaded - size: {len(self.dataset)} examples")

    def __len__(self):
        """
        Returns the total number of examples in the dataset.

        Returns:
            int: The total count of training examples.
        """
        return len(self.dataset)

    def tokenize(self, text, max_len):
        """
        Helper function to tokenize input text using the loaded tokenizer.

        Args:
            text (str): The input text to tokenize.
            max_len (int): The maximum length for truncation/padding.

        Returns:
            BatchEncoding: A dictionary containing 'input_ids' and 'attention_mask' tensors.
        """
        return self.tokenizer(
            text=text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        """
        Retrieves and tokenizes a specific example by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary of tensors containing tokenized inputs for:
                - q_input_ids, q_attention_mask (Query)
                - pos_input_ids, pos_attention_mask (Positive Passage)
                - neg_input_ids, neg_attention_mask (Negative Passage)
        """
        # Retrieve the raw example from the Hugging Face dataset
        item = self.dataset[idx]
        
        # 1. Get query 
        question = item.get("query")
        
        # 2. Get positive passage (answers the question)
        pos_text = item["answer"]
        
        # 3. Get negative passage
        neg_text = item[self.neg_key]

        # 4. Tokenize
        q_inputs = self.tokenize(text=question, max_len=self.max_q_len)
        pos_inputs = self.tokenize(text=pos_text, max_len=self.max_ctx_len)
        neg_inputs = self.tokenize(text=neg_text, max_len=self.max_ctx_len)

        # Use .squeeze(dim=0) because the tokenizer returns shape [1, Seq_Len], 
        # but the DataLoader expects [Seq_Len] to stack them correctly into batches
        return {
            "q_input_ids": q_inputs["input_ids"].squeeze(dim=0),
            "q_attention_mask": q_inputs["attention_mask"].squeeze(dim=0),
            "pos_input_ids": pos_inputs["input_ids"].squeeze(dim=0),
            "pos_attention_mask": pos_inputs["attention_mask"].squeeze(dim=0),
            "neg_input_ids": neg_inputs["input_ids"].squeeze(dim=0),
            "neg_attention_mask": neg_inputs["attention_mask"].squeeze(dim=0)}