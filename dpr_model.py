import torch.nn as nn
from transformers import AutoModel

class DPRModel(nn.Module):
    def __init__(self, model_path, config):
        """
        Initialises the DPR model.

        Args:
            model_path (str): Path to the pre-trained model.
            config (PretrainedConfig): Configuration object containing 
            specific dropout parameters from the relevant literature.
        """
        super(DPRModel, self).__init__()
        # Store the specific configuration (useful for saving the model)
        self.config = config

        # Load in the encoder (e.g. ModernBERT) architecture and its weights
        self.question_encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_path, 
            config=config,
        )
        self.ctx_encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_path, 
            config=config,
        )

    def forward(
        self,
        q_input_ids=None,
        q_attention_mask=None,
        ctx_input_ids=None,
        ctx_attention_mask=None,
    ):
        """
        Computes the embeddings for questions and (optionally) for contexts.
        
        This method uses the corresponding encoders to generate the [CLS] token 
        representation for both the queries and the passages.

        Args:
            q_input_ids (torch.Tensor, optional): Input token IDs for the questions. 
                Shape: (batch_size, sequence_length). Defaults to None.
            q_attention_mask (torch.Tensor, optional): Attention mask for the questions. 
                Shape: (batch_size, sequence_length). Defaults to None.
            ctx_input_ids (torch.Tensor, optional): Input token IDs for the contexts. 
                Shape: (batch_size, sequence_length). Defaults to None.
            ctx_attention_mask (torch.Tensor, optional): Attention mask for the contexts. 
                Shape: (batch_size, sequence_length). Defaults to None.

        Returns:
            tuple: A tuple containing:
                - q_out (torch.Tensor): The pooled question embeddings. 
                  Shape: (batch_size, hidden_size).
                - ctx_out (torch.Tensor or None): The pooled context embeddings. 
                  Shape: (batch_size, hidden_size). Returns None if contexts are not provided.
        """
        q_out = None
        ctx_out = None

        # Encode Questions
        if q_input_ids is not None:
            # 1. Get the ModelOutput object
            model_output = self.question_encoder(
                input_ids=q_input_ids, 
                attention_mask=q_attention_mask
            )
            # 2. Extract the sequence output: (batch, seq_len, hidden)
            sequence_output = model_output.last_hidden_state
            
            # 3. Extract the CLS token (batch, hidden)
            q_out = sequence_output[:, 0, :]

        # Encode Contexts
        if ctx_input_ids is not None:
            # 1. Get the ModelOutput object
            model_output = self.ctx_encoder(
                input_ids=ctx_input_ids, 
                attention_mask=ctx_attention_mask
            )
            # 2. Extract the sequence output: (batch, seq_len, hidden)
            sequence_output = model_output.last_hidden_state
            
            # 3. Extract the CLS token (batch, hidden)
            ctx_out = sequence_output[:, 0, :]

        return q_out, ctx_out