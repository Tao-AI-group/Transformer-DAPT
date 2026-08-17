import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .modeling_bert import BertCLS
from pycox.models.utils import pad_col, make_subgrid
import torchtuples as tt
import random
from easydict import EasyDict


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed to set for the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureEmbedding(nn.Module):
    """
    A feature embedding layer that maps categorical features to dense embeddings.

    Args:
        num_features (int): The number of distinct categorical features.
        embedding_dim (int): The dimension of the embeddings for each feature.
        padding_idx (int): The index for the padding token in embeddings.
    """
    def __init__(self, num_features: int, embedding_dim: int, padding_idx: int):
        super(FeatureEmbedding, self).__init__()
        self.feature_embeddings = nn.Embedding(
            num_features + 1,  # +1 for a padding token index
            embedding_dim,
            padding_idx=padding_idx
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeatureEmbedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Embedded features with dropout applied.
        """
        embeddings = self.feature_embeddings(x)
        return self.dropout(embeddings)


class Transformer_DAPT(nn.Module):
    """
    Transformer-based model for multi-feature processing that uses attention
    for feature interactions. This model includes both survival and binary
    classification heads.

    Args:
        config (EasyDict): Configuration object holding hyperparameters and
            model settings.
    """
    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config
        self.sub = config['sub']
        self.duration_index = config['duration_index']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature embedding
        self.embeddings = FeatureEmbedding(
            num_features=config.num_categorical_feature,
            embedding_dim=config.embedding_size,
            padding_idx=config.num_categorical_feature,
        )

        # Single feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.embedding_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )

        # Multi-head attention for feature interactions
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )

        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.dropout = nn.Dropout(0.3)

        # Classification head (BERT-style CLS)
        self.cls = BertCLS(config)
        self.sigmoid = nn.Sigmoid()

        # Store attention weights for interpretation
        self.attention_weights = None

    def forward(self, input_data):
        """
        Forward pass of the Transformer_DAPT model.

        Args:
            input_data (torch.Tensor or tuple):
                - If tuple, expects (input_ids, attention_mask) where
                  `input_ids` is the input feature indices and `attention_mask`
                  is the binary mask indicating non-padded features.
                - If tensor, just `input_ids` is assumed.

        Returns:
            tuple: (hidden_state, output_logits, None, binary_logits)
                - hidden_state (torch.Tensor): The final hidden state after attention.
                - output_logits (torch.Tensor): Survival prediction logits.
                - None: A placeholder (kept for compatibility).
                - binary_logits (torch.Tensor): Binary classification logits.
        """
        if isinstance(input_data, tuple):
            input_ids, attention_mask = input_data
        else:
            input_ids = input_data
            attention_mask = None

        batch_size, num_features = input_ids.size()

        # Embed categorical features
        embedded = self.embeddings(input_ids)

        # Reshape to (batch_size * num_features, embedding_dim)
        reshaped = embedded.view(-1, embedded.size(-1))

        # Encode features individually
        encoded = self.feature_encoder(reshaped)

        # Reshape back to (batch_size, num_features, hidden_size)
        stacked_features = encoded.view(batch_size, num_features, -1)

        # Apply multi-head attention
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
            attn_output, attn_weights = self.feature_attention(
                stacked_features, stacked_features, stacked_features,
                key_padding_mask=key_padding_mask
            )
        else:
            attn_output, attn_weights = self.feature_attention(
                stacked_features, stacked_features, stacked_features
            )

        self.attention_weights = attn_weights

        # Flatten the attention output to feed into classification head
        hidden_state = attn_output.reshape(attn_output.size(0), -1)
        hidden_state = self.dropout(hidden_state)

        # Get survival and binary logits
        hidden_state, output_logits, binary_logits = self.cls(hidden_state)

        return hidden_state, output_logits, None, binary_logits

    def get_attention_weights(self) -> torch.Tensor:
        """
        Returns the stored attention weights from the most recent forward pass.

        Returns:
            torch.Tensor: Attention weights of shape (num_heads, batch_size, num_features, num_features).
        """
        return self.attention_weights

    def predict(self, x_input, batch_size: int = 128):
        """
        Predict survival and binary scores in batches.

        Args:
            x_input (torch.Tensor or tuple):
                - If tuple, expects (x_cat, attention_mask).
                - Otherwise, expects x_cat only.
            batch_size (int, optional): Batch size to process data. Defaults to 128.

        Returns:
            tuple:
                - surv_preds (torch.Tensor): Concatenated survival predictions.
                - binary_preds (torch.Tensor): Concatenated binary logits.
        """
        device = self.device
        self.eval()
        torch.cuda.empty_cache()

        if isinstance(x_input, tuple):
            x_cat, attention_mask = x_input
            if not isinstance(x_cat, torch.Tensor):
                x_cat = torch.tensor(x_cat, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask, dtype=torch.float)
        else:
            x_cat = x_input
            attention_mask = None

        x_cat = x_cat.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        num_samples = len(x_cat)
        surv_preds = []
        binary_preds = []

        num_batches = int(np.ceil(num_samples / batch_size))
        with torch.no_grad():
            for idx in range(num_batches):
                batch_x_cat = x_cat[idx * batch_size:(idx + 1) * batch_size]
                if attention_mask is not None:
                    batch_mask = attention_mask[idx * batch_size:(idx + 1) * batch_size]
                    batch_input = (batch_x_cat, batch_mask)
                else:
                    batch_input = batch_x_cat

                # Forward pass
                batch_pred = self.forward(batch_input)
                surv_preds.append(batch_pred[1].cpu())
                binary_preds.append(batch_pred[3].cpu())

                torch.cuda.empty_cache()

        surv_preds = torch.cat(surv_preds)
        binary_preds = torch.cat(binary_preds)
        return surv_preds, binary_preds

    def predict_binary(self, x_input, batch_size: int = 128) -> torch.Tensor:
        """
        Predict only binary classification scores in batches, with a sigmoid activation.

        Args:
            x_input (torch.Tensor or tuple):
                - If tuple, expects (x_cat, attention_mask).
                - Otherwise, expects x_cat only.
            batch_size (int, optional): Batch size to process data. Defaults to 128.

        Returns:
            torch.Tensor: Concatenated binary predictions (after sigmoid).
        """
        binary_predict_scores = []
        device = self.device
        self.eval()

        if isinstance(x_input, tuple):
            x_cat, attention_mask = x_input
            x_cat = x_cat.to(device)
            attention_mask = attention_mask.to(device)
        else:
            # Convert input to tensor if not already
            if not isinstance(x_input, torch.Tensor):
                x_cat = torch.tensor(
                    x_input.iloc[:, :self.config.max_active_features].values,
                    dtype=torch.long
                )
            else:
                x_cat = x_input[:, :self.config.max_active_features].long()
            attention_mask = torch.ones_like(x_cat, dtype=torch.float)
            x_cat = x_cat.to(device)
            attention_mask = attention_mask.to(device)

        num_samples = len(x_cat)
        num_batches = int(np.ceil(num_samples / batch_size))

        with torch.no_grad():
            for i in range(num_batches):
                batch_x = x_cat[i * batch_size:(i + 1) * batch_size]
                batch_mask = attention_mask[i * batch_size:(i + 1) * batch_size]

                # Forward pass (we only use index [3] -> binary_logits)
                batch_output = self.forward((batch_x, batch_mask))[3]
                batch_score = self.sigmoid(batch_output)
                binary_predict_scores.append(batch_score.cpu())

                torch.cuda.empty_cache()

        return torch.cat(binary_predict_scores)

    def _check_out_features(self, target=None) -> None:
        """
        Internal method to ensure the network's output features match
        the duration index or the training labels.

        Args:
            target (tuple or torch.Tensor, optional): Contains the
                survival target indices for verification.

        Raises:
            ValueError: If the output dimensions of the network do not match
                the expected dimensions (duration_index).
        """
        m_output = self.config.out_feature
        if self.duration_index is not None:
            n_grid = len(self.duration_index)
            if n_grid == m_output:
                raise ValueError(
                    "Output of `net` is one too large. Should have length "
                    f"{len(self.duration_index) - 1}"
                )
            if n_grid != (m_output + 1):
                raise ValueError(
                    "Output of `net` does not correspond with `duration_index`"
                )
        if target is not None:
            max_idx = tt.tuplefy(target).to_numpy()[0].max()
            if m_output != (max_idx + 1):
                raise ValueError(
                    f"Output of `net` is {m_output}, but data only trains {max_idx + 1} indices. "
                    f"Output of `net` should be {max_idx + 1}."
                )

    def predict_surv(
        self,
        input,
        batch_size: int = None,
        numpy: bool = None,
        eval_: bool = True,
        to_cpu: bool = False,
        num_workers: int = 0
    ):
        """
        Predict survival function estimates based on the hazard function.

        Args:
            input (torch.Tensor or tuple): Input data (features, optional mask).
            batch_size (int, optional): Size of each batch. Defaults to None.
            numpy (bool, optional): Whether to return a NumPy array.
            eval_ (bool, optional): Whether to set model to eval mode. Defaults to True.
            to_cpu (bool, optional): Whether to move predictions to CPU. Defaults to False.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.

        Returns:
            torch.Tensor or np.ndarray: Predicted survival function (cumulative product of
            the negative exponentiated hazard).
        """
        hazard = self.predict_hazard(
            input, batch_size, False, eval_, to_cpu, num_workers
        )
        surv = hazard.cumsum(1).mul(-1).exp()
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_hazard(
        self,
        input,
        batch_size: int = 128,
        numpy: bool = None,
        eval_: bool = True,
        to_cpu: bool = False,
        num_workers: int = 0
    ):
        """
        Predict the hazard function for each time index.

        Args:
            input (torch.Tensor or tuple): Input data (features, optional mask).
            batch_size (int, optional): Size of each batch. Defaults to 128.
            numpy (bool, optional): Whether to return a NumPy array.
            eval_ (bool, optional): Whether to set model to eval mode. Defaults to True.
            to_cpu (bool, optional): Whether to move predictions to CPU. Defaults to False.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.

        Returns:
            torch.Tensor or np.ndarray: Predicted hazard values.
        """
        preds = self.predict(input, batch_size)[0]
        n = preds.shape[0]
        hazard = F.softplus(preds).view(-1, 1).repeat(1, self.sub).view(n, -1).div(self.sub)
        hazard = pad_col(hazard, where='start')  # shift or pad for PyCOX
        return tt.utils.array_or_tensor(hazard, numpy, input)

    def predict_surv_df(
        self,
        input,
        batch_size: int = 128,
        eval_: bool = True,
        num_workers: int = 0
    ) -> pd.DataFrame:
        """
        Predict the survival function and return it as a Pandas DataFrame.

        Args:
            input (torch.Tensor or tuple): Input data (features, optional mask).
            batch_size (int, optional): Size of each batch. Defaults to 128.
            eval_ (bool, optional): Whether to set model to eval mode. Defaults to True.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.

        Returns:
            pd.DataFrame: Predicted survival function. Rows correspond to time indices,
            and columns to samples.
        """
        self._check_out_features()
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        index = None
        if self.duration_index is not None:
            index = make_subgrid(self.duration_index, self.sub)
        return pd.DataFrame(surv.transpose(), index=index)
