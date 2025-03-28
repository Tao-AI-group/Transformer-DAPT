import torch
from torch import nn
import torch.nn.functional as F


class DenseVanillaBlock(nn.Module):
    """
    A basic dense block that applies a linear transformation, followed by an
    activation, optional batch normalization, and optional dropout.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
        batch_norm (bool, optional): If True, applies batch normalization. Default: True.
        dropout (float, optional): Probability of an element to be zeroed. Default: 0.
        activation (Callable, optional): The activation function to use. Default: nn.ReLU.
        w_init_ (Callable, optional): Weight initialization function. Default: kaiming_normal_ for 'relu'.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        batch_norm: bool = True,
        dropout: float = 0.0,
        activation=nn.ReLU,
        w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        if w_init_:
            w_init_(self.linear.weight.data)

        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DenseVanillaBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor after applying linear transformation,
            activation, optional batch normalization, and optional dropout.
        """
        x = self.activation(self.linear(x))
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class BertCLS(nn.Module):
    """
    A classification head for BERT-like models that first projects hidden states
    through an MLP and then produces two outputs:
    1) A multi-dimensional output (`output_logits`).
    2) A single-neuron binary output (`binary_logits`).

    Args:
        config (object): Configuration object that provides the following attributes:
            - hidden_size (int): The size of the hidden representation per feature.
            - num_feature (int): The number of features used to flatten the hidden representation.
            - intermediate_size (int): The size of the dense layer before the final output.
            - hidden_dropout_prob (float): Dropout probability.
            - out_feature (int): The number of output neurons (for survival logits or similar).
    """
    def __init__(self, config):
        super().__init__()
        w_init = lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")

        # Build a sequential network
        net = []
        net.append(
            DenseVanillaBlock(
                config.hidden_size * config.num_feature,
                config.intermediate_size,
                batch_norm=True,
                dropout=config.hidden_dropout_prob,
                activation=nn.ReLU,
                w_init_=w_init
            )
        )

        self.net = nn.Sequential(*net)
        self.output = nn.Linear(config.intermediate_size, config.out_feature)
        self.dense_1_neuron = nn.Linear(config.intermediate_size, 1)

    def forward(self, hidden_states: torch.Tensor):
        """
        Forward pass of the BertCLS head.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, num_features, hidden_size)
                or a flattened representation. Typically comes from a preceding transformer module.

        Returns:
            tuple:
                - hidden_states (torch.Tensor): The intermediate hidden states after MLP.
                - output_logits (torch.Tensor): The main output logits (e.g., survival logits).
                - binary_logits (torch.Tensor): The single-neuron output logits (e.g., for binary classification).
        """
        # Flatten hidden states from (batch_size, num_features, hidden_size) to (batch_size, -1)
        hidden_states = hidden_states.flatten(start_dim=1)

        hidden_states = self.net(hidden_states)
        output_logits = self.output(hidden_states)
        binary_logits = self.dense_1_neuron(hidden_states)

        return hidden_states, output_logits, binary_logits


class BertCLSMulti(nn.Module):
    """
    A classification head that can produce multiple outputs (one per event).
    Similar to BertCLS, but with multiple linear output layers in a list.

    Args:
        config (object): Configuration object that provides the following attributes:
            - hidden_size (int): The size of the hidden representation per feature.
            - num_feature (int): The number of features used to flatten the hidden representation.
            - intermediate_size (int): The size of the dense layer before the final output.
            - hidden_dropout_prob (float): Dropout probability.
            - num_event (int): Number of distinct events (outputs) to predict.
            - out_feature (int): The size of each individual event output layer.
    """
    def __init__(self, config):
        super().__init__()
        w_init = lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")

        # Build the shared MLP
        net = []
        net.append(
            DenseVanillaBlock(
                config.hidden_size * config.num_feature,
                config.intermediate_size,
                batch_norm=True,
                dropout=config.hidden_dropout_prob,
                activation=nn.ReLU,
                w_init_=w_init
            )
        )
        self.net = nn.Sequential(*net)

        # Build one output layer per event
        net_out = []
        for _ in range(config.num_event):
            net_out.append(nn.Linear(config.intermediate_size, config.out_feature))
        self.net_out = nn.ModuleList(net_out)

    def forward(self, hidden_states: torch.Tensor, event: int = 0) -> torch.Tensor:
        """
        Forward pass of the BertCLSMulti head for a specific event.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, num_features, hidden_size)
                or a flattened representation.
            event (int, optional): Index of the event to predict. Defaults to 0.

        Returns:
            torch.Tensor: The logits corresponding to the specified event.
        """
        # Flatten hidden states if needed
        hidden_states = hidden_states.flatten(start_dim=1)
        hidden_states = self.net(hidden_states)
        output = self.net_out[event](hidden_states)
        return output
