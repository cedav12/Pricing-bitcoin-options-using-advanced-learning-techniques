import torch
import torch.nn as nn
from typing import List, Dict, Any

class ModularMLP(nn.Module):
    """
    A generic MLP architecture for tabular option pricing.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "gelu",
        output_activation: str = "linear"
    ):
        super().__init__()
        self.input_dim = input_dim
        
        # Determine activation function
        activation_dict = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU
        }
        act_layer = activation_dict.get(activation.lower(), nn.GELU)

        layers = []
        current_dim = input_dim
        
        # Build hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(act_layer())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            current_dim = h_dim
            
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        
        # Optional output activation
        if output_activation.lower() == "softplus":
            layers.append(nn.Softplus())
        elif output_activation.lower() == "relu":
            layers.append(nn.ReLU())
            
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_model_from_config(input_dim: int, config: Dict[str, Any]) -> nn.Module:
    """Factory method to instantiate a model from a generic config dictionary."""
    model_name = config.get("name", "mlp").lower()
    
    if model_name == "mlp":
        return ModularMLP(
            input_dim=input_dim,
            hidden_dims=config.get("hidden_dims", [128, 128, 64]),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "gelu"),
            output_activation=config.get("output_activation", "linear")
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
