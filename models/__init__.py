"""Neural network models for task segmentation."""

import torch

from .boundary_prediction import (
    BoundaryPredictionMLP,
    ModelConfig,
    collate_dict
)
from .boundary_prediction_transformer import (
    BoundaryPredictionTransformer,
    TransformerModelConfig
)
from .encoders import TinyFASTEncoder, DinoFeatureEncoder
from .losses import (
    FocalLoss,
    SoftFocalLoss,
    FocalLossBCE,
    SoftBCELoss,
    WeightedBCELoss,
    MSEProbabilityLoss,
    get_loss_function
)


def create_model(config, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Factory function to create models based on architecture type.

    Args:
        config: Model configuration (ModelConfig or TransformerModelConfig)
        device: Device to place model on

    Returns:
        Model instance on specified device

    Examples:
        >>> # MLP model (default)
        >>> config = ModelConfig()
        >>> model = create_model(config)

        >>> # Transformer model
        >>> config = TransformerModelConfig()
        >>> model = create_model(config)

        >>> # From dict with architecture flag
        >>> from omegaconf import OmegaConf
        >>> config_dict = OmegaConf.create({'architecture': 'cross_modal_transformer', ...})
        >>> model = create_model(config_dict)
    """
    # Check architecture type
    architecture = getattr(config, 'architecture', 'mlp')

    if architecture == 'cross_modal_transformer':
        # Create transformer config if needed
        if not isinstance(config, TransformerModelConfig):
            # Convert dict-like config to TransformerModelConfig
            config_dict = dict(config.items()) if hasattr(config, 'items') else vars(config)
            config = TransformerModelConfig(**config_dict)

        model = BoundaryPredictionTransformer(config=config, device=device)
    else:
        # Default to MLP
        if not isinstance(config, ModelConfig):
            # Convert to ModelConfig, filtering transformer-specific keys
            model_config_keys = {
                'vocab_size', 'token_embed_dim', 'dino_feature_dim',
                'hidden_dims', 'prediction_window_size',
                'transformer_dropout', 'embedding_dropout',
                'dino_encoder_dropout', 'mlp_dropout', 'feature_noise_std',
                'use_proprio', 'use_dino'
            }
            config_dict = dict(config.items()) if hasattr(config, 'items') else vars(config)
            filtered_config = {k: v for k, v in config_dict.items() if k in model_config_keys}
            config = ModelConfig(**filtered_config)

        model = BoundaryPredictionMLP(config=config, device=device)

    model = model.to(device)
    return model


__all__ = [
    'BoundaryPredictionMLP',
    'BoundaryPredictionTransformer',
    'ModelConfig',
    'TransformerModelConfig',
    'create_model',
    'collate_dict',
    'TinyFASTEncoder',
    'DinoFeatureEncoder',
    'FocalLoss',
    'SoftFocalLoss',
    'FocalLossBCE',
    'SoftBCELoss',
    'WeightedBCELoss',
    'MSEProbabilityLoss',
    'get_loss_function',
]
